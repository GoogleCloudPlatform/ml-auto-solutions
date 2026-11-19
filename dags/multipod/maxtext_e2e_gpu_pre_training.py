# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxText E2E GPU Pre-Training Tests DAG

Executes end-to-end MaxText pre-training test workloads on GKE GPU clusters:
- Waits for model conversion in maxtext_e2e_checkpoint_conversion
  via ExternalTaskSensor.
- Runs one multi-node Cluster Toolkit workload per model that decodes the
  converted checkpoint, runs two steps of pre-training that save checkpoints
  to GCS, and decodes the trained checkpoint.
- Converts trained checkpoints back to Hugging Face format on CPU clusters
  to verify weight fidelity.
"""
import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.session import provide_session
from airflow.utils.task_group import TaskGroup

from dags.common import test_owner
from dags.common.goodput_utils import check_workload_goodput
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import DockerImage, GkeClusters
from dags.multipod.configs import gke_config

HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)

GPU_CLUSTER = GkeClusters.GPU_A3M_16_CLUSTER

# The MaxText GPU E2E scripts are written for, and were validated on, two nodes.
GPU_NUM_NODES = 2

# Cluster Toolkit gives every JobSet exclusive use of a node pool, and all of
# GPU_CLUSTER's GPU nodes are in one pool, so GPU workloads run one at a time
# and a workload waits while another JobSet runs on the pool.
GPU_WORKLOAD_PROVISION_TIMEOUT = datetime.timedelta(hours=4)

# The NCCL that JAX uses in the MaxText GPU image.
_PIP_NCCL = (
    "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2"
)

# Settings for GPU_CLUSTER and the current MaxText GPU image:
# - The a3-megagpu-8g nodes run without the GPUDirect-TCPXO receive daemon and
#   settings, so keep NCCL off GKE's TCPXO net and tuner plugins; NCCL then
#   uses its socket transport.
# - The image's TensorFlow exports an older NCCL whose symbols capture the calls
#   JAX makes into its own NCCL (`corrupted comm object detected`), so preload
#   the NCCL that JAX uses.
GPU_ENV_CMDS = (
    "export NCCL_NET_PLUGIN=none",
    "export NCCL_TUNER_PLUGIN=none",
    f"if [ -f {_PIP_NCCL} ]; then export LD_PRELOAD={_PIP_NCCL}; fi",
)
# - The cluster's Kueue (v0.13.3) is older than the v0.15.2 `gcluster job
#   submit` needs, and submit then offers to re-install Kueue, which deletes
#   every queued and suspended workload. Until Kueue is upgraded, let gcluster
#   only render the JobSets and create them with kubectl.
GPU_APPLY_WITH_KUBECTL = True
# - The scripts were validated without GKE's topology-aware scheduling, whose
#   scheduling gate holds the pods until GKE lifts it, so use the cluster's
#   default scheduler.
GPU_GKE_SCHEDULER = None


class ExternalTaskSensorWithBypass(ExternalTaskSensor):
  """Sensor that passes immediately if wait_for_conversion param is False."""

  @provide_session
  def poke(self, context, session=None):
    params = context.get("params") or {}
    dag_run = context.get("dag_run")
    conf = (dag_run.conf if dag_run else {}) or {}
    if not params.get("wait_for_conversion", True) or not conf.get(
        "wait_for_conversion", True
    ):
      self.log.info("Bypassing conversion sensor: wait_for_conversion is False")
      return True
    return super().poke(context, session=session)


with models.DAG(
    dag_id="maxtext_e2e_gpu_pre_training",
    schedule=None,
    tags=[
        "maxtext",
        "pre-training",
        "GPU",
    ],
    start_date=datetime.datetime(2026, 11, 19),
    catchup=False,
    params={
        "docker_image": Param(
            default=DockerImage.MAXTEXT_GPU_JAX_STABLE.value,
            type="string",
            description="GPU Docker image URI for the candidate to test",
        ),
        "run_name": Param(
            default="",
            type="string",
            description=(
                "Shared run name of the Stage 1 checkpoints (e.g."
                " e2e-<github_run_id>). Training and to-HF outputs go under"
                " gpu-<run_name>, apart from the TPU pre-training outputs."
            ),
        ),
        "wait_for_conversion": Param(
            default=True,
            type="boolean",
            description=(
                "Whether to wait for Stage 1 conversion DAG via sensor. Set"
                " False when running standalone with pre-existing checkpoints."
            ),
        ),
    },
) as dag:
  # Each GPU script decodes the unscanned Stage 1 checkpoint, trains for two
  # steps from the scanned or the unscanned one (see the script), which saves
  # checkpoints 0 and 1 under train/<run name>, and decodes checkpoint 1.
  # `to_hf_flags` must match the layout the script trains in.
  # pylint: disable=line-too-long
  test_models = {
      "llama3_1-8b": {
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER,
          "training": {
              "command": "bash tests/end_to_end/gpu/llama3.1/8b/test_llama3.1_8b.sh",
              "maxtext_ckpt_path": "gs://runner-maxtext-logs/llama3.1-8b/train/{run_name}/checkpoints/1/items",
          },
          "to_huggingface": "bash tests/end_to_end/cpu/llama3.1/8b/test_llama3.1_8b_to_hf.sh",
      },
  }
  # pylint: enable=line-too-long

  for model, test_config in test_models.items():
    with TaskGroup(group_id=model):
      run_name = (
          "{{ params.run_name if params.run_name else 'pre-' ~ ts_nodash }}"
      )
      # maxtext_e2e_tpu_pre_training trains under train/<run_name>, so train
      # and convert to HF under a GPU-specific name.
      gpu_run_name = f"gpu-{run_name}"
      goodput_job_name = f"{model}-gpu-pre-{run_name}"

      model_path = test_config["training"]["maxtext_ckpt_path"].format(
          run_name=gpu_run_name
      )
      base_output_dir = model_path.split("/checkpoints/", maxsplit=1)[0]
      cleanup_cmd = (
          f"if gsutil ls {base_output_dir} >/dev/null 2>&1; then "
          f'echo "Retry detected (directory exists). Cleaning up..."; '
          f"gsutil -m rm -rf {base_output_dir} || true; "
          f"fi"
      )

      training_script = test_config["training"]["command"]
      training_cmd = (
          f"export HF_TOKEN={HF_TOKEN}",
          *GPU_ENV_CMDS,
          "export M_ENABLE_GOODPUT_RECORDING=true",
          "export M_ENABLE_CHECKPOINT_CLOUD_LOGGER=true",
          f"export M_GOODPUT_JOB_NAME={goodput_job_name}",
          cleanup_cmd,
          # Arguments: the Stage 1 run name to read, the run name to train in.
          f"{training_script} {run_name} {gpu_run_name}",
      )
      training_task = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
          time_out_in_min=test_config.get("time_out_in_min", 60),
          test_name="pre",
          run_model_cmds=training_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=GPU_CLUSTER,
          test_owner=test_owner.KEVIN_Y,
          num_slices=GPU_NUM_NODES,
          restart_on_exit_codes=[137, 143],
          use_gcluster=True,
          priority="medium",
          max_restart=3,
          workload_provision_timeout=GPU_WORKLOAD_PROVISION_TIMEOUT,
          gke_scheduler=GPU_GKE_SCHEDULER,
          apply_with_kubectl=GPU_APPLY_WITH_KUBECTL,
      ).run(skip_post_process=True)

      check_goodput_task = check_workload_goodput(
          workload_id=goodput_job_name,
          project_id=GPU_CLUSTER.project,
          using_pathways=False,
      )

      to_hf_flags = test_config.get("to_hf_flags", "")
      to_hf_script = test_config["to_huggingface"]

      convert_to_huggingface_cmd = (
          f"export HF_TOKEN={HF_TOKEN}",
          'export HF_HOME="/mnt/shm/hf_cache"',
          f"{to_hf_script} {gpu_run_name} {model_path} {to_hf_flags}",
      )
      convert_to_huggingface_task = (
          gke_config.get_maxtext_cpu_end_to_end_gke_config(
              time_out_in_min=90,
              test_name="to-hf",
              run_model_cmds=convert_to_huggingface_cmd,
              docker_image="{{ params.docker_image }}",
              cluster=test_config["cpu_cluster"],
              test_owner=test_owner.KEVIN_Y,
              priority="very-high",
              use_gcluster=True,
              mounts="/dev/shm;/mnt/shm;rw",
          ).run(skip_post_process=True)
      )

      wait_for_conversion = ExternalTaskSensorWithBypass(
          task_id="wait_for_conversion",
          external_dag_id="maxtext_e2e_checkpoint_conversion",
          external_task_group_id=model,
          mode="reschedule",
          poke_interval=60,
          timeout=10800,
          allowed_states=["success"],
          failed_states=["failed", "upstream_failed"],
      )

      chain(
          wait_for_conversion,
          training_task,
          [check_goodput_task, convert_to_huggingface_task],
      )
