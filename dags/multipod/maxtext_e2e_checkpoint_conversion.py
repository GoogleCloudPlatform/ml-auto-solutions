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

"""MaxText E2E Checkpoint Conversion DAG (Stage 1).

Serves as the shared prerequisite pipeline for MaxText E2E testing:
- Converts Hugging Face checkpoints to MaxText format on CPU clusters.
- Saves checkpoints to GCS (gs://runner-maxtext-logs/<model>/to_maxtext/...).
- Downstream training DAGs (TPU pre-training, TPU post-training and GPU
  pre-training) listen to this DAG via ExternalTaskSensor and begin training
  as each model completes.
"""
import datetime

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.operators.python import get_current_context
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import GkeClusters
from dags.multipod.configs import gke_config

HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)


def _get_test_scope() -> str:
  """Resolves the requested scope, defaulting to running every suite."""
  context = get_current_context()
  dag_run = context["dag_run"]
  params = context["params"]
  conf = (dag_run.conf if dag_run else {}) or {}
  return conf.get("test_scope") or (params or {}).get("test_scope") or "all"


@task.short_circuit
def post_training_enabled() -> bool:
  """Skips post-training-only conversions unless TPU post-training will run."""
  return _get_test_scope() in ("all", "tpu", "tpu-post-training")


@task.short_circuit
def gpu_pre_training_enabled() -> bool:
  """Skips GPU-only conversions unless GPU pre-training will run."""
  return _get_test_scope() in ("all", "gpu-pre-training")


# TODO(KevinYenky): Use the tests/end_to_end/cpu/ paths directly and remove this
# helper once all images the E2E DAGs run include AI-Hypercomputer/maxtext#6453.
def _e2e_script(rel_path: str) -> str:
  """Returns a shell command that runs a MaxText checkpoint conversion script.

  MaxText is moving the conversion scripts from tests/end_to_end/tpu/ to
  tests/end_to_end/cpu/. Until every image has the move, run the tpu/ copy
  when the image still has it and the cpu/ copy otherwise.
  """
  tpu_path = f"tests/end_to_end/tpu/{rel_path}"
  cpu_path = f"tests/end_to_end/cpu/{rel_path}"
  return (
      f'bash "$(if [ -f {tpu_path} ]; then echo {tpu_path};'
      f' else echo {cpu_path}; fi)"'
  )


with models.DAG(
    dag_id="maxtext_e2e_checkpoint_conversion",
    schedule=None,
    tags=[
        "maxtext",
        "conversion",
        "checkpoint",
        "CPU",
    ],
    start_date=datetime.datetime(2026, 6, 10),
    catchup=False,
    params={
        "docker_image": Param(
            type="string",
            description="Docker image URI for the candidate to test",
        ),
        "run_name": Param(
            default="",
            type="string",
            description=(
                "Unique shared run name for checkpoints (defaults to"
                " conv-{{ ts_nodash }})"
            ),
        ),
        "test_scope": Param(
            default="all",
            type="string",
            enum=[
                "all",
                "tpu",
                "tpu-pre-training",
                "tpu-post-training",
                "gpu-pre-training",
            ],
            description=(
                "Which Stage 2 suites will consume the converted checkpoints."
                " Post-training-only models are skipped unless TPU"
                " post-training runs, and GPU-only models unless GPU"
                " pre-training runs."
            ),
        ),
    },
) as dag:
  # pylint: disable=line-too-long
  test_models = {
      "gemma3-4b": {
          "to_maxtext": _e2e_script("gemma3/4b/test_gemma3_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER,
      },
      "gemma4-26b": {
          "to_maxtext": _e2e_script("gemma4/26b/test_gemma4_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER,
      },
      "llama3_1-70b": {
          "to_maxtext": _e2e_script("llama3.1/70b/test_llama3.1_70b_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_128_CLUSTER,
      },
      "qwen3-30b": {
          "to_maxtext": _e2e_script("qwen3/30b/test_qwen3_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_128_CLUSTER,
      },
      "qwen3-4b": {
          "to_maxtext": _e2e_script("qwen3/4b/test_qwen3_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER,
      },
      "qwen3-vl-2b": {
          "to_maxtext": _e2e_script("qwen3/vl_2b/test_qwen3_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER,
          "post_training_only": True,
      },
      "gpt-oss-20b": {
          "to_maxtext": _e2e_script("gpt_oss/20b/test_gpt_oss_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER,
      },
      "qwen3_5-35b": {
          "to_maxtext": _e2e_script("qwen3.5/35b/test_qwen3.5_to_mt.sh"),
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_128_CLUSTER,
          "post_training_only": True,
      },
      "llama3_1-8b": {
          "to_maxtext": "bash tests/end_to_end/cpu/llama3.1/8b/test_llama3.1_8b_to_mt.sh",
          "cpu_cluster": GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER,
          "gpu_pre_training_only": True,
      },
  }
  # pylint: enable=line-too-long

  for model, config in test_models.items():
    with TaskGroup(group_id=model) as model_group:
      run_name = (
          "{{ params.run_name if params.run_name else 'conv-' ~ ts_nodash }}"
      )

      to_maxtext_cmd = config["to_maxtext"]
      convert_to_maxtext_cmd = (
          f"export HF_TOKEN={HF_TOKEN}",
          'export HF_HOME="/mnt/shm/hf_cache"',
          f"{to_maxtext_cmd} {run_name}",
      )

      convert_to_maxtext_task = (
          gke_config.get_maxtext_cpu_end_to_end_gke_config(
              time_out_in_min=120,
              test_name="to-mt",
              run_model_cmds=convert_to_maxtext_cmd,
              docker_image="{{ params.docker_image }}",
              cluster=config["cpu_cluster"],
              test_owner=test_owner.JACKY_F,
              priority="very-high",
              use_gcluster=True,
              mounts="/dev/shm;/mnt/shm;rw",
          ).run(skip_post_process=True)
      )

      if config.get("post_training_only"):
        chain(
            post_training_enabled(),
            convert_to_maxtext_task,
        )
      elif config.get("gpu_pre_training_only"):
        chain(
            gpu_pre_training_enabled(),
            convert_to_maxtext_task,
        )
