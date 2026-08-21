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

"""
MaxText E2E TPU Pre-Training Tests DAG (Stage 2).

Executes end-to-end MaxText pre-training test workloads on Cloud TPU:
- Waits for model conversion in maxtext_e2e_tpu_checkpoint_conversion via ExternalTaskSensor.
- Runs pre-training on dedicated TPU v5p topologies configured per model architecture.
- Converts trained checkpoints back to Hugging Face format to verify weight fidelity.
"""
import datetime
from airflow import models
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.session import provide_session
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import xpk_gke_config as gke_config

HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)


class ExternalTaskSensorWithBypass(ExternalTaskSensor):
  """ExternalTaskSensor that passes immediately if wait_for_conversion param is False."""

  @provide_session
  def poke(self, context, session=None):
    if not context.get("params", {}).get("wait_for_conversion", True):
      self.log.info("Bypassing conversion sensor: wait_for_conversion is False")
      return True
    return super().poke(context, session=session)


with models.DAG(
    dag_id="maxtext_e2e_tpu_pre_training",
    schedule=None,
    tags=[
        "maxtext",
        "pre-training",
        "TPU",
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
            description="Shared run name for checkpoints (e.g. conv-20260813T123008)",
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
  # pylint: disable=line-too-long
  test_models = {
      "gemma3-4b": {
          "core_count": 8,
          "training": {
              "command": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3.sh",
              "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma3-4b/train/{run_name}/checkpoints/1/items",
          },
          "to_huggingface": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_hf.sh",
      },
      "gemma4-26b": {
          "core_count": 32,
          "training": {
              "command": "bash tests/end_to_end/tpu/gemma4/26b/test_gemma4.sh",
              "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma4-26b/train/{run_name}/checkpoints/1/items",
          },
          "to_huggingface": "bash tests/end_to_end/tpu/gemma4/26b/test_gemma4_to_hf.sh",
      },
      "llama3_1-70b": {
          "core_count": 128,
          "training": {
              "command": "bash tests/end_to_end/tpu/llama3.1/70b/test_llama3.1_70b.sh",
              "maxtext_ckpt_path": "gs://runner-maxtext-logs/llama3.1-70b/train/{run_name}/checkpoints/1/items",
          },
          "to_huggingface": "bash tests/end_to_end/tpu/llama3.1/70b/test_llama3.1_70b_to_hf.sh",
      },
      "qwen3-30b": {
          "core_count": 32,
          "training": {
              "command": "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3.sh",
              "maxtext_ckpt_path": "gs://runner-maxtext-logs/qwen3-30b-a3b-base/train/{run_name}/checkpoints/1/items",
          },
          "to_huggingface": "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3_to_hf.sh",
          "to_hf_flags": "true",
      },
      "gpt-oss-20b": {
          "core_count": 32,
          "training": {
              "command": "bash tests/end_to_end/tpu/gpt_oss/20b/test_gpt_oss.sh",
              "maxtext_ckpt_path": "gs://runner-maxtext-logs/gpt-oss-20b/train/{run_name}/checkpoints/1/items",
          },
          "to_huggingface": "bash tests/end_to_end/tpu/gpt_oss/20b/test_gpt_oss_to_hf.sh",
      },
  }
  # pylint: enable=line-too-long

  for model, test_config in test_models.items():
    with TaskGroup(group_id=model) as model_group:
      run_name = (
          "{{ params.run_name if params.run_name else 'pre-' ~ ts_nodash }}"
      )

      training_cmd = (f"export HF_TOKEN={HF_TOKEN}",) + (
          f"{test_config['training']['command']} {run_name}",
      )
      training_core_count = test_config.get("core_count", 8)
      training_task = gke_config.get_gke_config(
          time_out_in_min=60,
          test_name="pre",
          run_model_cmds=training_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=XpkClusters.TPU_V5P_MLPERF_CLUSTER.override(
              core_count=training_core_count
          ),
          test_owner=test_owner.SURBHI_J,
          priority="very-high",
      ).run(skip_post_process=True)

      model_path = test_config["training"]["maxtext_ckpt_path"].format(
          run_name=run_name
      )
      to_hf_flags = test_config.get("to_hf_flags", "")

      convert_to_huggingface_cmd = (
          f"export HF_TOKEN={HF_TOKEN}",
          'export HF_HOME="/dev/shm/hf_cache"',
          'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
      ) + (
          f"{test_config['to_huggingface']} "
          f"{run_name} {model_path} {to_hf_flags}",
      )
      convert_to_huggingface_task = gke_config.get_gke_config(
          time_out_in_min=90,
          test_name="to-hf",
          run_model_cmds=convert_to_huggingface_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=XpkClusters.TPU_V5P_MLPERF_CLUSTER,
          test_owner=test_owner.SURBHI_J,
          priority="very-high",
      ).run(skip_post_process=True)

      wait_for_conversion = ExternalTaskSensorWithBypass(
          task_id="wait_for_conversion",
          external_dag_id="maxtext_e2e_tpu_checkpoint_conversion",
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
          convert_to_huggingface_task,
      )
