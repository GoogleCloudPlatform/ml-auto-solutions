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

"""MaxText E2E TPU Checkpoint Conversion DAG (Stage 1).

Serves as the shared prerequisite pipeline for MaxText E2E testing:
- Converts Hugging Face checkpoints to MaxText format on TPU v5p-8 slices.
- Saves checkpoints to GCS (gs://runner-maxtext-logs/<model>/to_maxtext/...).
- Downstream training DAGs (pre_training and post_training) listen to this
  DAG via ExternalTaskSensor and begin training as each model completes.
"""
import datetime
from airflow import models
from airflow.models.param import Param
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import GkeClusters
from dags.multipod.configs import gke_config

HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)

with models.DAG(
    dag_id="maxtext_e2e_tpu_checkpoint_conversion",
    schedule=None,
    tags=[
        "maxtext",
        "conversion",
        "checkpoint",
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
            description=(
                "Unique shared run name for checkpoints (defaults to"
                " conv-{{ ts_nodash }})"
            ),
        ),
    },
) as dag:
  # pylint: disable=line-too-long
  test_models = {
      "gemma3-4b": {
          "to_maxtext": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh",
      },
      "gemma4-26b": {
          "to_maxtext": "bash tests/end_to_end/tpu/gemma4/26b/test_gemma4_to_mt.sh",
      },
      "llama3_1-70b": {
          "to_maxtext": "bash tests/end_to_end/tpu/llama3.1/70b/test_llama3.1_70b_to_mt.sh",
      },
      "qwen3-30b": {
          "to_maxtext": "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3_to_mt.sh",
      },
      "qwen3-vl-2b": {
          "to_maxtext": "bash tests/end_to_end/tpu/qwen3/vl_2b/test_qwen3_to_mt.sh",
      },
      "gpt-oss-20b": {
          "to_maxtext": "bash tests/end_to_end/tpu/gpt_oss/20b/test_gpt_oss_to_mt.sh",
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
          'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
          f"{to_maxtext_cmd} {run_name}",
      )

      convert_to_maxtext_task = gke_config.get_gke_config(
          time_out_in_min=120,
          test_name="to-mt",
          run_model_cmds=convert_to_maxtext_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=GkeClusters.TPU_V5P_BODABORG_NAP_CLUSTER,
          test_owner=test_owner.JACKY_F,
          priority="medium",
          use_gcluster=True,
          mounts="/dev/shm;/mnt/shm;rw",
      ).run(skip_post_process=True)
