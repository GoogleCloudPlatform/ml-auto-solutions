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
A DAG to run MaxText E2E TPU Pre-Training tests.
"""
import datetime
from airflow import models
from airflow.models.param import Param
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config

HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)

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
    },
) as dag:
  test_models = {
      "gemma3-4b": {
          "checkpoint_conversion": {
              "to_maxtext": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh",
              "to_huggingface": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_hf.sh",
          },
          "training": {
              "command": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3.sh",
              "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma3-4b/train/{run_name}/checkpoints/4/items",
          },
      },
  }

  for model, test_config in test_models.items():
    with TaskGroup(group_id=model) as model_group:
      run_name = "pre-{{ ts_nodash }}"

      convert_to_maxtext_cmd = (f"export HF_TOKEN={HF_TOKEN}",) + (
          f"{test_config['checkpoint_conversion']['to_maxtext']} {run_name}",
      )
      convert_to_maxtext_task = gke_config.get_gke_config(
          time_out_in_min=60,
          test_name="convert-to-maxtext",
          run_model_cmds=convert_to_maxtext_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=XpkClusters.TPU_V5P_8_CLUSTER_V2,
          test_owner=test_owner.SURBHI_J,
      ).run(skip_post_process=True)

      training_cmd = (f"export HF_TOKEN={HF_TOKEN}",) + (
          f"{test_config['training']['command']} {run_name}",
      )
      training_task = gke_config.get_gke_config(
          time_out_in_min=60,
          test_name="training",
          run_model_cmds=training_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          test_owner=test_owner.SURBHI_J,
      ).run(skip_post_process=True)

      model_path = test_config["training"]["maxtext_ckpt_path"].format(
          run_name=run_name
      )
      convert_to_huggingface_cmd = (f"export HF_TOKEN={HF_TOKEN}",) + (
          f"{test_config['checkpoint_conversion']['to_huggingface']} {run_name} {model_path}",
      )
      convert_to_huggingface_task = gke_config.get_gke_config(
          time_out_in_min=60,
          test_name="convert-to-huggingface",
          run_model_cmds=convert_to_huggingface_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=XpkClusters.TPU_V5P_8_CLUSTER_V2,
          test_owner=test_owner.SURBHI_J,
      ).run(skip_post_process=True)

      convert_to_maxtext_task >> training_task >> convert_to_huggingface_task
