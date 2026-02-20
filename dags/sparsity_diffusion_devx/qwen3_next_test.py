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

"""A standalone DAG to test the Qwen3-Next 80B script with a custom Docker image."""

import datetime
from airflow import models
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config

# Retrieve the HF_TOKEN from Airflow variables
HF_TOKEN = models.Variable.get("HF_TOKEN", None)

with models.DAG(
    dag_id="qwen3_next_custom_image_test",
    schedule=None,  # Set to None so it only runs when manually triggered
    tags=[
        "maxtext",
        "tpu",
        "qwen3",
        "v5p-128",
    ],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
  # Your specified custom docker image
  custom_docker_image = "gcr.io/tpu-prod-env-multipod/maxtext_stable_stack_candidate:rbierneni-qwen-test"

  # Single unchained test configuration
  test_qwen3_next = gke_config.get_gke_config(
      time_out_in_min=90,
      test_name="maxtext_qwen3_next_80b_test",
      run_model_cmds=(
          f"export HF_TOKEN={HF_TOKEN}; export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash tests/end_to_end/tpu/qwen/next/qwen3-next-80b-a3b/2_test_qwen3_next_80b_a3b.sh",
      ),
      docker_image=custom_docker_image,
      test_owner=test_owner.ROHAN_B,  # Update the owner if necessary
      cluster=XpkClusters.TPU_V5P_128_CLUSTER,
  ).run()

  # No chained dependencies required since there is only one task
  test_qwen3_next
