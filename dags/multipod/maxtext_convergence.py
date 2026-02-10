# Copyright 2024 Google LLC
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
A DAG to run MaxText convergence tests for both bf16 and int8.
"""
import datetime
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config
from xlml.apis import metric_config

# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "15 15 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_convergence",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "mlscale_devx",
        "TPU",
        "v6e-256",
    ],
    start_date=datetime.datetime(2024, 3, 1),
    catchup=False,
    concurrency=2,
) as dag:
  current_time = datetime.datetime.now()
  current_date = current_time.strftime("%Y-%m-%d")
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext/stable/automated/{current_date}"
  )
  dataset_path = gcs_bucket.MAXTEXT_DIR

  steps = 10200  # Half Chinchilla
  loss_threshold = 2.7
  per_device_batch_size = 2.0  # 256 chips * 2 pdb = 512 gbs.
  eval_interval = 5000

  base_convergence_command = (
      "bash tests/end_to_end/tpu/test_convergence_1b_params.sh"
      f" OUTPUT_PATH={base_output_directory} DATASET_PATH={dataset_path}"
      f" LOSS_THRESHOLD={loss_threshold} STEPS={steps}"
      f" EVAL_INTERVAL={eval_interval}"
      f" PER_DEVICE_BATCH_SIZE={per_device_batch_size}"
  )
  convergence_tests = {
      "maxtext-convergence-bf16": ((base_convergence_command),),
      "maxtext-convergence-int8": (
          (f"export M_QUANTIZATION=int8; {base_convergence_command}"),
      ),
      "maxtext-convergence-subset-hosts": (
          (
              "export M_EXPANSION_FACTOR_REAL_DATA=2; "
              + base_convergence_command
          ),
      ),
      "maxtext-convergence-grain": (
          (f"{base_convergence_command} DATASET_TYPE=grain"),
      ),
      "maxtext-convergence-hf": (
          (f"{base_convergence_command} DATASET_TYPE=hf"),
      ),
  }

  # Tests that can be run in parallel to reduce execution time.
  parallel_test_names = ["maxtext-convergence-grain"]

  sequential_tests = []
  for test_name, run_command in convergence_tests.items():
    # The grain dataset takes longer to run, so we give it a longer timeout. The other tests are expected to complete within 5 hours.
    timeout_in_min = 360 if test_name == "maxtext-convergence-grain" else 300

    test_task = gke_config.get_gke_config(
        cluster=XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
        time_out_in_min=timeout_in_min,
        test_name=test_name,
        run_model_cmds=run_command,
        docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
        test_owner=test_owner.MATT_D,
        base_output_directory=base_output_directory,
        metric_aggregation_strategy=metric_config.AggregationStrategy.LAST,
    ).run_with_run_name_generation()
    if test_name not in parallel_test_names:
      sequential_tests.append(test_task)

  for i in range(len(sequential_tests) - 1):
    # pylint: disable-next=pointless-statement
    sequential_tests[i] >> sequential_tests[i + 1]
