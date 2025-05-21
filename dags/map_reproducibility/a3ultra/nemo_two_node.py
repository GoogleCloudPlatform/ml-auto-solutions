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

"""DAGs to run hypercomputer recipes"""

import datetime

from airflow import models
from dags import composer_env

from dags.map_reproducibility.utils.common_utils import run_workload


MODEL_ID = "mixtral-8x7b"
METRICS_MODEL_ID = "mixtral-7b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3ultra"
FRAMEWORK = "nemo"
WORKLOAD_LAUNCHER = "nemo-10-launcher.sh"

SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None

SOFTWARE_ID = "pytorch_nemo"
KUEUE_NAME = "a3-ultra"
NUM_GPUS = 16
NUM_STEPS = 1

with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_two_node_{FRAMEWORK}",
    schedule=SCHEDULED_TIME,
    tags=[
        "reproducibility",
        "experimental",
        "xlml",
        "regressiontests",
        "a3ultra",
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  run_workload(
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      kueue_name=KUEUE_NAME,
      metrics_model_id=METRICS_MODEL_ID,
      num_gpus=NUM_GPUS,
      num_steps=NUM_STEPS,
      config_model_name=f"{MODEL_ID}-256gpus-a3u-{PRECISION}.yaml",
      workload_launcher=WORKLOAD_LAUNCHER,
  )
