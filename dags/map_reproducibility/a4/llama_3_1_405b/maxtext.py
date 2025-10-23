# Copyright 2025 Google LLC
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

"""DAGs to run Aotc reproducibility benchmarks."""

import datetime
from airflow import models
from dags import composer_env

from dags.map_reproducibility.utils.common_utils import get_scheduled_time, run_workload_with_quarantine
from dags.map_reproducibility.utils.common_utils import run_workload

MODEL_ID = "llama3-1-405b"
METRICS_MODEL_ID = "llama3.1-405b"
PRECISION = "fp8"
HYPERCOMPUTER = "a4"
FRAMEWORK = "maxtext"
WORKLOAD_LAUNCHER = "maxtext-launcher.sh"
KUEUE_NAME = None
OPTIMIZER = "adam"
NUM_STEPS = 30

SCHEDULED_TIME = (
    get_scheduled_time(HYPERCOMPUTER, MODEL_ID, FRAMEWORK)
    if composer_env.is_prod_env()
    else None
)
DAG_ID = f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}"

with models.DAG(
    dag_id=DAG_ID,
    schedule=SCHEDULED_TIME,
    tags=[
        "reproducibility",
        "experimental",
        "xlml",
        "regressiontests",
        "a3ultra",
        "maxtext",
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  run_workload_with_quarantine(
      test_name=DAG_ID,
      workload_function=run_workload,
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      num_steps=NUM_STEPS,
      kueue_name=KUEUE_NAME,
      metrics_model_id=METRICS_MODEL_ID,
      optimizer=OPTIMIZER,
      workload_launcher=WORKLOAD_LAUNCHER,
      config_model_name=f"llama3-1-405b-256gpus-a4-{PRECISION}.yaml",
  )
