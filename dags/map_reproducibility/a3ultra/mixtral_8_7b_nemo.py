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
from dags.map_reproducibility.utils.common_utils import get_scheduled_time, run_workload_with_quarantine
from dags.map_reproducibility.utils.common_utils import run_workload


MODEL_ID = "mixtral-8x7b"
METRICS_MODEL_ID = "mixtral-7b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3ultra"
FRAMEWORK = "nemo"
WORKLOAD_LAUNCHER = "nemo-10-launcher.sh"


SCHEDULED_TIME = (
    get_scheduled_time(HYPERCOMPUTER, MODEL_ID, FRAMEWORK)
    if composer_env.is_prod_env()
    else None
)

KUEUE_NAME = "a3-ultra"
NUM_GPUS = 256
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
        "GPU",
        "nvidia-h200-80gb",
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
      kueue_name=KUEUE_NAME,
      metrics_model_id=METRICS_MODEL_ID,
      config_model_name=f"{MODEL_ID}-{NUM_GPUS}gpus-a3u-{PRECISION}.yaml",
      workload_launcher=WORKLOAD_LAUNCHER,
  )
