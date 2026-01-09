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
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.common import test_owner
from dags.map_reproducibility.utils.common_utils import get_scheduled_time, run_workload_with_quarantine
from dags.map_reproducibility.utils.common_utils import run_workload


MODEL_ID = "llama3-1-70b"
METRICS_MODEL_ID = "llama3.1-70b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3ultra"
FRAMEWORK = "maxtext"
WORKLOAD_LAUNCHER = "maxtext-launcher.sh"
OPTIMIZER = "adam"
NUM_STEPS = 20

SCHEDULED_TIME = (
    get_scheduled_time(HYPERCOMPUTER, MODEL_ID, FRAMEWORK)
    if composer_env.is_prod_env()
    else None
)

SOFTWARE_ID = "jax_maxtext"
KUEUE_NAME = "a3-ultra"


with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}",
    schedule="0 2 * * 0",
    tags=[
        "reproducibility",
        "experimental",
        "xlml",
        "regressiontests",
        "a3ultra",
        "maxtext",
        "GPU",
        "nvidia-h200-80gb",
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", prefix_group_id=False
  )
  run_256gpus = run_workload_with_quarantine(
      test_name=f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}_256gpus",
      owner=test_owner.BRYAN_W,
      workload_function=run_workload,
      quarantine_task_group=quarantine_task_group,
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      kueue_name=KUEUE_NAME,
      metrics_model_id=METRICS_MODEL_ID,
      workload_launcher=WORKLOAD_LAUNCHER,
      config_model_name=f"llama3-1-70b-256gpus-a3u-{PRECISION}.yaml",
      optimizer=OPTIMIZER,
      num_steps=NUM_STEPS,
  )
  run_512gpus = run_workload_with_quarantine(
      test_name=f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}_512gpus",
      workload_function=run_workload,
      owner=test_owner.BRYAN_W,
      quarantine_task_group=quarantine_task_group,
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      kueue_name=KUEUE_NAME,
      metrics_model_id=METRICS_MODEL_ID,
      workload_launcher=WORKLOAD_LAUNCHER,
      num_gpus=512,
      config_model_name=f"llama3-1-70b-512gpus-a3u-{PRECISION}.yaml",
      optimizer=OPTIMIZER,
      num_steps=NUM_STEPS,
  )

  run_256gpus >> run_512gpus
