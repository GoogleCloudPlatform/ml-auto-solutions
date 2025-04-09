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
import os

from airflow import models
from dags import composer_env
from dags.map_reproducibility.utils.constants import Schedule, Image
from dags.map_reproducibility.utils.internal_aotc_workload import run_internal_aotc_workload


# Configuration parameters
TEST_RUN = False
TURN_ON_SCHEDULE = True if composer_env.is_prod_env() else False
BACKFILL = False

# Get current date for image tags
utc_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
NIGHTLY_IMAGE = f"{Image.MAXTEXT_JAX_STABLE_NIGHTLY}:{utc_date}"
RELEASE_IMAGE = f"{Image.MAXTEXT_JAX_STABLE_RELEASE}:{utc_date}"

# Common DAG tags
DAG_TAGS = [
    "reproducibility",
    "experimental",
    "xlml",
    "v1.17",
    "internal",
    "regressiontests",
    "a3ultra",
]

# Model configurations with schedule and timeout settings
MODEL_CONFIGS = {
    # a3ultra_llama3.1-8b
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_bf16_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_6PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_fp8_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_6PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_bf16_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_6PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_fp8_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_6PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    # a3ultra_mixtral-8x7
    "recipes/a3ultra/a3ultra_mixtral-8x7b_8gpus_bf16_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_6PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    "recipes/a3ultra/a3ultra_mixtral-8x7b_16gpus_bf16_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_6PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    # a3ultra_llama3.1-70b
    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_bf16_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_6_30PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_fp8_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_7PM_EXCEPT_THURSDAY,
        "timeout_minutes": 15,
    },
    # a3ultra_llama3.1-405b
    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_fp8_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_7_30PM_EXCEPT_THURSDAY,
        "timeout_minutes": 30,
    },
    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_bf16_maxtext.yaml": {
        "schedule": Schedule.WEEKDAY_PDT_8PM_EXCEPT_THURSDAY,
        "timeout_minutes": 40,
    },
}


# Create DAGs for each configuration
for config_path, config_info in MODEL_CONFIGS.items():
  # Extract config name for the DAG ID
  config_name = os.path.basename(config_path).replace(".yaml", "")
  schedule = config_info["schedule"] if TURN_ON_SCHEDULE else None
  timeout = config_info["timeout_minutes"]

  # Create DAG for nightly build
  with models.DAG(
      dag_id=f"new_internal_{config_name}",
      schedule=schedule,
      tags=DAG_TAGS,
      start_date=datetime.datetime(2025, 4, 3),
      catchup=False,
  ) as dag:
    run_internal_aotc_workload(
        relative_config_yaml_path=config_path,
        test_run=TEST_RUN,
        backfill=BACKFILL,
        timeout=timeout,
        image_version=NIGHTLY_IMAGE,
    )

  # Create DAG for stable release
  with models.DAG(
      dag_id=f"new_internal_stable_release_{config_name}",
      schedule=schedule,
      tags=DAG_TAGS,
      start_date=datetime.datetime(2025, 4, 3),
      catchup=False,
  ) as dag:
    run_internal_aotc_workload(
        relative_config_yaml_path=config_path,
        test_run=TEST_RUN,
        backfill=BACKFILL,
        timeout=timeout,
        image_version=RELEASE_IMAGE,
    )
