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
from dags.map_reproducibility.utils.constants import Image
from dags.map_reproducibility.internal_runs.dag_configs import DAG_CONFIGS_ULTRA
from dags.map_reproducibility.utils.internal_aotc_workload import run_internal_aotc_workload


# Configuration parameters
TEST_RUN = False if composer_env.is_prod_env() else True
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


# Create DAGs for each configuration
for config_path, config_info in DAG_CONFIGS_ULTRA.items():
  # Extract config name for the DAG ID
  config_name = os.path.basename(config_path).replace(".yaml", "")
  schedule = config_info["schedule"] if not TEST_RUN else None
  timeout = config_info["timeout_minutes"]
  # Set retry parameter based on timeout
  retries = 1 if timeout <= 15 else 2
  retry_delay = datetime.timedelta(minutes=1)

  dag_default_args = {
      "retries": retries,
      "retry_delay": retry_delay,
  }

  # Create DAG for nightly build
  with models.DAG(
      dag_id=f"new_internal_{config_name}",
      default_args=dag_default_args,
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
      default_args=dag_default_args,
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
