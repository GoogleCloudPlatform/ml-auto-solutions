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

"""DAGs to run Aotc inference reproducibility benchmarks."""

import datetime
import os

from airflow import models
from dags import composer_env
from dags.map_reproducibility.internal_runs.dag_configs_inference import DAG_CONFIGS_INFERENCE_MEGA
from dags.map_reproducibility.utils.internal_aotc_inference_workload import run_internal_aotc_inference_workload


# Configuration parameters
TEST_RUN = False
TURN_ON_SCHEDULE = True if composer_env.is_prod_env() else False

# Pull the MaxText container from NVIDIA
IMAGE = "ghcr.io/nvidia/jax:maxtext"

# Define common tags
DAG_TAGS = [
    "reproducibility",
    "experimental",
    "xlml",
    "v1.13",
    "internal",
    "regressiontests",
    "a3mega",
    "inference",
]

# Create DAGs for each configuration
for config_path, config_info in DAG_CONFIGS_INFERENCE_MEGA.items():
  # Extract config name for the DAG ID
  config_name = os.path.basename(config_path).replace(".yaml", "")
  nightly_schedule = (
      config_info["nightly_schedule"] if TURN_ON_SCHEDULE else None
  )
  release_schedule = (
      config_info["release_schedule"] if TURN_ON_SCHEDULE else None
  )
  timeout = config_info["timeout_minutes"]

  # Create DAG for nightly build
  with models.DAG(
      dag_id=f"new_internal_inference_{config_name}",
      schedule=nightly_schedule,
      tags=DAG_TAGS,
      start_date=datetime.datetime(2025, 4, 17),
      catchup=False,
  ) as dag:
    run_internal_aotc_inference_workload(
        relative_config_yaml_path=config_path,
        test_run=TEST_RUN,
        timeout=timeout,
        image_version=IMAGE,
    )
