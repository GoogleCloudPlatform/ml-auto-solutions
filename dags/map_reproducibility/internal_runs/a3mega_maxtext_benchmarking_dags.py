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
from dags.map_reproducibility.utils.constants import Schedule
from dags.map_reproducibility.utils.aotc_workload import run_aotc_workload


TEST_RUN = False
TURN_ON_SCHEDULE = False

# List of configuration setups as a dictionary with schedule times
config_yamls = {
    "recipes/a3mega/a3mega_llama3.1-8b_8gpus_bf16_maxtext.yaml": Schedule.DAILY_6PM_EXCEPT_THURSDAY, # < 5mins
    # Add more config paths as needed
}

# Define common tags
common_tags = [
    "reproducibility",
    "experimental",
    "xlml",
    "v1.9",
    "internal",
    "regressiontests",
    "a3ultra",
]

# Create a DAG for each config
for relative_config_yaml_path, schedule_time in config_yamls.items():
    # Extract config name for the DAG ID
    config_yaml_name = relative_config_yaml_path.rsplit('/', maxsplit=1)[-1].replace(".yaml", "")
    
    dag_id = f"new_internal_{config_yaml_name}"
    actual_schedule = schedule_time if TURN_ON_SCHEDULE else None
    
    # Define the DAG
    with models.DAG(
        dag_id=dag_id,
        schedule=actual_schedule,  # Use the specific schedule time
        tags=common_tags,
        start_date=datetime.datetime(2025, 3, 15),
        catchup=False,
        is_paused_upon_creation=True if TEST_RUN else False
    ) as dag:
        # Create the workload for this specific config
        run_aotc_workload(relative_config_yaml_path=relative_config_yaml_path, test_run=TEST_RUN)
        # run_aotc_workload(relative_config_yaml_path=relative_config_yaml_path)
