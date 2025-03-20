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

"""DAGs to run Aotc reproducibility benchmarks."""

import datetime

from airflow import models
from dags.map_reproducibility.utils.aotc_workload import run_aotc_workload


# List of configuration setups
config_yamls = [
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_bf16_maxtext.yaml",
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_fp8_maxtext.yaml",
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_bf16_maxtext.yaml",
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_fp8_maxtext.yaml",

    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_bf16_maxtext.yaml",
    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_fp8_maxtext.yaml",

    "recipes/a3ultra/a3ultra_mixtral8x7b_8gpus_bf16_maxtext.yaml",
    "recipes/a3ultra/a3ultra_mixtral8x7b_8gpus_fp8_maxtext.yaml",
    "recipes/a3ultra/a3ultra_mixtral8x7b_16gpus_bf16_maxtext.yaml",
    "recipes/a3ultra/a3ultra_mixtral8x7b_16gpus_fp8_maxtext.yaml",

    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_fp8_maxtext.yaml",
    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_bf16_maxtext.yaml",
    # Add more config paths as needed
]

SCHEDULED_TIME = None

# Define common tags
common_tags = [
    "reproducibility",
    "experimental",
    "xlml",
    "v1.7"
    "internal",
    "regressiontests",
    "a3ultra",
]

# Create a DAG for each config
for relative_config_yaml_path in config_yamls:
    # Extract config name for the DAG ID
    config_yaml_name = relative_config_yaml_path.rsplit('/', maxsplit=1)[-1].replace(".yaml", "")
    
    dag_id = f"new_internal_{config_yaml_name}"
    
    # Define the DAG
    with models.DAG(
        dag_id=dag_id,
        schedule=SCHEDULED_TIME,
        tags=common_tags,
        start_date=datetime.datetime(2025, 3, 15),
        catchup=False,
    ) as dag:
        # Create the workload for this specific config
        # run_aotc_workload(relative_config_yaml_path=relative_config_yaml_path, test_run=True)
        run_aotc_workload(relative_config_yaml_path=relative_config_yaml_path)