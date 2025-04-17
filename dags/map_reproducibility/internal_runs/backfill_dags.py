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

"""
Factory function to create Aotc reproducibility backfill DAGs with adaptive,
split-image grouping based on provided model configurations.
"""

import datetime
import os
from collections import defaultdict
import logging
from typing import Dict, Any, List, Optional

from airflow import models
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
# Assuming these are accessible in your Airflow environment
from dags.map_reproducibility.utils.constants import Image
from dags.map_reproducibility.internal_runs.dag_configs import DAG_CONFIGS_ULTRA, DAG_CONFIGS_MEGA
from dags.map_reproducibility.utils.internal_aotc_workload import run_internal_aotc_workload

# --- Default Configuration ---
# These can be overridden when calling the factory function if needed
DEFAULT_TEST_RUN = False
DEFAULT_BACKFILL = True

# --- Image Tag Generation ---
# Using today's date by default:
default_utc_date = "2025-04-08"
logging.info(f"Default UTC date for image tags: {default_utc_date}")

# You can override these tags when calling the factory function
DEFAULT_NIGHTLY_IMAGE_TAG = (
    f"{Image.MAXTEXT_JAX_STABLE_NIGHTLY}:{default_utc_date}"
)
DEFAULT_RELEASE_IMAGE_TAG = (
    f"{Image.MAXTEXT_JAX_STABLE_RELEASE}:{default_utc_date}"
)

# --- Base DAG Tags ---
BASE_DAG_TAGS = [
    "reproducibility",
    "experimental",
    "xlml",
    "v1.17",
    "internal",
    "regressiontests",
    "backfill",
]


# --- DAG Factory Function ---
def create_adaptive_backfill_dag(
    dag_id: str,
    model_configs: Dict[str, Dict[str, Any]],
    start_date: datetime.datetime,
    dag_tags: Optional[List[str]] = None,
    schedule: Optional[str] = None,  # Default None for backfill
    test_run: bool = DEFAULT_TEST_RUN,
    backfill: bool = DEFAULT_BACKFILL,
    nightly_image_tag: str = DEFAULT_NIGHTLY_IMAGE_TAG,
    release_image_tag: str = DEFAULT_RELEASE_IMAGE_TAG,
    retries: int = 2,  # Number of retries for tasks
) -> models.DAG:
  """
  Creates an Airflow DAG for backfilling Aotc reproducibility benchmarks.

  Features:
      - Sequentially runs tasks in numbered groups.
      - Allows assigning different group numbers for nightly vs. release images
        for the same base configuration via 'backfill_group_nightly'/'backfill_group_release' keys.
      - Dynamically chains groups based on assigned numbers.
      - Continues execution of subsequent groups even if tasks in previous groups fail.

  Args:
      dag_id: Unique identifier for the DAG.
      model_configs: Dictionary where keys are relative config YAML paths and
          values are dictionaries containing 'timeout_minutes', 'backfill_group_nightly' (int),
          and 'backfill_group_release' (int).
      start_date: The DAG's start date.
      dag_tags: Optional list of tags to add to the base tags.
      schedule: Airflow schedule interval (defaults to None).
      test_run: If True, potentially runs smaller test versions of tasks.
      backfill: Flag indicating if this is for backfill purposes.
      nightly_image_tag: Full tag for the nightly image to use.
      release_image_tag: Full tag for the release image to use.
      retries: Number of retries for each task before marking as failed.

  Returns:
      An Airflow DAG object.
  """
  effective_tags = BASE_DAG_TAGS + (dag_tags or [])

  # Define default_args to set retries for all tasks
  default_args = {
      "retries": retries,
      "retry_delay": datetime.timedelta(minutes=2),
  }

  with models.DAG(
      dag_id=dag_id,
      schedule=schedule,
      tags=effective_tags,
      start_date=start_date,
      catchup=False,  # Important for backfills
      default_args=default_args,
  ) as dag:
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(
        task_id="end",
        # This ensures the end task will run regardless of upstream failures
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # Organize configs by group number - UPDATED TO PREVENT DUPLICATES
    group_configs = defaultdict(list)
    # Track which config+image combinations have been added to which groups
    processed_configs = set()

    for config_path, config_info in model_configs.items():
      nightly_group = config_info.get("backfill_group_nightly")
      if isinstance(nightly_group, int):
        # Create a unique key for this config+image+group combination
        config_key = (config_path, "nightly", nightly_group)
        if config_key not in processed_configs:
          group_configs[nightly_group].append(
              (config_path, config_info, "nightly")
          )
          processed_configs.add(config_key)

      release_group = config_info.get("backfill_group_release")
      if isinstance(release_group, int):
        # Create a unique key for this config+image+group combination
        config_key = (config_path, "release", release_group)
        if config_key not in processed_configs:
          group_configs[release_group].append(
              (config_path, config_info, "release")
          )
          processed_configs.add(config_key)

    # Get sorted group numbers
    sorted_group_numbers = sorted(group_configs.keys())
    if not sorted_group_numbers:
      logging.warning(
          f"[{dag_id}] No tasks found in any group. Linking start >> end."
      )
      start >> end
      return dag

    logging.info(
        f"[{dag_id}] Found configurations for groups: {sorted_group_numbers}"
    )

    # Create TaskGroups for each group number
    task_groups = {}
    group_gateways = {}  # Store gateway tasks for each group

    for idx, group_num in enumerate(sorted_group_numbers):
      group_task_id = f"group_{group_num}"

      # Create a TaskGroup for this group number
      with TaskGroup(group_id=group_task_id) as tg:
        # Create start and end gateways for this group
        group_start = EmptyOperator(
            task_id=f"group_{group_num}_start",
            trigger_rule=TriggerRule.ALL_SUCCESS,  # Default behavior
        )

        group_end = EmptyOperator(
            task_id=f"group_{group_num}_end",
            # This allows the group to complete even if some tasks fail
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        )

        # Remember these gateways
        group_gateways[group_num] = (group_start, group_end)

        # Create tasks within this group - UPDATED TO PREVENT DUPLICATES
        task_list = []

        # Process each config with its specific image type
        for config_path, config_info, image_type in sorted(
            group_configs[group_num],
            key=lambda x: (x[0], x[2]),  # Sort by config path and image type
        ):
          config_name = os.path.basename(config_path).replace(".yaml", "")
          timeout = config_info.get("timeout_minutes", 60)

          # Determine image tag based on type
          image_tag = (
              nightly_image_tag
              if image_type == "nightly"
              else release_image_tag
          )
          custom_task_id = f"{config_name}_{image_type}"

          # Create task within the TaskGroup
          task = run_internal_aotc_workload.override(task_id=custom_task_id)(
              relative_config_yaml_path=config_path,
              test_run=test_run,
              backfill=backfill,
              timeout=timeout,
              image_version=image_tag,
          )
          task_list.append(task)

        # Set up the task dependencies within the group
        if task_list:
          # Connect all tasks to the group start and end
          for task in task_list:
            group_start >> task >> group_end
        else:
          # If no tasks, just link start and end
          group_start >> group_end

      # Store TaskGroup reference
      task_groups[group_num] = tg

    # Set up dependencies between groups and start/end
    (
        start >> group_gateways[sorted_group_numbers[0]][0]
    )  # Start to first group's start

    # Chain TaskGroups in sequence with appropriate trigger rules
    for i in range(len(sorted_group_numbers) - 1):
      current_group = sorted_group_numbers[i]
      next_group = sorted_group_numbers[i + 1]

      # Connect current group's end to next group's start
      # The next group will start even if some tasks in the current group fail
      group_gateways[current_group][1] >> group_gateways[next_group][0]

    # Connect last group's end to DAG end
    group_gateways[sorted_group_numbers[-1]][1] >> end

  return dag


# --- Instantiate specific DAGs using the factory ---
# a3ultra backfill DAG
dag1 = create_adaptive_backfill_dag(
    dag_id="new_internal_backill_a3ultra",
    model_configs=DAG_CONFIGS_ULTRA,
    start_date=datetime.datetime(2025, 4, 11),
    dag_tags=BASE_DAG_TAGS,
)

dag2 = create_adaptive_backfill_dag(
    dag_id="new_internal_backill_a3mega",
    model_configs=DAG_CONFIGS_MEGA,
    start_date=datetime.datetime(2025, 4, 11),
    dag_tags=BASE_DAG_TAGS,
)
