# Copyright 2026 Google LLC
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
Parent DAG that triggers MaxText E2E TPU pre-training and post-training child
DAGs in parallel, waits for both to complete, then fires a GitHub
repository_dispatch callback with the aggregated result.
"""
import datetime
from airflow import models
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from xlml.utils.github import fire_github_callback, validate_git_trigger

with models.DAG(
    dag_id="maxtext_e2e_tests",
    schedule=None,
    tags=[
        "maxtext",
        "e2e",
        "pre-training",
        "post-training",
    ],
    start_date=datetime.datetime(2026, 6, 10),
    catchup=False,
    params={
        "build_mode": Param(
            type="string",
            description="Build mode: stable or nightly",
        ),
        "maxtext_sha": Param(
            type="string",
            description="Commit SHA being tested",
        ),
        "github_run_id": Param(
            type="string",
            description="GitHub Actions run ID of the original build workflow",
        ),
        "github_repo": Param(
            type="string",
            description="GitHub repository in owner/repo format",
        ),
        "github_callback_token": Param(
            type="string",
            description="GitHub PAT used to fire the repository_dispatch callback",
        ),
    },
) as dag:
  trigger_pre_training = TriggerDagRunOperator(
      task_id="trigger_tpu_pre_training",
      trigger_dag_id="maxtext_e2e_tpu_pre_training",
      conf={
          "docker_image": "gcr.io/tpu-prod-env-multipod/maxtext_jax_{{ params.build_mode }}:{{ params.github_run_id }}"
      },
      wait_for_completion=True,
      poke_interval=600,  # check every 10 minutes for child DAG completion
  )

  trigger_post_training = TriggerDagRunOperator(
      task_id="trigger_tpu_post_training",
      trigger_dag_id="maxtext_e2e_tpu_post_training",
      conf={
          "docker_image": "gcr.io/tpu-prod-env-multipod/maxtext_post_training_{{ params.build_mode }}:{{ params.github_run_id }}"
      },
      wait_for_completion=True,
      poke_interval=600,  # check every 10 minutes for child DAG completion
  )

  github_callback = PythonOperator(
      task_id="fire_github_callback",
      python_callable=fire_github_callback,
      trigger_rule=TriggerRule.ALL_SUCCESS,  # Only fire if all upstream tasks succeeded
  )

  validate_task = PythonOperator(
      task_id="validate_trigger",
      python_callable=validate_git_trigger,
  )

  (
      validate_task
      >> [trigger_pre_training, trigger_post_training]
      >> github_callback
  )
