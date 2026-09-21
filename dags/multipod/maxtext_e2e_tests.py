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
MaxText E2E Tests Orchestrator DAG.

Coordinates the multi-stage MaxText end-to-end testing pipeline for GitHub CI:
1. Stage 1 (maxtext_e2e_tpu_checkpoint_conversion):
   Converts Hugging Face checkpoints to MaxText format on TPU v5p-8.
2. Stage 2 (maxtext_e2e_tpu_pre_training & maxtext_e2e_tpu_post_training):
   Triggers pre-training and post-training test suites once
   checkpoints are ready.
   The `test_scope` param selects which of the two suites run, so manual or
   GitHub-triggered runs can exercise a single suite without burning TPU
   capacity on the other one.
3. Callbacks & Reporting:
   Fires GitHub repository_dispatch events upon stage completion
   for automated CI.
"""
import datetime
import enum

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.trigger_rule import TriggerRule
from xlml.utils.github import (
    trigger_github_repository_dispatch,
    validate_git_trigger,
)

# Images for the candidate build under test. Unlike the daily builds in
# `DockerImage`, these are published by the MaxText build workflow and tagged
# with the GitHub run ID, so the build mode and tag are only known at run time.
PRE_TRAINING_DOCKER_IMAGE = (
    "us-docker.pkg.dev/tpu-prod-env-multipod/maxtext-images/"
    "maxtext_jax_"
    "{{ params.build_mode }}:{{ params.github_run_id }}"
)
POST_TRAINING_DOCKER_IMAGE = (
    "us-docker.pkg.dev/tpu-prod-env-multipod/maxtext-images/"
    "maxtext_post_training_"
    "{{ params.build_mode }}:{{ params.github_run_id }}"
)


class TestScope(enum.IntFlag):
  """Stage 2 suites selected by the `test_scope` param."""

  PRE_TRAINING = enum.auto()
  POST_TRAINING = enum.auto()

  ALL = PRE_TRAINING | POST_TRAINING

  @classmethod
  def from_string(cls, value: str) -> "TestScope":
    return {
        "all": cls.ALL,
        "pre-training": cls.PRE_TRAINING,
        "post-training": cls.POST_TRAINING,
    }[value]


def _get_test_scope() -> TestScope:
  """Resolves the requested scope, defaulting to running every suite."""
  context = get_current_context()
  dag_run = context["dag_run"]
  params = context["params"]

  conf = (dag_run.conf if dag_run else {}) or {}
  scope = conf.get("test_scope") or (params or {}).get("test_scope") or "all"

  return TestScope.from_string(scope)


@task.short_circuit
def test_scope_enabled(required_scope: int) -> bool:
  """Skips everything downstream unless `required_scope` was requested.

  `short_circuit` skips the whole downstream branch, including the GitHub
  callback, so a suite that was not selected is never reported back as a
  success. The flag is passed as an int because task arguments have to survive
  DAG serialization.
  """
  return bool(_get_test_scope() & TestScope(required_scope))


with models.DAG(
    dag_id="maxtext_e2e_tests",
    schedule=None,
    tags=[
        "maxtext",
        "e2e",
        "pre-training",
        "post-training",
        "checkpoint-conversion",
    ],
    start_date=datetime.datetime(2026, 6, 10),
    catchup=False,
    params={
        "build_mode": Param(
            type="string",
            description="Build mode: stable or nightly",
        ),
        "commit_sha": Param(
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
        "github_token": Param(
            type="string",
            description=(
                "GitHub PAT used to fire the repository_dispatch callback"
            ),
        ),
        "test_scope": Param(
            default="all",
            type="string",
            enum=["all", "pre-training", "post-training"],
            description=(
                "Which Stage 2 suites to run. Checkpoint conversion always"
                " runs because both suites consume its checkpoints."
            ),
        ),
    },
) as dag:
  validate_task = validate_git_trigger(
      repo="{{ params.github_repo }}",
      token="{{ params.github_token }}",
      run_id="{{ params.github_run_id }}",
      commit_sha="{{ params.commit_sha }}",
  )

  shared_run_name = "e2e-{{ params.github_run_id }}"

  trigger_checkpoint_conversion = TriggerDagRunOperator(
      task_id="trigger_checkpoint_conversion",
      trigger_dag_id="maxtext_e2e_tpu_checkpoint_conversion",
      trigger_run_id="{{ run_id }}__checkpoint_conversion",
      execution_date="{{ logical_date }}",
      conf={
          "docker_image": POST_TRAINING_DOCKER_IMAGE,
          "run_name": shared_run_name,
      },
      wait_for_completion=False,
  )

  trigger_pre_training = TriggerDagRunOperator(
      task_id="trigger_tpu_pre_training",
      trigger_dag_id="maxtext_e2e_tpu_pre_training",
      trigger_run_id="{{ run_id }}__pre_training",
      execution_date="{{ logical_date }}",
      conf={
          "docker_image": PRE_TRAINING_DOCKER_IMAGE,
          "run_name": shared_run_name,
      },
      wait_for_completion=True,
      poke_interval=600,  # check every 10 minutes for child DAG completion
  )

  trigger_post_training = TriggerDagRunOperator(
      task_id="trigger_tpu_post_training",
      trigger_dag_id="maxtext_e2e_tpu_post_training",
      trigger_run_id="{{ run_id }}__post_training",
      execution_date="{{ logical_date }}",
      conf={
          "docker_image": POST_TRAINING_DOCKER_IMAGE,
          "run_name": shared_run_name,
      },
      wait_for_completion=True,
      poke_interval=600,  # check every 10 minutes for child DAG completion
  )

  github_callback_pre_training = trigger_github_repository_dispatch.override(
      task_id="fire_github_callback_pre_training",
      trigger_rule=TriggerRule.ALL_SUCCESS,
  )(
      repo="{{ params.github_repo }}",
      token="{{ params.github_token }}",
      client_payload={
          "state": "success",
          "dag_id": "{{ dag.dag_id }}",
          "dag_run_id": "{{ run_id }}",
          "sha": "{{ params.commit_sha }}",
          "github_run_id": "{{ params.github_run_id }}",
          "test_type": "pre_training",
      },
  )

  github_callback_post_training = trigger_github_repository_dispatch.override(
      task_id="fire_github_callback_post_training",
      trigger_rule=TriggerRule.ALL_SUCCESS,
  )(
      repo="{{ params.github_repo }}",
      token="{{ params.github_token }}",
      client_payload={
          "state": "success",
          "dag_id": "{{ dag.dag_id }}",
          "dag_run_id": "{{ run_id }}",
          "sha": "{{ params.commit_sha }}",
          "github_run_id": "{{ params.github_run_id }}",
          "test_type": "post_training",
      },
  )

  # Checkpoint conversion is never gated: both suites wait on its per-model task
  # groups through an ExternalTaskSensor and share the converted checkpoints.
  chain(
      validate_task,
      trigger_checkpoint_conversion,
  )
  chain(
      validate_task,
      test_scope_enabled.override(task_id="pre_training_enabled")(
          TestScope.PRE_TRAINING.value
      ),
      trigger_pre_training,
      github_callback_pre_training,
  )
  chain(
      validate_task,
      test_scope_enabled.override(task_id="post_training_enabled")(
          TestScope.POST_TRAINING.value
      ),
      trigger_post_training,
      github_callback_post_training,
  )
