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

"""Utilities for GitHub API integration."""

from typing import Any
import requests

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.operators.python import get_current_context


@task
def validate_git_trigger(
    repo: str | None = None,
    token: str | None = None,
    run_id: str | None = None,
    commit_sha: str | None = None,
) -> None:
  """Validates that the DAG run was externally triggered with required params.

  Enforces external trigger check and ensures required GitHub parameters exist.

  Args:
    repo: Target GitHub repository in owner/repo format.
    token: GitHub PAT used to fire the repository_dispatch callback.
    run_id: GitHub Actions run ID of the originating workflow.
    commit_sha: Commit SHA being tested.
  """
  context = get_current_context()
  dag_run = context.get("dag_run")

  if not dag_run or not dag_run.external_trigger:
    raise AirflowFailException(
        "This DAG should not be run manually from the Airflow UI."
    )

  if not repo or not token or not run_id or not commit_sha:
    raise AirflowFailException(
        "Missing required GitHub parameters (repo, token, run_id, commit_sha)."
    )


@task
def trigger_github_repository_dispatch(
    repo: str,
    token: str,
    event_type: str = "airflow-dag-complete",
    client_payload: dict[str, Any] | None = None,
) -> None:
  """Fires a GitHub repository_dispatch event via the GitHub API."""
  if not repo or not token:
    raise AirflowFailException(
        "Missing required GitHub parameters (repo, token) to fire callback."
    )

  payload = client_payload or {}

  response = requests.post(
      f"https://api.github.com/repos/{repo}/dispatches",
      headers={
          "Authorization": f"Bearer {token}",
          "Accept": "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
      },
      json={
          "event_type": event_type,
          "client_payload": payload,
      },
      timeout=30,
  )
  response.raise_for_status()
