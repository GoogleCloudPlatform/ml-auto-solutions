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

import requests
from airflow.exceptions import AirflowFailException


def validate_git_trigger(**context):
  """Validates that the DAG run has required GitHub parameters.

  Prevents manual runs from the Airflow UI by ensuring github_run_id,
  github_repo, and github_callback_token are present.
  """
  params = context["params"]
  run_id = params.get("github_run_id")
  repo = params.get("github_repo")
  token = params.get("github_callback_token")

  if not run_id or not repo or not token:
    raise AirflowFailException(
        "Missing required GitHub parameters (run_id, repo, token). "
        "This DAG should not be run manually from the Airflow UI."
    )


def fire_github_callback(test_type: str | None = None, **context):
  """Fires a GitHub repository_dispatch callback with the DAG run result."""
  params = context["params"]
  dag_run = context["dag_run"]

  client_payload = {
      "state": "success",
      "dag_id": dag_run.dag_id,
      "dag_run_id": dag_run.run_id,
      "sha": params["maxtext_sha"],
      "github_run_id": params["github_run_id"],
  }
  if test_type:
    client_payload["test_type"] = test_type

  response = requests.post(
      f"https://api.github.com/repos/{params['github_repo']}/dispatches",
      headers={
          "Authorization": f"Bearer {params['github_callback_token']}",
          "Accept": "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
      },
      json={
          "event_type": "airflow-dag-complete",
          "client_payload": client_payload,
      },
      timeout=30,
  )
  response.raise_for_status()
