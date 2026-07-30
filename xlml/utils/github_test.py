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

"""Tests for github.py."""

import unittest
from unittest import mock
from airflow.exceptions import AirflowFailException
from xlml.utils import github


class GithubTest(unittest.TestCase):

  @mock.patch("xlml.utils.github.get_current_context")
  def test_validate_git_trigger_success(self, mock_context):
    mock_dag_run = mock.MagicMock()
    mock_dag_run.external_trigger = True
    mock_context.return_value = {
        "dag_run": mock_dag_run,
        "params": {
            "github_run_id": "12345",
            "github_repo": "owner/repo",
            "github_token": "secret_token",
        },
    }
    # Should not raise exception
    github.validate_git_trigger.function(
        repo="owner/repo",
        token="secret_token",
        run_id="12345",
        commit_sha="abc1234",
    )

  @mock.patch("xlml.utils.github.get_current_context")
  def test_validate_git_trigger_manual_run(self, mock_context):
    mock_dag_run = mock.MagicMock()
    mock_dag_run.external_trigger = False
    mock_context.return_value = {"dag_run": mock_dag_run}
    with self.assertRaises(AirflowFailException):
      github.validate_git_trigger.function(
          repo="owner/repo",
          token="secret_token",
          run_id="12345",
          commit_sha="abc1234",
      )

  @mock.patch("xlml.utils.github.get_current_context")
  def test_validate_git_trigger_missing_params(self, mock_context):
    mock_dag_run = mock.MagicMock()
    mock_dag_run.external_trigger = True
    mock_context.return_value = {"dag_run": mock_dag_run}
    with self.assertRaises(AirflowFailException):
      github.validate_git_trigger.function(
          repo="",
          token="secret_token",
          run_id="12345",
          commit_sha="abc1234",
      )

  @mock.patch("requests.post")
  def test_trigger_github_repository_dispatch_success(self, mock_post):
    mock_response = mock.MagicMock()
    mock_post.return_value = mock_response

    github.trigger_github_repository_dispatch.function(
        repo="owner/repo",
        token="secret_token",
        event_type="custom-event",
        client_payload={"state": "success", "sha": "abc1234"},
    )

    mock_post.assert_called_once_with(
        "https://api.github.com/repos/owner/repo/dispatches",
        headers={
            "Authorization": "Bearer secret_token",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={
            "event_type": "custom-event",
            "client_payload": {"state": "success", "sha": "abc1234"},
        },
        timeout=30,
    )
    mock_response.raise_for_status.assert_called_once()

  def test_trigger_github_repository_dispatch_missing_token(self):
    with self.assertRaises(AirflowFailException):
      github.trigger_github_repository_dispatch.function(
          repo="owner/repo",
          token="",
      )


if __name__ == "__main__":
  unittest.main()
