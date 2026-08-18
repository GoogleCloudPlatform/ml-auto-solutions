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

"""Tests for xpk.py."""

import unittest
from unittest import mock
from xlml.utils import xpk


class XpkTest(unittest.TestCase):

  @mock.patch("xlml.utils.xpk.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.xpk.SubprocessHook")
  def test_run_workload_with_custom_namespace(
      self, mock_hook_class, mock_log_metadata
  ):
    mock_hook = mock.MagicMock()
    mock_hook.run_command.return_value.exit_code = 0
    mock_hook_class.return_value = mock_hook

    xpk.run_workload.function(
        task_id="test_task",
        cluster_project="test_proj",
        zone="us-central1-a",
        cluster_name="test_cluster",
        benchmark_id="test_bench",
        workload_id="workload_123",
        gcs_path="gs://test-bucket/path",
        docker_image="gcr.io/test-image",
        accelerator_type="v5p-8",
        run_cmds="echo test",
        namespace="automation-testing",
    )

    call_args = mock_hook.run_command.call_args[0][0]
    full_cmd = call_args[2]
    self.assertIn("--namespace=automation-testing", full_cmd)

  @mock.patch("xlml.utils.xpk.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.xpk.SubprocessHook")
  def test_run_workload_with_default_namespace_no_flag(
      self, mock_hook_class, mock_log_metadata
  ):
    mock_hook = mock.MagicMock()
    mock_hook.run_command.return_value.exit_code = 0
    mock_hook_class.return_value = mock_hook

    xpk.run_workload.function(
        task_id="test_task",
        cluster_project="test_proj",
        zone="us-central1-a",
        cluster_name="test_cluster",
        benchmark_id="test_bench",
        workload_id="workload_123",
        gcs_path="gs://test-bucket/path",
        docker_image="gcr.io/test-image",
        accelerator_type="v5p-8",
        run_cmds="echo test",
        namespace="default",
    )

    call_args = mock_hook.run_command.call_args[0][0]
    full_cmd = call_args[2]
    self.assertNotIn("--namespace", full_cmd)

  @mock.patch("xlml.utils.xpk.SubprocessHook")
  def test_clean_up_workload_with_custom_namespace(self, mock_hook_class):
    mock_hook = mock.MagicMock()
    mock_hook.run_command.return_value.exit_code = 0
    mock_hook_class.return_value = mock_hook

    xpk.clean_up_workload.function(
        workload_id="workload_123",
        project_id="test_proj",
        zone="us-central1-a",
        cluster_name="test_cluster",
        namespace="automation-testing",
    )

    call_args = mock_hook.run_command.call_args[0][0]
    full_cmd = call_args[2]
    self.assertIn("--namespace=automation-testing", full_cmd)

  @mock.patch("xlml.utils.xpk.SubprocessHook")
  def test_clean_up_workload_with_default_namespace(self, mock_hook_class):
    mock_hook = mock.MagicMock()
    mock_hook.run_command.return_value.exit_code = 0
    mock_hook_class.return_value = mock_hook

    xpk.clean_up_workload.function(
        workload_id="workload_123",
        project_id="test_proj",
        zone="us-central1-a",
        cluster_name="test_cluster",
        namespace="default",
    )

    call_args = mock_hook.run_command.call_args[0][0]
    full_cmd = call_args[2]
    self.assertNotIn("--namespace", full_cmd)


if __name__ == "__main__":
  unittest.main()
