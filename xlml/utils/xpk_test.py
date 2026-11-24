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

"""Unit tests for xpk.py."""

import re
import unittest
from unittest import mock

from xlml.utils import xpk


def _run_workload(**kwargs):
  xpk.run_workload.function(
      task_id="run_workload",
      cluster_project="test-project",
      zone="us-central1-a",
      cluster_name="test-cluster",
      benchmark_id="test-bench",
      workload_id="test-workload",
      gcs_path="gs://test-bucket/output",
      docker_image="us-docker.pkg.dev/img:latest",
      accelerator_type="v5p-8",
      run_cmds="set -xue;bash run.sh",
      **kwargs,
  )


@mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
@mock.patch("xlml.utils.xpk.SubprocessHook")
class RunWorkloadSecretEnvTest(unittest.TestCase):

  @mock.patch("xlml.utils.xpk.secret.resolve_env")
  def test_secret_is_passed_by_name_only(self, mock_resolve, mock_hook_cls, _):
    mock_resolve.return_value = {"HF_TOKEN": "hf_canary"}
    seen = {}

    def run_command(command, env):
      seen["script"] = command[2]
      seen["env"] = env
      # The temporary directory is removed when run_workload returns.
      env_file = re.search(r"--env-file=(\S+)", command[2]).group(1)
      with open(env_file, encoding="utf-8") as f:
        seen["env_file"] = f.read()
      return mock.MagicMock(exit_code=0)

    mock_hook_cls.return_value.run_command.side_effect = run_command

    _run_workload(secret_env={"HF_TOKEN": "HF_TOKEN_MOE"})

    mock_resolve.assert_called_once_with({"HF_TOKEN": "HF_TOKEN_MOE"})
    self.assertEqual(
        seen["env_file"], "GCS_OUTPUT=gs://test-bucket/output\nHF_TOKEN\n"
    )
    # xpk rejects --env together with --env-file.
    self.assertNotIn(" --env ", seen["script"])
    self.assertEqual(seen["env"]["HF_TOKEN"], "hf_canary")
    self.assertNotIn("hf_canary", seen["script"])

  def test_no_secret_adds_no_env_file(self, mock_hook_cls, _):
    mock_hook = mock_hook_cls.return_value
    mock_hook.run_command.return_value.exit_code = 0

    _run_workload()

    script = mock_hook.run_command.call_args[0][0][2]
    self.assertNotIn("--env-file", script)
    self.assertIn(" --env GCS_OUTPUT=gs://test-bucket/output", script)


if __name__ == "__main__":
  unittest.main()
