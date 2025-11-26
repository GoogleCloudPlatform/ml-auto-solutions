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

"""Utility functions for running subprocess commands.

This module uses the built-in Python `subprocess` module directly, rather than
relying on the Airflow `SubProcessHook`.

The `SubprocessHook.run_command()` method has a known limitation where its
result object (`result.output`) only stores the last line of the standard output
(STDOUT).
Our requirement is to capture the complete output from the command (e.g., the
full `tpu-info` table).
Therefore, we are using Python's native `subprocess.run()` instead, which allows
us to capture the full STDOUT and STDERR streams.
"""

import logging
import subprocess

from airflow.exceptions import AirflowFailException


def run_exec(
    cmd: str,
    env: dict[str, str] | None = None,
    log_command: bool = True,
    log_output: bool = True,
) -> str:
  """Executes a shell command and logs its output."""
  if log_command:
    logging.info("[subprocess] executing command:\n %s\n", cmd)

  res = subprocess.run(
      # This is the command to execute (e.g., 'ls -l | grep file'), in a single
      # string.
      cmd,
      # Optional: Environment variables to use for the command
      # (overrides parent environment)
      env=env,
      # REQUIRED for shell features (e.g., pipes, redirects).
      shell=True,
      # Optional: Do NOT raise CalledProcessError if the command returns a
      # non-zero exit code. We handle errors manually.
      check=False,
      # Optional: Capture stdout and stderr in the result object.
      capture_output=True,
      # Optional: Decode stdout/stderr output as text strings
      # (using the default system encoding).
      text=True,
  )

  if res.returncode != 0:
    logging.info("[subprocess] stderr: %s", res.stderr)
    raise AirflowFailException(
        "Caught an error while executing a command. stderr Message:"
        f" {res.stderr}"
    )

  if log_output:
    logging.info("[subprocess] stdout: %s", res.stdout)

  return res.stdout
