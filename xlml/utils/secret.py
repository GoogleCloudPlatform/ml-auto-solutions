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

"""Reads secrets at task run time.

A secret must never be part of a command string: test scripts run under
`set -x`, which prints every command to the logs. A test config therefore only
names the Airflow Variable that holds a secret (`TestConfig.secret_env`). The
task that launches the workload calls `resolve_env` to read the value and
passes it to the workload as an environment variable.
"""

from collections.abc import Mapping

from airflow.exceptions import AirflowFailException
from airflow.utils.log.secrets_masker import mask_secret
from dags.common.quarantined_tests import safe_get_from_variable


def _unquote(value: str | None) -> str | None:
  """Removes whitespace and one pair of matching quotes around a value.

  Some Variables are stored as `'value'`. When the value was part of an
  `export NAME=<value>` command, the shell removed the quotes. An environment
  variable keeps them, so remove them here.
  """
  if not value:
    return value
  value = value.strip()
  if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
    value = value[1:-1]
  return value


def resolve_env(secret_env: Mapping[str, str] | None) -> dict[str, str]:
  """Reads secrets from Airflow Variables. Call this inside a task.

  Args:
    secret_env: Maps an environment variable name to the name of the Airflow
      Variable that holds its value.

  Returns:
    A dict that maps each environment variable name to its secret value. The
    values are registered with the Airflow log masker.
  """
  resolved = {}
  for env_name, variable_name in (secret_env or {}).items():
    value = _unquote(safe_get_from_variable(variable_name, None))
    if not value:
      raise AirflowFailException(
          f"Airflow Variable {variable_name} is not set; {env_name} needs it."
      )
    mask_secret(value)
    resolved[env_name] = value
  return resolved
