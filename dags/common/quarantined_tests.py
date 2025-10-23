# Copyright 2024 Google LLC
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

"""Lists all currently broken tests."""

import dataclasses
import fnmatch
import logging
import os
from typing import Set

from airflow.models import Variable

from dags.common.test_owner import Team


@dataclasses.dataclass
class TestInfo:
  """Description of a flaky test."""

  owner: Team
  date_added: str
  details: str = ""


def parse_quarantine_patterns(quarantine_patterns_str: str) -> Set[str]:
  pattern_set = set()
  if len(quarantine_patterns_str.strip()) == 0:
    return pattern_set
  for pattern in quarantine_patterns_str.split("\n"):
    if len(pattern.strip()) > 0:
      pattern_set.add(pattern.strip().lower())  # Check capital?
  return pattern_set


def match_quarantine_patterns(
    test_name: str, quarantine_patterns_set: Set[str]
) -> bool:
  test_name_lower = test_name.lower()
  for pattern in quarantine_patterns_set:
    if fnmatch.fnmatch(test_name_lower, pattern):
      return True
  return False


def safe_get_from_variable(key: str, default_var: str):
  """
  Check whether the current runtime is GitHub Actions. Skip retrieving variables in GitHub Actions to avoid excessive log output.
  """
  value = default_var
  is_ci_env = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
  if is_ci_env:
    logging.info("In GitHub Actions, skip getting variables")
  else:
    value = Variable.get(key, default_var=default_var)
  return value


"""
The quarantine list is defined by a set of UNIX Shell Glob Patterns. 
These patterns are used to match and quarantine tests. 

The patterns are stored in the Airflow Variable named 'quarantine_patterns'.

Sample values:
- maxtext-profiling-* (Matches all 'maxtext-profiling-' tests)
- maxd-sdxl-* (Matches all 'maxd-sdxl-' tests)
"""
quarantine_patterns = parse_quarantine_patterns(
    safe_get_from_variable("quarantine_patterns", "")
)


class QuarantineTests:
  """A list of currently-flaky tests."""

  @staticmethod
  def is_quarantined(test_name) -> bool:
    """
    Checks if a test is quarantined using both legacy and current methods.

    The legacy method checks if `test_name` is present in `QuarantineTests.tests`.
    The current method checks against a runtime quarantine list fetched from Airflow Variables
    (key: 'quarantine_list') using `is_in_runtime_quarantine_list()`.

    The test is considered quarantined if it's found by either method.
    """
    return match_quarantine_patterns(test_name, quarantine_patterns)
