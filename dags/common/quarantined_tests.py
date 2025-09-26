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
import logging
from typing import Dict, Any

from airflow.models import Variable

from dags.common.test_owner import Team


@dataclasses.dataclass
class TestInfo:
  """Description of a flaky test."""

  owner: Team
  date_added: str
  details: str = ""


def parse_quarantine_list(content: str) -> Dict[str, Any]:
  tests = {}
  for line in content.split("\n"):
    try:
      segments = line.split(",")
      if len(segments) != 4:
        raise Exception(f"Wrong format of the quarantine item: {line}. Please fix it.")
      key = segments[0]
      team = Team(segments[1])
      date = segments[2]
      details = segments[3]
      tests[key] = TestInfo(team, date, details)
    except Exception as e:
      logging.error(e)
  return tests


new_dict = parse_quarantine_list(Variable.get("quarantine_list", ""))


class QuarantineTests:
  """A list of currently-flaky tests."""

  @staticmethod
  def is_quarantined(test_name) -> bool:
    return test_name in new_dict
