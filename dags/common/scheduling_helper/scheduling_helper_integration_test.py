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

"""
Integration test to ensure all DAGs in the folder are registered in the helper.
"""

import os
import yaml

from absl.testing import absltest
from airflow.models import DagBag

from dags.common.scheduling_helper import scheduling_helper


class TestSchedulingHelperIntegration(absltest.TestCase):
  """
  Ensures all non-whitelisted DAGs are registered
  in the scheduling helper.
  """

  WHITELIST_PATH = "dags/common/scheduling_helper/dag_integrity_whitelist.yaml"
  DAG_ROOT = "dags/"

  def _load_whitelist(self):
    """Loads the list of DAG IDs to skip from the YAML config."""
    if not os.path.exists(self.WHITELIST_PATH):
      return set()
    with open(self.WHITELIST_PATH, "r", encoding="utf-8") as f:
      config = yaml.safe_load(f)
      return set(config.get("whitelisted_dags", []))

  def test_registration_check(self):
    """
    Fails if a DAG is NOT whitelisted
    and NOT registered in scheduling_helper.
    """
    # 1. Load all DAGs found in the project
    dagbag = DagBag(dag_folder=self.DAG_ROOT, include_examples=False)
    all_found_dag_ids = set(dagbag.dag_ids)

    # 2. Get IDs from the helper and the whitelist
    registered_ids = set()
    for dags_dict in scheduling_helper.REGISTERED_DAGS.values():
      registered_ids.update(dags_dict.keys())

    whitelisted_ids = self._load_whitelist()

    # 3. Logic: (All DAGs) - (Whitelisted) = Must be registered
    required_to_register = all_found_dag_ids - whitelisted_ids
    missing_from_helper = required_to_register - registered_ids

    self.assertEmpty(
        missing_from_helper,
        msg=(
            f"Registration Failed: The following new DAGs were found "
            f"but are not registered: {missing_from_helper}."
            f"Either register them or add them to {self.WHITELIST_PATH}."
        ),
    )

    # 4. Check for "Stale" entries in the whitelist
    stale_whitelist = whitelisted_ids - all_found_dag_ids
    if stale_whitelist:
      print(f"Warning: Whitelist contains non-existent DAGs: {stale_whitelist}")


if __name__ == "__main__":
  absltest.main()
