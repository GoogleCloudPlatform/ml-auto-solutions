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

"""Tests for xpk_cluster_config.py."""

import unittest
from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags.common.vm_resource import TpuVersion, Project, Zone


class XpkClusterConfigTest(unittest.TestCase):

  def test_override_core_count(self):
    config = XpkClusterConfig(
        name="mlperf-v5p",
        device_version=TpuVersion.V5P,
        core_count=8,
        project=Project.CLOUD_TPU_MULTIPOD_DEV.value,
        zone=Zone.EUROPE_WEST4_B.value,
    )
    overridden = config.override(core_count=128)

    # Verify immutability
    self.assertEqual(config.core_count, 8)
    self.assertEqual(overridden.core_count, 128)
    self.assertEqual(overridden.name, "mlperf-v5p")
    self.assertEqual(overridden.device_version, TpuVersion.V5P)
    self.assertEqual(overridden.project, Project.CLOUD_TPU_MULTIPOD_DEV.value)
    self.assertEqual(overridden.zone, Zone.EUROPE_WEST4_B.value)


if __name__ == "__main__":
  unittest.main()
