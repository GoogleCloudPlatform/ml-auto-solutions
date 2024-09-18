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

"""Config file for XPK cluster."""

import dataclasses
from typing import Union


@dataclasses.dataclass
class XpkClusterConfig:
  """Defines common XPK cluster attributes.

  Attributes:
    name: Name of the cluster
    device_version: Device version of the cluster
    core_count: Core count of the cluster
    project: Project of the cluster
    zone: Zone of the cluster
  """

  name: str
  device_version: Union['GpuVersion', 'CpuVersion', 'TpuVersion']
  core_count: int
  project: str
  zone: str
