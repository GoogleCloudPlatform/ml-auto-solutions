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

"""Utilities to construct configs for PyTorch DAG on GKE."""

from typing import List
from dags.common.vm_resource import DockerImage
from dags.multipod.configs import xpk_gke_config as gke_config
from xlml.apis import task
from xlml.apis.xpk_cluster_config import XpkClusterConfig


def get_nightly_pytorch_config(
    test_name: str,
    test_owner: str,
    run_commands: List[str],
    cluster: XpkClusterConfig,
    num_slices: int,
) -> task.XpkTask:
  cmds = (
      'git clone https://github.com/pytorch/xla /pytorch/xla',
      *run_commands,
  )
  return gke_config.get_gke_config(
      cluster=cluster,
      test_name=test_name,
      run_model_cmds=cmds,
      num_slices=num_slices,
      docker_image=DockerImage.PYTORCH_NIGHTLY.value,
      test_owner=test_owner,
      time_out_in_min=60,
  )
