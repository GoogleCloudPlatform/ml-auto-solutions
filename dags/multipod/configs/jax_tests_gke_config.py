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

"""Utilities to construct configs for JAX tests for GCE."""

from dags import test_owner
from dags.multipod.configs import gke_config
from dags.vm_resource import XpkClusterConfig


def get_jax_distributed_initialize_config(
    cluster: XpkClusterConfig,
    time_out_in_min: int,
    test_name: str,
    docker_image: str,
    num_slices: int = 1,
):
  run_model_cmds = [
      "bash end_to_end/test_jdi.sh",
  ]

  return gke_config.get_gke_config(
      cluster=cluster,
      test_name=test_name,
      run_model_cmds=run_model_cmds,
      num_slices=num_slices,
      docker_image=docker_image,
      test_owner=test_owner.AKANKSHA_G,
      time_out_in_min=time_out_in_min,
  )
