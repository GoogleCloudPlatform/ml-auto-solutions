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

"""Utilities to construct configs for solutionsteam_jax_bite DAG."""


import datetime
from typing import Tuple, Optional
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket
from dags.sparsity_diffusion_devx.configs import common
from dags.common.vm_resource import TpuVersion, Project
from airflow.models.taskmixin import DAGNode


GCS_SUBFOLDER_PREFIX = test_owner.Team.SPARSITY_DIFFUSION_DEVX.value


def set_up_axlearn(pinned_version, jax_version) -> Tuple[str]:
  reset_version = ""
  if pinned_version:
    reset_version = f"cd axlearn && git reset --hard {pinned_version} && cd .."

  setup_jax = None
  if jax_version:
    setup_jax = common.set_up_jax_version(jax_version)
  else:
    setup_jax = common.set_up_nightly_jax()

  return (
      common.UPGRADE_PIP,
      common.UPGRADE_SETUPTOOLS,
      common.UPGRADE_PACKAGING,
      "git clone https://github.com/lkolluru05/axlearn",
      reset_version,
      "python -m pip install ./axlearn[core,dev]",
      "python -m pip install ./axlearn[gcp]",
      *setup_jax,
  )


def get_axlearn_tpu_config(
    cluster_name: str,
    docker_image: str,
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    runtime_version: str,
    model_config: str,
    time_out_in_min: int,
    task_owner: str,
    num_replica: int,
    is_tpu_reserved: bool = False,
    project_name: Optional[Project] = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    network: str = "default",
    subnetwork: str = "default",
) -> task.AxlearnTask:
  """Setup the axlearn tpu env config."""
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )
  test_name = f"axl-{model_config.lower()}"

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
      ),
      test_name=test_name,
      run_model_cmds=None,
      set_up_cmds=None,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner,
      num_slices=num_replica,
      cluster_name=cluster_name,
      docker_image=docker_image,
  )

  return task.AxlearnTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
