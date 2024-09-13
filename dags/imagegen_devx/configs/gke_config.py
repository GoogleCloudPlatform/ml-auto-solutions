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

from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner, gcs_bucket
from dags.vm_resource import TpuVersion, Project, ClusterName, GpuVersion, CpuVersion, Zone
from typing import Iterable
import datetime

tpu_versions = {
    # accelerator: tpu versions
    "v4-8": TpuVersion.V4,
    "v4-16": TpuVersion.V4,
    "v5-8": TpuVersion.V5P,
    "v6e-256": TpuVersion.TRILLIUM,
}
cluster_names = {
    # accelerator: cluster names
    "v4-8": ClusterName.V4_8_MULTISLICE_CLUSTER,
    "v4-16": ClusterName.V4_16_MULTISLICE_CLUSTER,
    "v5-8": ClusterName.V5P_8_MULTISLICE_CLUSTER,
    "v6e-256": ClusterName.BODABORG_V6E_256_EUROPE_WEST4_A,
}
tpu_zones = {
    # accelerator: cluster name
    "v4-8": Zone.US_CENTRAL2_B,
    "v4-16": Zone.US_CENTRAL2_B,
    "v5-8": Zone.US_EAST5_A,
    "v6e-256": Zone.EUROPE_WEST4_A,
}
project_names = {
    # accelerator: project names
    "v4-8": Project.TPU_PROD_ENV_MULTIPOD,
    "v4-16": Project.TPU_PROD_ENV_MULTIPOD,
    "v5-8": Project.CLOUD_TPU_MULTIPOD_DEV,
    "v6e-256": Project.TPU_PROD_ENV_LARGE_ADHOC,
}


def get_current_datetime() -> str:
  current_time = datetime.datetime.now()
  current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")
  return current_datetime


def get_gke_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    docker_image: str,
    run_model_cmds: Iterable[str],
    test_owner: str,
    cluster_name: str = ClusterName.V4_8_MULTISLICE_CLUSTER.value,
    project_name: str = Project.TPU_PROD_ENV_MULTIPOD.value,
    num_slices: int = 1,
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.XLML_DATASET,
    dataset_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    composer_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
) -> task.XpkTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=dataset_name,
      dataset_project=dataset_project,
      composer_project=composer_project,
  )

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
      ),
      test_name=test_name,
      run_model_cmds=run_model_cmds,
      set_up_cmds=None,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner,
      num_slices=num_slices,
      cluster_name=cluster_name,
      docker_image=docker_image,
  )

  return task.XpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
