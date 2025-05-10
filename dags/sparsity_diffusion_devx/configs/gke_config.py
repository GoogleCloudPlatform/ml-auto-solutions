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

from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags import gcs_bucket
from dags.common.vm_resource import TpuVersion, Project, XpkClusters, GpuVersion, CpuVersion, Zone
from typing import Iterable, Optional
import datetime

clusters = {
    # accelerator: cluster names
    "v4-8": XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
    "v4-16": XpkClusters.TPU_V4_16_CLUSTER,
    "v5-8": XpkClusters.TPU_V5P_8_CLUSTER,
    "v6e-256": XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
    "a3": XpkClusters.GPU_A3_CLUSTER,
    "a3plus": XpkClusters.GPU_A3PLUS_CLUSTER,
}


def get_current_datetime() -> str:
  current_time = datetime.datetime.now()
  current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")
  return current_datetime


def get_gke_config(
    time_out_in_min: int,
    test_name: str,
    docker_image: str,
    run_model_cmds: Iterable[str],
    test_owner: str,
    cluster: XpkClusterConfig = XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
    num_slices: int = 1,
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.XLML_DATASET,
    dataset_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    composer_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    tensorboard_summary_config: Optional[metric_config.SummaryConfig] = None,
) -> task.XpkTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=dataset_name,
      dataset_project=dataset_project,
      composer_project=composer_project,
  )

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=cluster.device_version,
          cores=cluster.core_count,
      ),
      test_name=test_name,
      run_model_cmds=run_model_cmds,
      set_up_cmds=None,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner,
      num_slices=num_slices,
      cluster_name=cluster.name,
      docker_image=docker_image,
  )

  job_metric_config = None
  if tensorboard_summary_config:
    job_metric_config = metric_config.MetricConfig(
        tensorboard_summary=tensorboard_summary_config,
    )

  return task.XpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


def get_gpu_gke_test_config(
    time_out_in_min: int,
    test_name: str,
    run_model_cmds: str,
    cluster: XpkClusterConfig,
    test_owner: str,
    docker_image: str,
    num_slices: int = 1,
) -> task.GpuCreateResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  job_test_config = test_config.GpuXpkTest(
      test_config.Gpu(
          machine_type=None,
          image_family=None,
          count=None,
          accelerator_type=cluster.device_version.value,
          runtime_version=None,
      ),
      test_name=test_name,
      set_up_cmds=None,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner,
      cluster_name=cluster.name,
      docker_image=docker_image,
      num_slices=num_slices,
  )

  return task.XpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
