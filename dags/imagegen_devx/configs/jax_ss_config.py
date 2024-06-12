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
from dags.vm_resource import TpuVersion, Project, ClusterName, GpuVersion, CpuVersion
from typing import Iterable
import datetime


def get_gke_maxtext_jax_ss_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    docker_image: str,
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

  current_time = datetime.datetime.now()
  current_date = current_time.strftime("%Y-%m-%d")
  current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext/jax-ss/automated/{current_date}"
  )
  run_name = f"{num_slices}slice-V{tpu_version.value}_{tpu_cores}-maxtext-jax-ss-{current_datetime}"

  run_model_cmds = (
      f"python MaxText/train.py MaxText/configs/base.yml run_name={run_name} "
      "steps=30 per_device_batch_size=1 max_target_length=4096 model_name=llama2-7b "
      "enable_checkpointing=false attention=dot_product remat_policy=minimal_flash use_iota_embed=true scan_layers=false "
      "dataset_type=synthetic async_checkpointing=false "
      f"base_output_directory={base_output_directory}",
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


def get_gke_maxdiffusion_jax_ss_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    docker_image: str,
    test_owner: str,
    cluster_name: str = ClusterName.V4_8_MULTISLICE_CLUSTER.value,
    project_name: str = Project.TPU_PROD_ENV_MULTIPOD.value,
    num_slices: int = 1,
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.XLML_DATASET,
    dataset_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    composer_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    base_output_directory: str = None,
    metric_aggregation_strategy: metric_config.AggregationStrategy = None,
    user_specified_job_metric_config: metric_config.MetricConfig = None,
) -> task.XpkTask:
  current_time = datetime.datetime.now()
  current_date = current_time.strftime("%Y-%m-%d")
  current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext/jax-ss/automated/{current_date}"
  )
  run_name = f"{num_slices}slice-V{tpu_version.value}_{tpu_cores}-maxtext-jax-ss-{current_datetime}"

  run_model_cmds = (
      f"python -m src.maxdiffusion.models.train src/maxdiffusion/configs/base_2_base.yml run_name={run_name} base_output_directory={base_output_directory}",
  )
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
  job_metric_config = user_specified_job_metric_config
  if job_metric_config is None:
    job_metric_config = (
        metric_config.MetricConfig(
            tensorboard_summary=metric_config.SummaryConfig(
                file_location=base_output_directory,
                aggregation_strategy=metric_aggregation_strategy,
                use_regex_file_location=True,
            ),
        )
        if base_output_directory and metric_aggregation_strategy
        else None
    )

  return task.XpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
