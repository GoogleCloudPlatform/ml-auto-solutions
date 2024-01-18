# Copyright 2023 Google LLC
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

import datetime
from airflow import models
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from airflow.utils.state import State
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import composer_env, gcs_bucket, vm_resource
from dags.multipod.pytorch.configs import llama as llama_config
from dags.pytorch_xla.pytorchxla_llama import DAG_ID as SINGLE_SLICE_DAG_ID

# Run Sundays at 12 pm UTC (4 am PST)
SCHEDULED_TIME = "0 12 0 * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="multipod-pytorch-llama",
    schedule=SCHEDULED_TIME,
    tags=["multipod", "pytorch", "supported", "xlml"],
    start_date=datetime.datetime(2024, 1, 3),
    catchup=False,
):
  if composer_env.is_prod_env():
    # Ensure single-slice tests are passing before running multislice
    single_slice_sensor = ExternalTaskSensor(
        task_id="single-slice-sensor",
        external_dag_id=SINGLE_SLICE_DAG_ID,
    )
  else:
    single_slice_sensor = EmptyOperator(task_id="single-slice-sensor")

  docker_image_build = task.DockerBuildTask(
      build_dir="dags/multipod/pytorch",
      image_name="gcr.io/cloud-ml-auto-solutions/pytorch-multislice-baseline",
  ).run()

  gcs_bucket_prefix = f"{gcs_bucket.XLML_OUTPUT_DIR}/multipod/pytorch/nightly/perf"
  llama_perf_1_slice = llama_config.get_pytorch_llama2_perf_config(
      tpu_version=vm_resource.TpuVersion.V5E,
      tpu_cores="256",
      num_slices=1,
      tp_axis=1,
      per_slice_batch_size=512,
      docker_image=docker_image_build.output,
      gcs_bucket_prefix=gcs_bucket_prefix,
      tpu_zone=vm_resource.Zone.US_EAST5_B.value,
      cluster_name=vm_resource.ClusterName.V5E_256_MULTISLICE_CLUSTER.value,
      cluster_project_name=vm_resource.Project.TPU_PROD_ENV_MULTIPOD.value,
      config_name="70B",
  ).run()

  llama_perf_2_slice = llama_config.get_pytorch_llama2_perf_config(
      tpu_version=vm_resource.TpuVersion.V5E,
      tpu_cores="256",
      num_slices=2,
      tp_axis=1,
      per_slice_batch_size=512,
      docker_image=docker_image_build.output,
      gcs_bucket_prefix=gcs_bucket_prefix,
      tpu_zone=vm_resource.Zone.US_EAST5_B.value,
      cluster_name=vm_resource.ClusterName.V5E_256_MULTISLICE_CLUSTER.value,
      cluster_project_name=vm_resource.Project.TPU_PROD_ENV_MULTIPOD.value,
      config_name="70B",
  ).run()

  single_slice_sensor >> docker_image_build >> llama_perf_1_slice >> llama_perf_2_slice
