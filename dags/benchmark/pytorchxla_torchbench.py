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

"""A DAG to run all TorchBench tests with nightly version."""

import datetime
from airflow import models
from configs import composer_env, vm_resource
from configs.benchmark.pytorch import pytorchxla_torchbench_config as config

# Schudule the job to run once per two days at 5:00PM UTC.
SCHEDULED_TIME = "0 17 */2 * *" if composer_env.is_prod_env() else None
NETWORK_PREFIX = "projects/tpu-prod-env-automated"
V5_NETWORKS = f"{NETWORK_PREFIX}/global/networks/mas-test"
V5E_SUBNETWORKS = f"{NETWORK_PREFIX}/regions/us-east1/subnetworks/mas-test"
V5P_SUBNETWORKS = f"{NETWORK_PREFIX}/regions/us-east5/subnetworks/mas-test"


with models.DAG(
    dag_id="pytorchxla-torchbench",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "nightly", "torchbench"],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
  model = "all"
  torchbench_extra_flags = [f"--filter={model}"]
  # Running on V4-8:
  config.get_torchbench_tpu_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V5P
  config.get_torchbench_tpu_config(
      tpu_version="5p",
      tpu_cores=8,
      project_name=vm_resource.PROJECT_TPU_PROD_ENV_AUTOMATED,
      tpu_zone=vm_resource.Zone.US_EAST5_A.value,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5.value,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
      time_out_in_min=700,
      model_name=model,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V5E
  config.get_torchbench_tpu_config(
      tpu_version="5litepod",
      tpu_cores=4,
      project_name=vm_resource.PROJECT_TPU_PROD_ENV_AUTOMATED,
      tpu_zone=vm_resource.Zone.US_EAST1_C.value,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5_LITE.value,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
      time_out_in_min=1600,
      model_name=model,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V100 GPU
  config.get_torchbench_gpu_config(
      machine_type="n1-standard-32",
      image_project="deeplearning-platform-release",
      image_family="common-cu121-debian-11",
      accelerator_type="nvidia-tesla-v100",
      count=4,
      gpu_zone=vm_resource.Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on A100 GPU
  config.get_torchbench_gpu_config(
      machine_type="a2-highgpu-4g",
      image_project="deeplearning-platform-release",
      image_family="common-cu121-debian-11",
      accelerator_type="nvidia-tesla-a100",
      count=4,
      gpu_zone=vm_resource.Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on H100 GPU
  config.get_torchbench_gpu_config(
      machine_type="a3-highgpu-8g",
      image_project="deeplearning-platform-release",
      image_family="common-cu121-debian-11",
      accelerator_type="nvidia-h100-80gb",
      count=8,
      gpu_zone=vm_resource.Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on L4 GPU
  config.get_torchbench_gpu_config(
      machine_type="g2-standard-4",
      image_project="deeplearning-platform-release",
      image_family="common-cu121-debian-11",
      accelerator_type="nvidia-l4",
      count=1,
      gpu_zone=vm_resource.Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()
