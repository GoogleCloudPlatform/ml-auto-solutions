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
from configs import composer_env
from configs.benchmark.pytorch import pytorchxla_torchbench_config as config
import configs.vm_resource as vm
from configs.vm_resource import GpuVersion, ImageFamily, ImageProject, MachineVersion, RuntimeVersion, Project, TpuVersion, Zone

# Schudule the job to run once per two days at 5:00PM UTC.
SCHEDULED_TIME = "0 17 */2 * *" if composer_env.is_prod_env() else None


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
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      runtime_version=RuntimeVersion.TPU_UBUNTU2204_BASE.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V5P
  config.get_torchbench_tpu_config(
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      project_name=Project.TPU_PROD_ENV_AUTOMATED,
      tpu_zone=Zone.US_EAST5_A.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
      network=vm.V5_NETWORKS,
      subnetwork=vm.V5E_SUBNETWORKS,
      time_out_in_min=700,
      model_name=model,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V5E
  config.get_torchbench_tpu_config(
      tpu_version=TpuVersion.V5E,
      tpu_cores=4,
      project_name=Project.TPU_PROD_ENV_AUTOMATED,
      tpu_zone=Zone.US_EAST1_C.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5_LITE.value,
      network=vm.V5_NETWORKS,
      subnetwork=vm.V5E_SUBNETWORKS,
      time_out_in_min=1600,
      model_name=model,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on V100 GPU
  config.get_torchbench_gpu_config(
      machine_type=MachineVersion.N1_STANDARD_32,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.V100,
      count=4,
      gpu_zone=Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on A100 GPU
  config.get_torchbench_gpu_config(
      machine_type=MachineVersion.A2_HIGHGPU_4G,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.A100,
      count=4,
      gpu_zone=Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on H100 GPU
  config.get_torchbench_gpu_config(
      machine_type=MachineVersion.A3_HIGHGPU_8G,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.H100,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on L4 GPU
  config.get_torchbench_gpu_config(
      machine_type=MachineVersion.G2_STAND_4,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.L4,
      count=1,
      gpu_zone=Zone.US_CENTRAL1_C.value,
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()
