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

"""A DAG to run TensorRT-LLM inference benchmarks with nightly version."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import GpuVersion, Zone, ImageFamily, ImageProject, MachineVersion
from dags.inference.configs import tensorrt_llm_inference_config
from dags.multipod.configs.common import SetupMode, Platform

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="tensorrt_llm_inference",
    schedule=SCHEDULED_TIME,
    tags=["inference_team", "tensorrt_llm", "nightly", "benchmark"],
    start_date=datetime.datetime(2024, 5, 10),
    catchup=False,
) as dag:
  test_name_prefix = "tensorrt-llm-inference"

  # Running on H100 GPU
  tensorrt_llm_inference_config.get_tensorrt_llm_gpu_config(
      machine_type=MachineVersion.A3_HIGHGPU_8G,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.H100,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_A,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-llama-7b-h100-8",
  ).run()

  # # Running on L4 GPU
  # tensorrt_llm_inference_config.get_tensorrt_llm_gpu_config(
  #     machine_type=MachineVersion.G2_STAND_4,
  #     image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
  #     image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
  #     accelerator_type=GpuVersion.L4,
  #     count=1,
  #     gpu_zone=Zone.US_CENTRAL1_C,
  #     time_out_in_min=1600,
  #     test_name=f"{test_name_prefix}-nightly-llama-7b-l4-1",
  # ).run()

  # # Running on A100 GPU
  # tensorrt_llm_inference_config.get_tensorrt_llm_gpu_config(
  #     machine_type=MachineVersion.A2_HIGHGPU_1G,
  #     image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
  #     image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
  #     accelerator_type=GpuVersion.A100,
  #     count=1,
  #     gpu_zone=Zone.US_CENTRAL1_F,
  #     time_out_in_min=1600,
  #     test_name=f"{test_name_prefix}-nightly-llama-7b-a100-1",
  # ).run()

  # Running on V100 GPU
  tensorrt_llm_inference_config.get_tensorrt_llm_gpu_config(
      machine_type=MachineVersion.N1_STANDARD_8,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.V100,
      count=1,
      gpu_zone=Zone.US_CENTRAL1_C,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-llama-7b-v100-1",
  ).run()
