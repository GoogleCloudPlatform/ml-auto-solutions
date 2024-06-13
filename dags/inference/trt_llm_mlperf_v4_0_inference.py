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

"""A DAG to run TensorRT-LLM MLPerf4.0 Inference benchmarks."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import GpuVersion, Zone, ImageFamily, ImageProject, MachineVersion
from dags.inference.configs import trt_llm_mlperf_v4_0_config

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="trt_llm_mlperf_4.0",
    schedule=SCHEDULED_TIME,
    tags=[
        "inference_team",
        "tensorrt_llm",
        "mlperf_v4.0",
        "nightly",
        "benchmark",
    ],
    start_date=datetime.datetime(2024, 6, 12),
    catchup=False,
) as dag:
  test_name_prefix = "tensorrt-llm-mlperf-v4.0-inference"

  model_name = "llama2-70b"
  # Running on H100 GPU
  trt_llm_mlperf_v4_0_config.get_trt_llm_mlperf_v4_0_gpu_config(
      machine_type=MachineVersion.A3_HIGHGPU_8G,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.H100,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_A,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-{model_name}-h100-8",
      model_configs=dict(model_name=model_name),
  ).run()

  # Running on A100 GPU
  trt_llm_mlperf_v4_0_config.get_trt_llm_mlperf_v4_0_gpu_config(
      machine_type=MachineVersion.A2_HIGHGPU_1G,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.A100,
      count=1,
      gpu_zone=Zone.US_CENTRAL1_F,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-{model_name}-a100-1",
      model_configs=dict(model_name=model_name),
  ).run()

  # Running on V100 GPU
  trt_llm_mlperf_v4_0_config.get_trt_llm_mlperf_v4_0_gpu_config(
      machine_type=MachineVersion.N1_STANDARD_8,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.V100,
      count=1,
      gpu_zone=Zone.US_CENTRAL1_C,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-{model_name}-v100-1",
      model_configs=dict(model_name=model_name),
  ).run()
