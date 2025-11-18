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
from dags.common.vm_resource import H200_INFERENCE_SUBNETWORKS, INFERENCE_NETWORKS, GpuVersion, Zone, ImageFamily, ImageProject, MachineVersion, Project
from dags.inference.configs import trtllm_bench_inference_config

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="trtllm_bench_inference",
    schedule=SCHEDULED_TIME,
    tags=[
        "inference_team",
        "tensorrt_llm",
        "nightly",
        "benchmark",
        "GPU",
        "a3-ultragpu-8g",
        "nvidia-h200-80gb",
    ],
    start_date=datetime.datetime(2025, 1, 25),
    catchup=False,
) as dag:
  test_name_prefix = "trtllm_bench_inference"

  # Running on H200 GPU
  trtllm_bench_inference_config.get_trtllm_bench_config(
      machine_type=MachineVersion.A3_ULTRAGPU_8G,
      image_project=ImageProject.ML_IMAGES,
      image_family=ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=GpuVersion.H200,
      count=8,
      gpu_zone=Zone.EUROPE_WEST1_B,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-h200-8",
      project=Project.CLOUD_TPU_INFERENCE_TEST,
      network=INFERENCE_NETWORKS,
      subnetwork=H200_INFERENCE_SUBNETWORKS,
      existing_instance_name="yijiaj-a3u-test-h200x8",
  ).run()
