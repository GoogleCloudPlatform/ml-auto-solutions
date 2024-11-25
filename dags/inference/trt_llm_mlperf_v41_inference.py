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

"""DAGs to run TensorRT-LLM MLPerf Inference benchmarks on multiple GPUs."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import A100_INFERENCE_SUBNETWORKS, GpuVersion, Zone, ImageFamily, ImageProject, MachineVersion, Project, INFERENCE_NETWORKS, L4_INFERENCE_SUBNETWORKS
from dags.inference.configs import trt_llm_mlperf_v41_config

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="trt_llm_mlperf_v41",
    schedule=SCHEDULED_TIME,
    tags=[
        "inference_team",
        "tensorrt_llm",
        "mlperf",
        "nightly",
        "benchmark",
    ],
    start_date=datetime.datetime(2024, 9, 9),
    catchup=False,
) as dag:
  test_name_prefix = "tensorrt-llm-mlperf-v41-inference"

  config_ver = "default,high_accuracy"
  test_mode = "PerformanceOnly"
  scenario = "Offline,Server"
  g2_configs = {
      "model_name": "bert",
      "scenario": scenario,
      "config_ver": config_ver,
      "test_mode": test_mode,
      "docker_config": "gs://yijiaj/mlperf/config.json",
      "models": "gs://yijiaj/mlperf/g2/models",
      "preprocessed_data": "gs://yijiaj/mlperf/g2/preprocessed_data",
  }
  g2_model_parameters = {
      "bert": {
          "Offline": {
              "offline_expected_qps": (1000, 1200),
          },
          "Server": {
              "server_target_qps": (900, 1200),
          },
      },
  }
  g2_parameter_position = {
      "bert": {
          "Offline": {
              "offline_expected_qps": 309,
          },
          "Server": {
              "server_target_qps": 278,
          },
      },
  }
  a2_configs = {
      "model_name": "bert",
      "scenario": scenario,
      "config_ver": config_ver,
      "test_mode": test_mode,
      "docker_config": "gs://yijiaj/mlperf/config.json",
      "models": "gs://yijiaj/mlperf/a2/models",
      "preprocessed_data": "gs://yijiaj/mlperf/a2/preprocessed_data",
  }
  a2_model_parameters = {
      "bert": {
          "Offline": {
              "offline_expected_qps": (27000, 27500),
          },
          "Server": {
              "server_target_qps": (25400, 25600),
          },
      },
  }
  a2_parameter_position = {
      "bert": {
          "Offline": {
              "offline_expected_qps": 411,
          },
          "Server": {
              "server_target_qps": 560,
          },
      },
  }

  # Running on A100 GPU
  trt_llm_mlperf_v41_config.get_trt_llm_mlperf_gpu_config(
      machine_type=MachineVersion.A2_ULTRAGPU_8G,
      image_project=ImageProject.ML_IMAGES,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.A100_80G,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_C,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-test-a100-8",
      project=Project.CLOUD_TPU_INFERENCE_TEST,
      network=INFERENCE_NETWORKS,
      subnetwork=A100_INFERENCE_SUBNETWORKS,
      general_configs=a2_configs,
      model_parameters=a2_model_parameters,
      parameter_positions=a2_parameter_position,
      binary_search_steps=2,
  ).run()

  # Running on L4 GPU
  trt_llm_mlperf_v41_config.get_trt_llm_mlperf_gpu_config(
      machine_type=MachineVersion.G2_STAND_96,
      image_project=ImageProject.ML_IMAGES,
      image_family=ImageFamily.COMMON_CU121_DEBIAN_11,
      accelerator_type=GpuVersion.L4,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_A,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-test-l4-1",
      project=Project.CLOUD_TPU_INFERENCE_TEST,
      network=INFERENCE_NETWORKS,
      subnetwork=L4_INFERENCE_SUBNETWORKS,
      general_configs=g2_configs,
      model_parameters=g2_model_parameters,
      parameter_positions=g2_parameter_position,
      binary_search_steps=2,
  ).run()
