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
from dags.common.vm_resource import A100_INFERENCE_SUBNETWORKS, H100_INFERENCE_SUBNETWORKS, GpuVersion, Zone, ImageFamily, ImageProject, MachineVersion, Project, INFERENCE_NETWORKS, L4_INFERENCE_SUBNETWORKS
from dags.inference.configs import trt_llm_mlperf_v41_config

# Run once a day at 1 pm UTC (5 am PST)
SCHEDULED_TIME = "1 3 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="trt_llm_mlperf_v41",
    schedule=SCHEDULED_TIME,
    tags=[
        "inference_team",
        "tensorrt_llm",
        "mlperf",
        "nightly",
        "benchmark",
        "GPU",
        "a2-ultragpu-8g",
        "nvidia-a100-80gb",
        "g2-standard-96",
        "nvidia-l4",
        "a3-highgpu-8g",
        "nvidia-h100-80gb",
    ],
    start_date=datetime.datetime(2024, 9, 9),
    catchup=False,
) as dag:
  test_name_prefix = "tensorrt-llm-mlperf-v41-inference"

  config_ver = "default,high_accuracy"
  test_mode = "PerformanceOnly"
  g2_configs = {
      "model_name": "bert,3d-unet",
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
      "3d-unet": {
          "Offline": {
              "offline_expected_qps": (1.3, 2.6),
          },
      },
      "dlrm-v2": {
          "Offline": {
              "offline_expected_qps": (3400, 3500),
          },
          "Server": {
              "server_target_qps": (3300, 3500),
          },
      },
      "gptj": {
          "Offline": {
              "offline_expected_qps": (1.3, 1.6),
          },
          "Server": {
              "server_target_qps": (0.88, 1),
          },
      },
      "resnet50": {
          "Offline": {
              "offline_expected_qps": (13000, 15000),
          },
          "Server": {
              "server_target_qps": (11532.8125, 11600),
          },
      },
      "retinanet": {
          "Offline": {
              "offline_expected_qps": (220, 230),
          },
          "Server": {
              "server_target_qps": (200, 220),
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
      "3d-unet": {
          "Offline": {
              "offline_expected_qps": 55,
          },
      },
      "dlrm-v2": {
          "Offline": {
              "offline_expected_qps": 233,
          },
          "Server": {
              "server_target_qps": 176,
          },
      },
      "gptj": {
          "Offline": {
              "offline_expected_qps": 191,
          },
          "Server": {
              "server_target_qps": 158,
          },
      },
      "resnet50": {
          "Offline": {
              "offline_expected_qps": 48,
          },
          "Server": {
              "server_target_qps": 52,
          },
      },
      "retinanet": {
          "Offline": {
              "offline_expected_qps": 51,
          },
          "Server": {
              "server_target_qps": 57,
          },
      },
  }
  a2_configs = {
      "model_name": "bert,3d-unet",
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
      "3d-unet": {
          "Offline": {
              "offline_expected_qps": (30, 40),
          },
      },
      "resnet50": {
          "Offline": {
              "offline_expected_qps": (340000, 360000),
          },
          "Server": {
              "server_target_qps": (290000, 299000),
          },
      },
      "retinanet": {
          "Offline": {
              "offline_expected_qps": (5840, 5980),
          },
          "Server": {
              "server_target_qps": (5600, 5800),
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
      "3d-unet": {
          "Offline": {
              "offline_expected_qps": 623,
          },
      },
      "resnet50": {
          "Offline": {
              "offline_expected_qps": 456,
          },
          "Server": {
              "server_target_qps": 396,
          },
      },
      "retinanet": {
          "Offline": {
              "offline_expected_qps": 269,
          },
          "Server": {
              "server_target_qps": 244,
          },
      },
  }
  a3_configs = {
      "model_name": "resnet50,retinanet,stable-diffusion-xl,llama2-70b,mixtral-8x7b",
      "config_ver": config_ver,
      "test_mode": test_mode,
      "docker_config": "gs://yijiaj/mlperf/config.json",
      "models": "gs://yijiaj/mlperf/a3/models",
      "preprocessed_data": "gs://yijiaj/mlperf/a3/preprocessed_data",
  }
  a3_model_parameters = {
      "bert": {
          "Offline": {
              "offline_expected_qps": (75200, 76000),
          },
          "Server": {
              "server_target_qps": (56000, 60000),
          },
      },
      "3d-unet": {
          "Offline": {
              "offline_expected_qps": (54.4, 64),
          },
      },
      "dlrm-v2": {
          "Offline": {
              "offline_expected_qps": (616000, 620000),
          },
          "Server": {
              "server_target_qps": (458203.125, 510000),
          },
      },
      "gptj": {
          "Offline": {
              "offline_expected_qps": (288, 300),
          },
          "Server": {
              "server_target_qps": (279.36, 285),
          },
      },
      "resnet50": {
          "Offline": {
              "offline_expected_qps": (720000, 740000),
          },
          "Server": {
              "server_target_qps": (584000, 586000),
          },
      },
      "retinanet": {
          "Offline": {
              "offline_expected_qps": (13600, 14000),
          },
          "Server": {
              "server_target_qps": (12880, 13000),
          },
      },
      "stable-diffusion-xl": {
          "Offline": {
              "offline_expected_qps": (16, 18),
          },
          "Server": {
              "server_target_qps": (16.3, 18),
          },
      },
      "llama2-70b": {
          "Offline": {
              "offline_expected_qps": (80, 86),
          },
          "Server": {
              "server_target_qps": (75, 80),
          },
      },
      "mixtral-8x7b": {
          "Offline": {
              "offline_expected_qps": (368, 386),
          },
          "Server": {
              "server_target_qps": (345, 360),
          },
      },
  }
  a3_parameter_position = {
      "bert": {
          "Offline": {
              "offline_expected_qps": 196,
          },
          "Server": {
              "server_target_qps": 238,
          },
      },
      "3d-unet": {
          "Offline": {
              "offline_expected_qps": 160,
          },
      },
      "dlrm-v2": {
          "Offline": {
              "offline_expected_qps": 65,
          },
          "Server": {
              "server_target_qps": 65,
          },
      },
      "gptj": {
          "Offline": {
              "offline_expected_qps": 48,
          },
          "Server": {
              "server_target_qps": 91,
          },
      },
      "resnet50": {
          "Offline": {
              "offline_expected_qps": 84,
          },
          "Server": {
              "server_target_qps": 132,
          },
      },
      "retinanet": {
          "Offline": {
              "offline_expected_qps": 139,
          },
          "Server": {
              "server_target_qps": 127,
          },
      },
      "stable-diffusion-xl": {
          "Offline": {
              "offline_expected_qps": 55,
          },
          "Server": {
              "server_target_qps": 59,
          },
      },
      "llama2-70b": {
          "Offline": {
              "offline_expected_qps": 75,
          },
          "Server": {
              "server_target_qps": 74,
          },
      },
      "mixtral-8x7b": {
          "Offline": {
              "offline_expected_qps": 74,
          },
          "Server": {
              "server_target_qps": 64,
          },
      },
  }

  # Running on A100 GPU
  trt_llm_mlperf_v41_config.get_trt_llm_mlperf_gpu_config(
      machine_type=MachineVersion.A2_ULTRAGPU_8G,
      image_project=ImageProject.ML_IMAGES,
      image_family=ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=GpuVersion.A100_80G,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_A,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-test-a100-8",
      project=Project.CLOUD_TPU_INFERENCE_TEST,
      network=INFERENCE_NETWORKS,
      subnetwork=A100_INFERENCE_SUBNETWORKS,
      benchmark_configs=a2_configs,
      model_parameters=a2_model_parameters,
      parameter_positions=a2_parameter_position,
      binary_search_steps=2,
  ).run()

  # Running on L4 GPU
  trt_llm_mlperf_v41_config.get_trt_llm_mlperf_gpu_config(
      machine_type=MachineVersion.G2_STAND_96,
      image_project=ImageProject.ML_IMAGES,
      image_family=ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=GpuVersion.L4,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_C,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-test-l4-1",
      project=Project.CLOUD_TPU_INFERENCE_TEST,
      network=INFERENCE_NETWORKS,
      subnetwork=L4_INFERENCE_SUBNETWORKS,
      benchmark_configs=g2_configs,
      model_parameters=g2_model_parameters,
      parameter_positions=g2_parameter_position,
      binary_search_steps=2,
  ).run()

  # Running on H100 GPU
  trt_llm_mlperf_v41_config.get_trt_llm_mlperf_gpu_config(
      machine_type=MachineVersion.A3_HIGHGPU_8G,
      image_project=ImageProject.DEEP_LEARNING_PLATFORM_RELEASE,
      image_family=ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=GpuVersion.H100,
      count=8,
      gpu_zone=Zone.US_CENTRAL1_A,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-test-h100-8",
      project=Project.CLOUD_TPU_INFERENCE_TEST,
      network=INFERENCE_NETWORKS,
      subnetwork=H100_INFERENCE_SUBNETWORKS,
      benchmark_configs=a3_configs,
      model_parameters=a3_model_parameters,
      parameter_positions=a3_parameter_position,
      binary_search_steps=2,
  ).run()
