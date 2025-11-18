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

from airflow import models
import datetime
from dags import composer_env
from dags.pytorch_xla.configs import pytorchxla_torchbench_config as config
import dags.common.vm_resource as resource

SCHEDULED_TIME = None


with models.DAG(
    dag_id="pytorchxla-torchbench-release-gpu",
    schedule=SCHEDULED_TIME,
    tags=[
        "pytorchxla",
        "release",
        "torchbench",
        "GPU",
        "nvidia-tesla-v100",
        "n1-standard-16",
        "nvidia-tesla-a100",
        "a3-highgpu-8g",
        "nvidia-h100-80gb",
        "a2-highgpu-1g",
        "nvidia-l4",
        "g2-standard-16",
    ],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
  model = "all" if composer_env.is_prod_env() else "BERT_pytorch"
  torchbench_extra_flags = [f"--filter={model}"]
  test_version = config.VERSION.R2_8
  # Running on V100 GPU
  config.get_torchbench_gpu_gke_config(
      machine_type=resource.MachineVersion.N1_STANDARD_16,
      image_family=resource.ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=resource.GpuVersion.V100,
      count=2,
      gpu_zone=resource.Region.US_CENTRAL1,
      test_version=test_version,
      project_name=resource.Project.CLOUD_ML_BENCHMARKING,
      cluster_name="benchmarking-gpu-uc1",
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on A100 GPU
  config.get_torchbench_gpu_gke_config(
      machine_type=resource.MachineVersion.A2_HIGHGPU_1G,
      image_family=resource.ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=resource.GpuVersion.A100,
      count=1,
      gpu_zone=resource.Region.US_CENTRAL1,
      test_version=test_version,
      project_name=resource.Project.CLOUD_ML_BENCHMARKING,
      cluster_name="benchmarking-gpu-uc1",
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on H100 GPU
  config.get_torchbench_gpu_gke_config(
      machine_type=resource.MachineVersion.A3_HIGHGPU_8G,
      image_family=resource.ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=resource.GpuVersion.H100,
      count=8,
      gpu_zone=resource.Region.US_CENTRAL1,
      test_version=test_version,
      project_name=resource.Project.CLOUD_ML_BENCHMARKING,
      cluster_name="benchmarking-gpu-uc1",
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()

  # Running on L4 GPU
  config.get_torchbench_gpu_gke_config(
      machine_type=resource.MachineVersion.G2_STAND_16,
      image_family=resource.ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=resource.GpuVersion.L4,
      count=1,
      gpu_zone=resource.Region.US_CENTRAL1,
      test_version=test_version,
      project_name=resource.Project.CLOUD_ML_BENCHMARKING,
      cluster_name="benchmarking-gpu-uc1",
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()
