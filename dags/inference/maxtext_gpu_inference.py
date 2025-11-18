# Copyright 2025 Google LLC
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

"""A DAG to run Maxtext inference benchmarks with nightly version."""

import datetime
from airflow import models
from dags import composer_env
from dags.common.vm_resource import US_EAST5_INFERENCE_SUBNETWORKS, INFERENCE_NETWORKS, GpuVersion, Zone, ImageFamily, ImageProject, MachineVersion, Project
from dags.inference.configs import maxtext_gpu_inference_config

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_gpu_inference",
    schedule=SCHEDULED_TIME,
    tags=[
        "inference_team",
        "maxtext_gpu",
        "nightly",
        "benchmark",
        "GPU",
        "a3-megagpu-8g",
        "nvidia-h100-80gb",
    ],
    start_date=datetime.datetime(2025, 3, 26),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext-gpu-inference"

  # Running on H100 GPU
  maxtext_gpu_inference_config.get_maxtext_gpu_inference_config(
      machine_type=MachineVersion.A3_MEGAGPU_8G,
      image_project=ImageProject.ML_IMAGES,
      image_family=ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=GpuVersion.H100,
      count=8,
      gpu_zone=Zone.US_EAST5_A,
      time_out_in_min=1600,
      test_name=f"{test_name_prefix}-nightly-h100-8",
      project=Project.CLOUD_TPU_INFERENCE_TEST,
      network=INFERENCE_NETWORKS,
      subnetwork=US_EAST5_INFERENCE_SUBNETWORKS,
      existing_instance_name="yijiaj-a3m-h100x8",
  ).run()
