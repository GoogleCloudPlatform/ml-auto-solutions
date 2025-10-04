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

# Schudule the job to run everyday at 3:00AM PST (11:00AM UTC).
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None


# @task_group(prefix_group_id=False)
# def llama():
#   llama_3_train_trillium = task.run_queued_resource_test(
#       test_config.JSonnetTpuVmTest.from_pytorch(
#           "pt-nightly-llama3-train-func-v6e-4-1vm",
#           network=V5_NETWORKS,
#           subnetwork=V6E_SUBNETWORKS,
#       ),
#       US_CENTRAL2_B_TPU_PROD_ENV,
#   )

with models.DAG(
    dag_id="pytorchxla-torchbench",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "nightly", "torchbench"],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
  # llama()

  model = "all" if composer_env.is_prod_env() else "BERT_pytorch"
  torchbench_extra_flags = [f"--filter={model}"]

  # LLaMA3 on V6E:
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.TRILLIUM,
      tpu_cores=8,
      project=resource.Project.CLOUD_ML_BENCHMARKING,
      tpu_zone=resource.Zone.US_CENTRAL2_B,
      runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV6,
      network=resource.BM_NETWORKS,
      subnetwork=resource.V4_BM_SUBNETWORKS,
      time_out_in_min=1600,
      model_name="llama3",
      reserved=False,
      preemptible=False,
      extraFlags=" ".join(torchbench_extra_flags),
      simple_model_test=True,
  )

  # SD2 on V6E:
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.TRILLIUM,
      tpu_cores=8,
      project=resource.Project.CLOUD_ML_BENCHMARKING,
      tpu_zone=resource.Zone.US_CENTRAL2_B,
      runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV6,
      network=resource.BM_NETWORKS,
      subnetwork=resource.V4_BM_SUBNETWORKS,
      time_out_in_min=1600,
      model_name="sd2",
      reserved=False,
      preemptible=False,
      extraFlags=" ".join(torchbench_extra_flags),
      simple_model_test=True,
  )
  
  # Running on V4-8:
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.V4,
      tpu_cores=8,
      project=resource.Project.CLOUD_ML_BENCHMARKING,
      tpu_zone=resource.Zone.US_CENTRAL2_B,
      runtime_version=resource.RuntimeVersion.TPU_UBUNTU2204_BASE,
      network=resource.BM_NETWORKS,
      subnetwork=resource.V4_BM_SUBNETWORKS,
      model_name=model,
      time_out_in_min=1800,
      reserved=False,
      preemptible=True,
      extraFlags=" ".join(torchbench_extra_flags),
  )

  # Running on V5E
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.V5E,
      tpu_cores=4,
      project=resource.Project.CLOUD_ML_BENCHMARKING,
      tpu_zone=resource.Zone.US_WEST1_C,
      runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV5_LITE,
      network=resource.BM_NETWORKS,
      subnetwork=resource.V5E_BM_SUBNETWORKS,
      time_out_in_min=1600,
      model_name=model,
      reserved=False,
      preemptible=False,
      extraFlags=" ".join(torchbench_extra_flags),
  )

  # Running on V5P
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.V5P,
      tpu_cores=8,
      project=resource.Project.TPU_PROD_ENV_AUTOMATED,
      tpu_zone=resource.Zone.US_EAST5_A,
      runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV5,
      network=resource.V5_NETWORKS,
      subnetwork=resource.V5P_SUBNETWORKS,
      time_out_in_min=1800,
      model_name=model,
      reserved=True,
      preemptible=False,
      extraFlags=" ".join(torchbench_extra_flags),
  )

  # Running on V6E
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.TRILLIUM,
      tpu_cores=8,
      project=resource.Project.CLOUD_ML_BENCHMARKING,
      tpu_zone=resource.Zone.US_CENTRAL2_B,
      runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV6,
      network=resource.BM_NETWORKS,
      subnetwork=resource.V4_BM_SUBNETWORKS,
      time_out_in_min=1600,
      model_name=model,
      reserved=False,
      preemptible=False,
      extraFlags=" ".join(torchbench_extra_flags),
  )

  # Running on V100 GPU
  config.get_torchbench_gpu_gke_config(
      machine_type=resource.MachineVersion.N1_STANDARD_16,
      image_family=resource.ImageFamily.COMMON_CU124_DEBIAN_11,
      accelerator_type=resource.GpuVersion.V100,
      count=2,
      gpu_zone=resource.Region.US_CENTRAL1,
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
      project_name=resource.Project.CLOUD_ML_BENCHMARKING,
      cluster_name="benchmarking-gpu-uc1",
      model_name=model,
      time_out_in_min=1600,
      extraFlags=" ".join(torchbench_extra_flags),
  ).run()
