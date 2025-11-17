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
    dag_id="pytorchxla-torchbench-release-tpu",
    schedule=SCHEDULED_TIME,
    tags=[
        "pytorchxla",
        "release",
        "torchbench",
        "TPU",
        "v4-8",
        "v5e-4",
        "v5p-8",
    ],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
  model = "all" if composer_env.is_prod_env() else "BERT_pytorch"
  torchbench_extra_flags = [f"--filter={model}"]
  test_version = config.VERSION.R2_8
  # Running on V4-8:
  config.get_torchbench_tpu_config(
      tpu_version=resource.TpuVersion.V4,
      tpu_cores=8,
      project=resource.Project.CLOUD_ML_BENCHMARKING,
      tpu_zone=resource.Zone.US_CENTRAL2_B,
      runtime_version=resource.RuntimeVersion.TPU_UBUNTU2204_BASE,
      network=resource.BM_NETWORKS,
      subnetwork=resource.V4_BM_SUBNETWORKS,
      test_version=test_version,
      model_name=model,
      time_out_in_min=1800,
      reserved=False,
      preemptible=True,
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
      reserved=False,
      preemptible=False,
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
      test_version=test_version,
      model_name=model,
      reserved=False,
      preemptible=False,
      extraFlags=" ".join(torchbench_extra_flags),
  )
