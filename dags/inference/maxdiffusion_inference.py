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

"""A DAG to run Maxdiffusion inference benchmarks"""

import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import maxdiffusion_inference_gce_config
from dags.multipod.configs.common import SetupMode, Platform


# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxdiffusion_inference",
    schedule=SCHEDULED_TIME,
    tags=["inference_team", "maxdiffusion", "nightly", "benchmark"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxdiffusion-inference"
  test_models = {
      "SDXL-Base-1.0": {
          "model_configs": [
              (TpuVersion.V5E, 8, [1, 2], ["dot_attention", "flash"]),
              (TpuVersion.V5E, 4, [1, 2], ["dot_attention", "flash"]),
              (
                  TpuVersion.V5P,
                  8,
                  [2, 10, 20, 40, 80, 320],
                  "dot_attention",
                  "flash",
              ),
          ]
      },
      "SDXL-Lightning": {
          "model_configs": [
              (TpuVersion.V5E, 8, [1, 2], ["dot_attention", "flash"]),
              (TpuVersion.V5E, 4, [1, 2], ["dot_attention", "flash"]),
              (
                  TpuVersion.V5P,
                  8,
                  [2, 10, 20, 40, 80, 320],
                  ["dot_attention", "flash"],
              ),
          ]
      },
      "SDXL-ContolNet": {
          "model_configs": [
              (TpuVersion.V5E, 8, [1], ["dot_attention", "flash"]),
              (TpuVersion.V5E, 4, [1], ["dot_attention", "flash"]),
              (
                  TpuVersion.V5P,
                  8,
                  [2, 10, 20, 40, 80],
                  ["dot_attention", "flash"],
              ),
          ]
      },
  }

  for model, sweep_model_configs in test_models.items():
    # tasks_per_model = []
    for tpu_version, tpu_cores, per_device_batch_sizes, attentions in sweep_model_configs["model_configs"]:
      for per_device_batch_size in per_device_batch_sizes:
        for attention in attentions:
          model_configs = {}
          model_configs["model_name"] = model
          model_configs["per_device_batch_size"] = per_device_batch_size
          model_configs["attention"] = attention

          if tpu_version == TpuVersion.V5E:
            # v5e benchmarks
            project_name = Project.TPU_PROD_ENV_AUTOMATED.value
            zone = Zone.US_EAST1_C.value
            network = V5_NETWORKS
            subnetwork = V5E_SUBNETWORKS
            runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value
          elif tpu_version == TpuVersion.V5P:
            zone = Zone.US_EAST5_A.value
            runtime_version = RuntimeVersion.V2_ALPHA_TPUV5.value
            project_name = Project.TPU_PROD_ENV_AUTOMATED.value
            network = V5_NETWORKS
            subnetwork = V5P_SUBNETWORKS

          maxdiffusion_stable_1slice = maxdiffusion_inference_gce_config.get_maxdiffusion_inference_nightly_config(
              tpu_version=tpu_version,
              tpu_cores=tpu_cores,
              tpu_zone=zone,
              runtime_version=runtime_version,
              project_name=project_name,
              time_out_in_min=60,
              is_tpu_reserved=True,
              test_name=f"{test_name_prefix}-stable-{model}-per_device_batch_size-{per_device_batch_size}",
              test_mode=SetupMode.STABLE,
              network=network,
              subnetwork=subnetwork,
              model_configs=model_configs,
          ).run()
          maxdiffusion_nightly_1slice = maxdiffusion_inference_gce_config.get_maxdiffusion_inference_nightly_config(
              tpu_version=tpu_version,
              tpu_cores=tpu_cores,
              tpu_zone=zone,
              runtime_version=runtime_version,
              project_name=project_name,
              time_out_in_min=60,
              is_tpu_reserved=True,
              test_name=f"{test_name_prefix}-nightly-{model}-per_device_batch_size-{per_device_batch_size}",
              test_mode=SetupMode.NIGHTLY,
              network=network,
              subnetwork=subnetwork,
              model_configs=model_configs,
          ).run()
          maxdiffusion_stable_1slice >> maxdiffusion_nightly_1slice
