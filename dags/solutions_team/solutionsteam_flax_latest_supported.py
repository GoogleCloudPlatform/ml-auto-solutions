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

"""A DAG to run all supported ML models with the latest JAX/FLAX version."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import Project, TpuVersion, Zone, RuntimeVersion, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS
from dags.solutions_team.configs.flax import solutionsteam_flax_latest_supported_config as flax_config


# Run once a day at 2 am UTC (6 pm PST)
SCHEDULED_TIME = "0 2 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="flax_latest_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "flax", "latest", "supported", "xlml"],
    start_date=datetime.datetime(2023, 8, 16),
    catchup=False,
) as dag:
  # ResNet
  jax_resnet_v2_8 = flax_config.get_flax_resnet_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v2_32 = flax_config.get_flax_resnet_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL1_A.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v3_8 = flax_config.get_flax_resnet_config(
      tpu_version=TpuVersion.V3,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST1_D.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v3_32 = flax_config.get_flax_resnet_config(
      tpu_version=TpuVersion.V3,
      tpu_cores=32,
      tpu_zone=Zone.US_EAST1_D.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v4_8 = flax_config.get_flax_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
  ).run()

  jax_resnet_v4_32 = flax_config.get_flax_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_tpu_reserved=False,
  ).run()

  jax_resnet_v5e_4 = flax_config.get_flax_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5E,
      tpu_cores=4,
      tpu_zone=Zone.US_EAST1_C.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5_LITE.value,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  jax_resnet_v5e_16 = flax_config.get_flax_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5E,
      tpu_cores=16,
      tpu_zone=Zone.US_EAST1_C.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5_LITE.value,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  jax_resnet_v5p_8 = flax_config.get_flax_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST5_A.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  jax_resnet_v5p_32 = flax_config.get_flax_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=32,
      tpu_zone=Zone.US_EAST5_A.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
      time_out_in_min=60,
  ).run()

  # WMT
  jax_wmt_v4_8 = flax_config.get_flax_wmt_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      num_train_steps=10,
  ).run()

  # Test dependencies
  jax_resnet_v2_8 >> jax_resnet_v2_32
  jax_resnet_v3_8 >> jax_resnet_v3_32
  jax_resnet_v4_8 >> jax_resnet_v4_32
  jax_resnet_v5e_4 >> jax_resnet_v5e_16
  jax_resnet_v5p_8 >> jax_resnet_v5p_32
  jax_vit_v4_8 >> jax_vit_conv_v4_32
  jax_vit_v5e_4
  jax_gpt2_v4_8 >> jax_gpt2_v4_32
  jax_gpt2_v5e_4
  jax_sd_v4_8 >> jax_sd_v4_32
  jax_sd_v5p_8 >> jax_sd_v5p_32
  jax_bart_v4_8 >> jax_bart_conv_v4_32
  jax_bert_mnli_v4_8 >> jax_bert_mnli_conv_v4_32
  jax_bert_mrpc_v4_8 >> jax_bert_mrpc_conv_v4_32
  jax_wmt_v4_8
