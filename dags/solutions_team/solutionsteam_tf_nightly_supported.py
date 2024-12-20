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

"""A DAG to run all supported ML models with the nightly TensorFlow version."""

import datetime
import time
from airflow import models
from dags import composer_env
from dags.common.vm_resource import TpuVersion, Project, Zone, RuntimeVersion, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS
from dags.solutions_team.configs.tensorflow import solutionsteam_tf_nightly_supported_config as tf_config
from dags.solutions_team.configs.tensorflow import common


# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="tf_nightly_supported",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "tf", "nightly", "supported", "xlml"],
    start_date=datetime.datetime(2023, 8, 16),
    catchup=False,
) as dag:
  # ResNet
  tf_resnet_v2_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V2,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL1_C.value,
      time_out_in_min=60,
      global_batch_size=1024,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )

  tf_resnet_v3_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V3,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST1_D.value,
      time_out_in_min=60,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )

  tf_resnet_v4_8 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )

  tf_resnet_v4_32 = tf_config.get_tf_resnet_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=32,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      is_pod=True,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )

  tf_resnet_v5e_4 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5E,
      tpu_cores=4,
      tpu_zone=Zone.US_EAST1_C.value,
      time_out_in_min=60,
      global_batch_size=2048,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
  )

  tf_resnet_v5e_16 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5E,
      tpu_cores=16,
      tpu_zone=Zone.US_EAST1_C.value,
      time_out_in_min=60,
      global_batch_size=2048,
      network=V5_NETWORKS,
      subnetwork=V5E_SUBNETWORKS,
      is_pod=True,
      runtime_version=RuntimeVersion.TPU_VM_TF_NIGHTLY_POD.value,
  )

  tf_resnet_v5p_8 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
  )

  tf_resnet_v5p_32 = tf_config.get_tf_resnet_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=32,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
      is_pod=True,
      runtime_version=RuntimeVersion.TPU_VM_TF_NIGHTLY_POD.value,
  )
  embedding_dim = 8
  tf_dlrm_v5p_8 = tf_config.get_tf_dlrm_v1_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=8,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pod=False,
      is_pjrt=True,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
      global_batch_size=4096,
  )
  embedding_dim = 128
  tf_dlrm_v5p_32 = tf_config.get_tf_dlrm_v2_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=32,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=120,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pod=True,
      is_pjrt=True,
      network=V5_NETWORKS,
      subnetwork=V5P_SUBNETWORKS,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )

  # Test dependencies
  tf_resnet_v2_8
  tf_resnet_v3_8
  tf_resnet_v4_8 >> tf_resnet_v4_32
  tf_resnet_v5e_4 >> tf_resnet_v5e_16
  tf_resnet_v5p_8 >> tf_resnet_v5p_32
  tf_dlrm_v5p_8 >> tf_dlrm_v5p_32
