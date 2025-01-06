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

"""A DAG to run all supported ML models with the nightly TensorFlow version."""

import time
import datetime
from airflow import models
from dags import composer_env
from dags.common.vm_resource import TpuVersion, Project, Zone, RuntimeVersion, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS
from dags.solutions_team.configs.tensorflow import solutionsteam_tf_release_supported_config as tf_config
from dags.solutions_team.configs.tensorflow import common


# Release tests only need to run once, they can be run manually as needed
SCHEDULED_TIME = None
VERSION = f"{tf_config.MAJOR_VERSION}.{tf_config.MINOR_VERSION}"


with models.DAG(
    dag_id=f"tf_dlrm_{tf_config.MAJOR_VERSION}_{tf_config.MINOR_VERSION}",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "tf", "se", VERSION, "supported", "xlml"],
    start_date=datetime.datetime(2024, 1, 4),
    catchup=False,
) as dag:
  embedding_dim = 32
  tf_dlrm_v4_8 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )

  embedding_dim = 128
  tf_dlrm_v4_64 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=64,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )
  embedding_dim = 128
  tf_dlrm_v4_128 = tf_config.get_tf_dlrm_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=128,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      time_out_in_min=60,
      bottom_mlp=[512, 256, embedding_dim],
      embedding_dim=embedding_dim,
      train_steps=10000,
      extraFlags="--mode=train",
      is_pod=True,
      is_pjrt=False,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
  )
  embedding_dim = 32
  tf_dlrm_v5p_8 = tf_config.get_tf_dlrm_config(
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
  )
  embedding_dim = 64
  tf_dlrm_v5p_64 = tf_config.get_tf_dlrm_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=64,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
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

  embedding_dim = 128
  tf_dlrm_v5p_128 = tf_config.get_tf_dlrm_config(
      project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
      tpu_version=TpuVersion.V5P,
      tpu_cores=128,
      tpu_zone=Zone.US_EAST5_A.value,
      time_out_in_min=60,
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
  # tf_dlrm_v4_8 >> tf_dlrm_v4_64 >> tf_dlrm_v4_128
  # tf_dlrm_v5p_64 >> tf_dlrm_v5p_128
