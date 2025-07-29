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

"""A DAG to run all supported ML models with the latest JAX/FLAX version."""

import datetime
from airflow import models
from dags import composer_env
from dags.multipod.configs.common import SetupMode
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, RuntimeVersion, DockerImage
from dags.axlearn.configs import axlearn_config as config
from airflow.utils.task_group import TaskGroup
from datetime import timedelta


# Run once a day at 6 pm UTC (11 am PST)
SCHEDULED_TIME = '0 18 * * *' if composer_env.is_prod_env() else None

v5p_conf = {
    'cluster_name': 'ml-auto-solutions-orbax',
    'tpu_version': TpuVersion.V5P,
    'tpu_cores': 128,
    'tpu_zone': Zone.US_EAST5_B.value,
    'is_tpu_reserved': False,
    'project_name': 'cloud-tpu-multipod-dev',
    'network': 'ml-auto-solutions-orbax-mtu9k-1-us-east5-b',
    'subnetwork': 'ml-auto-solutions-orbax-privatesubnet-1-us-east5-b',
    'module': 'text.gpt.c4_trainer',
    'model_config': 'fuji-7B-v2-flash',
    'num_replica': 2,
    'runtime_version': RuntimeVersion.V2_ALPHA_TPUV5.value,
}

common = {
    'time_out_in_min': 180,
    'task_owner': test_owner.CAMILO_Q,
}

default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=25),
}

with models.DAG(
    dag_id='axlearn_chk_save_gcs',
    schedule=SCHEDULED_TIME,
    tags=[
        'multipod_team',
        'tpu',
        'axlearn',
    ],
    catchup=False,
    default_args=default_args,
) as dag:
  with TaskGroup(
      group_id='axl-tpu-train', prefix_group_id=False
  ) as axlearn_training:
    docker_images = [(
        SetupMode.NIGHTLY,
        DockerImage.AXLEARN_CUSTOM,
    )]

    test_configs = {
        'tpu-v5p-128': [2],
    }
    for mode, image in docker_images:
      for accelerator, slices in test_configs.items():
        for slice_num in slices:
          # AXLearn head against JAX head
          # Runs Fuji training on v5p-128 in the provided GCP Project
          jax_main_fuji_v5p_8 = config.get_axlearn_tpu_config(
              cluster_name=v5p_conf['cluster_name'],
              docker_image=image.value,
              tpu_version=v5p_conf['tpu_version'],
              tpu_cores=v5p_conf['tpu_cores'],
              tpu_zone=v5p_conf['tpu_zone'],
              runtime_version=v5p_conf['runtime_version'],
              project_name=v5p_conf['project_name'],
              network=v5p_conf['network'],
              subnetwork=v5p_conf['subnetwork'],
              is_tpu_reserved=v5p_conf['is_tpu_reserved'],
              num_replica=v5p_conf['num_replica'],
              model_config=v5p_conf['model_config'],
              time_out_in_min=common['time_out_in_min'],
              task_owner=common['task_owner'],
          ).run(
              module=v5p_conf['module'],
              model_config=v5p_conf['model_config'],
          )
