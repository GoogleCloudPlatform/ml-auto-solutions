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

"""
A DAG to run MXLA MaxText tests.
"""
import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, DockerImage, XpkClusters, Project
from dags.multipod.configs import gke_config

# Run once a day at 9 am UTC (1 am PST)
SCHEDULED_TIME = "0 9 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="mxla_maxtext_nightly_gke",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "gke",
        "nightly",
        "mlscale_perfx",
    ],
    start_date=datetime.datetime(2024, 3, 12),
    catchup=False,
) as dag:
  jax_nightly_image = DockerImage.MAXTEXT_TPU_JAX_NIGHTLY
  default_test_name = "mxla-maxtext-nightly-gke"

  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  # v5p tests
  maxtext_nightly_1slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RAYMOND_Z,
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_2slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=2,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RAYMOND_Z,
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_4slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=4,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RAYMOND_Z,
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_8slice_v5p_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=8,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RAYMOND_Z,
  ).run_with_quarantine(quarantine_task_group)

  # v6e tests
  maxtext_nightly_1slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RISHABH_B,
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_2slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=2,
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RISHABH_B,
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_4slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=4,
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RISHABH_B,
  ).run_with_quarantine(quarantine_task_group)

  maxtext_nightly_8slice_v6e_8 = gke_config.get_gke_maxtext_nightly_config(
      num_slices=8,
      cluster=XpkClusters.TPU_V6E_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.RISHABH_B,
  ).run_with_quarantine(quarantine_task_group)

  # Define dependencies for v5p tests to run sequentially
  (
      maxtext_nightly_1slice_v5p_8
      >> maxtext_nightly_2slice_v5p_8
      >> maxtext_nightly_4slice_v5p_8
      >> maxtext_nightly_8slice_v5p_8
  )

  # Define dependencies for v6e tests to run sequentially
  (
      maxtext_nightly_1slice_v6e_8
      >> maxtext_nightly_2slice_v6e_8
      >> maxtext_nightly_4slice_v6e_8
      >> maxtext_nightly_8slice_v6e_8
  )
