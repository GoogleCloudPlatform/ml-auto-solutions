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
# Pause test on GKE
SCHEDULED_TIME = "0 1 * * *"

with models.DAG(
    dag_id="mxla_gpt_6b_nightly_gke",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "gke",
        "nightly",
        "gpt_6b",
        "mlscale_perfx",
        "TPU",
        "v4-8",
        "v5p-8",
    ],
    start_date=datetime.datetime(2024, 3, 18),
    catchup=False,
) as dag:
  jax_nightly_image = DockerImage.MAXTEXT_TPU_JAX_NIGHTLY
  default_gpt3_6b_test_name = "mxla-gpt3-6b-nightly-gke"

  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  gpt3_6b_nightly_1slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  gpt3_6b_nightly_2slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      num_slices=2,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  gpt3_6b_nightly_4slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      num_slices=4,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  gpt3_6b_nightly_8slice_v4_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      num_slices=8,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  gpt3_6b_nightly_1slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  gpt3_6b_nightly_2slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      num_slices=2,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  gpt3_6b_nightly_4slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      num_slices=4,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  gpt3_6b_nightly_8slice_v5p_8 = gke_config.get_gke_gpt3_6b_nightly_config(
      num_slices=8,
      cluster=XpkClusters.TPU_V5P_8_CLUSTER,
      time_out_in_min=60,
      test_name=default_gpt3_6b_test_name,
      docker_image=jax_nightly_image.value,
      test_owner=test_owner.TONY_C,
  ).run_with_quarantine(quarantine_task_group)

  (
      gpt3_6b_nightly_1slice_v4_8
      >> gpt3_6b_nightly_2slice_v4_8
      >> gpt3_6b_nightly_4slice_v4_8
      >> gpt3_6b_nightly_8slice_v4_8
  )

  (
      gpt3_6b_nightly_1slice_v5p_8
      >> gpt3_6b_nightly_2slice_v5p_8
      >> gpt3_6b_nightly_4slice_v5p_8
      >> gpt3_6b_nightly_8slice_v5p_8
  )
