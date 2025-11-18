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

"""A DAG to run JAX tests."""

import datetime
from airflow import models
from dags import composer_env
from dags.common.vm_resource import DockerImage, TpuVersion, Zone, Project, V5_NETWORKS, V5P_SUBNETWORKS, RuntimeVersion, XpkClusters
from dags.multipod.configs import jax_tests_gce_config, jax_tests_gke_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 14 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="jax_functional_tests",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "jax", "mlscale_devx", "TPU", "v4-8", "v5p-8"],
    start_date=datetime.datetime(2024, 10, 23),
    catchup=False,
) as dag:
  default_test_name = "jax-distributed-initialize"
  v5p_project_name = Project.TPU_PROD_ENV_AUTOMATED.value
  v5p_network = V5_NETWORKS
  v5p_subnetwork = V5P_SUBNETWORKS
  v5p_runtime_version = RuntimeVersion.V2_ALPHA_TPUV5.value
  test_modes_with_docker_images = [
      (SetupMode.STABLE, None),
      (SetupMode.JAX_STABLE_STACK, DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]

  v4_task_arr, v5p_task_arr = [], []

  for test_mode, gke_docker_image in test_modes_with_docker_images:
    for num_slices in (1, 2):
      # v4 GCE
      jax_gce_v4_8 = jax_tests_gce_config.get_jax_distributed_initialize_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          num_slices=num_slices,
          test_name=f"{default_test_name}-gce-{test_mode.value}",
          test_mode=test_mode,
      )
      if len(v4_task_arr) > 1:
        # pylint: disable-next=pointless-statement
        v4_task_arr[-1] >> jax_gce_v4_8
      v4_task_arr.append(jax_gce_v4_8)

      # v4 GKE
      if gke_docker_image is not None:
        jax_gke_v4_8 = (
            jax_tests_gke_config.get_jax_distributed_initialize_config(
                cluster=XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
                time_out_in_min=60,
                num_slices=num_slices,
                test_name=f"{default_test_name}-gke-{test_mode.value}",
                docker_image=gke_docker_image.value,
            ).run()
        )
        # pylint: disable-next=pointless-statement
        v4_task_arr[-1] >> jax_gke_v4_8
        v4_task_arr.append(jax_gke_v4_8)

      # v5p GCE
      jax_gce_v5p_8 = (
          jax_tests_gce_config.get_jax_distributed_initialize_config(
              tpu_version=TpuVersion.V5P,
              tpu_cores=8,
              num_slices=num_slices,
              tpu_zone=Zone.US_EAST5_A.value,
              runtime_version=v5p_runtime_version,
              project_name=v5p_project_name,
              time_out_in_min=60,
              is_tpu_reserved=True,
              test_name=f"{default_test_name}-gce-{test_mode.value}",
              test_mode=test_mode,
              network=v5p_network,
              subnetwork=v5p_subnetwork,
          )
      )
      if len(v5p_task_arr) > 1:
        # pylint: disable-next=pointless-statement
        v5p_task_arr[-1] >> jax_gce_v5p_8
      v5p_task_arr.append(jax_gce_v5p_8)

      # v5p GKE
      if gke_docker_image is not None:
        jax_gke_v5p_8 = (
            jax_tests_gke_config.get_jax_distributed_initialize_config(
                cluster=XpkClusters.TPU_V5P_8_CLUSTER,
                time_out_in_min=60,
                num_slices=num_slices,
                test_name=f"{default_test_name}-gke-{test_mode.value}",
                docker_image=gke_docker_image.value,
            ).run()
        )
        # pylint: disable-next=pointless-statement
        v5p_task_arr[-1] >> jax_gke_v5p_8
        v5p_task_arr.append(jax_gke_v5p_8)
