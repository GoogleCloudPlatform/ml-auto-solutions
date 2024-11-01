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

"""A DAG to run JAX tests."""

import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.multipod.configs import jax_tests_gce_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="jax_functional_tests",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "jax"],
    start_date=datetime.datetime(2024, 10, 23),
    catchup=False,
) as dag:
  default_test_name = "jax"
  test_mode = SetupMode.NIGHTLY

  # v4
  jax_nightly_1slice_v4_8 = (
      jax_tests_gce_config.get_jax_distributed_initialize_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          test_name=default_test_name,
          test_mode=test_mode,
      )
  )

  jax_nightly_2slice_v4_8 = (
      jax_tests_gce_config.get_jax_distributed_initialize_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          is_tpu_reserved=False,
          num_slices=2,
          test_name=default_test_name,
          test_mode=test_mode,
      )
  )

  # v5p
  v5p_project_name = Project.TPU_PROD_ENV_AUTOMATED.value
  v5p_network = V5_NETWORKS
  v5p_subnetwork = V5P_SUBNETWORKS
  v5p_runtime_version = RuntimeVersion.V2_ALPHA_TPUV5.value

  jax_nightly_1slice_v5p_8 = (
      jax_tests_gce_config.get_jax_distributed_initialize_config(
          tpu_version=TpuVersion.V5P,
          tpu_cores=8,
          tpu_zone=Zone.US_EAST5_A.value,
          runtime_version=v5p_runtime_version,
          project_name=v5p_project_name,
          time_out_in_min=60,
          is_tpu_reserved=True,
          test_name=default_test_name,
          test_mode=test_mode,
          network=v5p_network,
          subnetwork=v5p_subnetwork,
      )
  )

  jax_nightly_2slice_v5p_8 = (
      jax_tests_gce_config.get_jax_distributed_initialize_config(
          tpu_version=TpuVersion.V5P,
          tpu_cores=8,
          num_slices=2,
          tpu_zone=Zone.US_EAST5_A.value,
          runtime_version=v5p_runtime_version,
          project_name=v5p_project_name,
          time_out_in_min=60,
          is_tpu_reserved=True,
          test_name=default_test_name,
          test_mode=test_mode,
          network=v5p_network,
          subnetwork=v5p_subnetwork,
      )
  )
