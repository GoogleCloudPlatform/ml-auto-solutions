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

"""A DAG to run end-to-end MaxText JAX Stable Stack tests."""


import datetime
from airflow import models
from dags import composer_env, test_owner, gcs_bucket
from dags.vm_resource import Project, TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, ClusterName
from dags.multipod.configs import gke_config
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_jax_ss_e2e",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "jax-ss"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_configs = {
      # accelerator: list of slices to test
      "v4-16": [1, 2],
      "v5-8": [1, 2],
  }
  tpu_versions = {
      # accelerator: list of slices to test
      "v4-16": TpuVersion.V4,
      "v5-8": TpuVersion.V5P,
  }
  cluster_names = {
      # accelerator: cluster name
      "v4-16": ClusterName.V4_16_MULTISLICE_CLUSTER,
      "v5-8": ClusterName.V5P_8_MULTISLICE_CLUSTER,
  }
  tpu_zones = {
      # accelerator: cluster name
      "v4-16": Zone.US_CENTRAL2_B,
      "v5-8": Zone.US_EAST5_A,
  }
  project_names = {
      # accelerator: cluster name
      "v4-16": Project.TPU_PROD_ENV_MULTIPOD,
      "v5-8": Project.CLOUD_TPU_MULTIPOD_DEV,
  }
  for accelerator, slices in test_configs.items():
      cores = accelerator.rsplit("-", maxsplit=1)[-1]
      for slice_num in slices:
        maxtext_jax_ss_test = gke_config.get_gke_maxtext_jax_ss_config(
            tpu_version=tpu_versions[accelerator],
            tpu_cores=cores,
            num_slices=slice_num,
            cluster_name=cluster_names[accelerator].value,
            tpu_zone=tpu_zones[accelerator].value,
            project_name=project_names[accelerator].value,
            time_out_in_min=60,
            test_name=f"maxtext-jax-ss-{accelerator}-{slice_num}x",
            docker_image=DockerImage.MAXTEXT_TPU_JAX_SS.value,
            test_owner=test_owner.PARAM_B,
        ).run()