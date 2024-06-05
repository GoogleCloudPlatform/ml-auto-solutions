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
from dags.vm_resource import TpuVersion, Zone, RuntimeVersion
from dags.imagegen_devx.configs import project_bite_config as config


# Run once a day at 6 pm UTC (11 am PST)
SCHEDULED_TIME = "0 18 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="project_bite",
    schedule=SCHEDULED_TIME,
    tags=["imagegen_devx", "jax", "nightly", "bite"],
    start_date=datetime.datetime(2024, 4, 4),
    catchup=False,
) as dag:
  # AXLearn head against JAX head
  jax_fuji_v4_8 = config.get_bite_tpu_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      runtime_version=RuntimeVersion.TPU_UBUNTU2204_BASE.value,
      model_config="fuji-test-v1",
      time_out_in_min=180,
  )

  # AXLearn pinned version against JAX head
  # pinned_version commit: 2a44f58fe3d3f33eaa9d10bfa8a2f8ce9bec029e
  # pinned_version PR: https://github.com/apple/axlearn/pull/505
  # pinned_version date: Jun 3, 2024
  jax_pinned_fuji_v4_8 = config.get_bite_tpu_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      runtime_version=RuntimeVersion.TPU_UBUNTU2204_BASE.value,
      model_config="fuji-test-v1",
      pinned_version="2a44f58fe3d3f33eaa9d10bfa8a2f8ce9bec029e",
      time_out_in_min=180,
  )
