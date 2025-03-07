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
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, RuntimeVersion
from dags.sparsity_diffusion_devx.configs import project_bite_config as config


# Run once a day at 6 pm UTC (11 am PST)
SCHEDULED_TIME = "0 18 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="project_bite_tpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "sparsity_diffusion_devx",
        "multipod_team",
        "tpu",
        "axlearn",
        "bite",
    ],
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
      task_owner=test_owner.Maggie_Z,
  )

  # AXLearn pinned version against JAX head
  # pinned_version commit: e918d7c219d067dfcace8a25e619d90c5a54c36b
  # pinned_version PR: https://github.com/apple/axlearn/pull/752
  # pinned_version date: Oct 16, 2024
  jax_pinned_fuji_v4_8 = config.get_bite_tpu_config(
      tpu_version=TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=Zone.US_CENTRAL2_B.value,
      runtime_version=RuntimeVersion.TPU_UBUNTU2204_BASE.value,
      model_config="fuji-test-v1",
      pinned_version="e918d7c219d067dfcace8a25e619d90c5a54c36b",
      time_out_in_min=180,
      task_owner=test_owner.Maggie_Z,
  )

default_unittest_args = {
    "retries": 0,
}

with models.DAG(
    dag_id="project_bite_tpu_unittests",
    schedule=SCHEDULED_TIME,
    tags=[
        "sparsity_diffusion_devx",
        "tpu",
        "axlearn",
        "bite",
    ],
    start_date=datetime.datetime(2025, 2, 24),
    catchup=False,
    default_args=default_unittest_args,
) as dag:
  unittests = config.get_bite_tpu_unittests_config(
      tpu_version=TpuVersion.TRILLIUM,
      tpu_cores=4,
      tpu_zone=Zone.EUROPE_WEST4_A.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
      time_out_in_min=180,
      task_owner=test_owner.Andrew_S,
  )
