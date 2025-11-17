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
from dags.common.vm_resource import TpuVersion, Zone, RuntimeVersion, Project
from dags.sparsity_diffusion_devx.configs import project_bite_config as config
from airflow.utils.task_group import TaskGroup


# Run once a day at 6 pm UTC (11 am PST)
SCHEDULED_TIME = '0 18 * * *' if composer_env.is_prod_env() else None

trillium_conf = {
    'tpu_version': TpuVersion.TRILLIUM,
    'tpu_cores': 4,
    'tpu_zone': Zone.EUROPE_WEST4_A.value,
    'runtime_version': RuntimeVersion.V2_ALPHA_TPUV6.value,
}

v5p_conf = {
    'tpu_version': TpuVersion.V5P,
    'tpu_cores': 8,
    'runtime_version': RuntimeVersion.V2_ALPHA_TPUV5.value,
    'tpu_zone': Zone.US_CENTRAL1_A.value,
    'is_tpu_reserved': False,
    'project_name': Project.TPU_PROD_ENV_MULTIPOD.value,
    'network': 'mas-test',
    'subnetwork': 'mas-test',
}

v5e_conf = {
    'tpu_version': TpuVersion.V5E,
    'tpu_cores': 8,
    'tpu_zone': Zone.US_EAST1_C.value,
    'is_tpu_reserved': True,
    'runtime_version': RuntimeVersion.V2_ALPHA_TPUV5_LITE.value,
    'project_name': Project.TPU_PROD_ENV_AUTOMATED.value,
    'network': 'mas-test',
    'subnetwork': 'mas-test',
}

common = {
    'time_out_in_min': 180,
    'task_owner': test_owner.Andrew_S,
}


with models.DAG(
    dag_id='project_bite_tpu_e2e',
    schedule=SCHEDULED_TIME,
    tags=[
        'multipod_team',
        'tpu',
        'axlearn',
        'bite',
        'v6e-8',
        'v5p-8',
        'v5e-8',
    ],
    start_date=datetime.datetime(2024, 4, 4),
    catchup=False,
) as dag:
  with TaskGroup(
      group_id='bite_tpu_training', prefix_group_id=False
  ) as bite_training:
    # AXLearn head against JAX v0.5.3
    # Runs Fuji training on v5p-8 in the provided GCP Project
    jax_053_fuji_v5p_8 = config.get_bite_tpu_config(
        tpu_version=TpuVersion.V5P,
        tpu_cores=8,
        tpu_zone=Zone.US_EAST5_A.value,
        runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
        project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
        network='mas-test',
        subnetwork='mas-test',
        is_tpu_reserved=True,
        jax_version='0.5.3',
        model_config='fuji-test-v1',
        time_out_in_min=180,
        task_owner=test_owner.Maggie_Z,
    )

    # AXLearn head against JAX v0.5.3
    # Runs Fuji training on v6e-8
    jax_053_fuji_v6e_8 = config.get_bite_tpu_config(
        tpu_version=TpuVersion.TRILLIUM,
        tpu_cores=8,
        tpu_zone=Zone.US_EAST5_B.value,
        runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
        jax_version='0.5.3',
        model_config='fuji-test-v1',
        time_out_in_min=180,
        task_owner=test_owner.Maggie_Z,
    )

    # AXLearn head against JAX head
    # Runs Fuji training on v5p-8 in the provided GCP Project
    jax_main_fuji_v5p_8 = config.get_bite_tpu_config(
        tpu_version=TpuVersion.V5P,
        tpu_cores=8,
        tpu_zone=Zone.US_EAST5_A.value,
        runtime_version=RuntimeVersion.V2_ALPHA_TPUV5.value,
        project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
        network='mas-test',
        subnetwork='mas-test',
        is_tpu_reserved=True,
        model_config='fuji-test-v1',
        time_out_in_min=180,
        task_owner=test_owner.Maggie_Z,
    )

    # AXLearn head against JAX head
    # Runs Fuji training on v6e-8
    jax_main_fuji_v6e_8 = config.get_bite_tpu_config(
        tpu_version=TpuVersion.TRILLIUM,
        tpu_cores=8,
        tpu_zone=Zone.US_EAST5_B.value,
        runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
        model_config='fuji-test-v1',
        time_out_in_min=180,
        task_owner=test_owner.Maggie_Z,
    )

    # AXLearn pinned version against JAX head
    # pinned_version commit: 35a189c15bdd06416be743cecde272693363ce3c
    # pinned_version PR: https://github.com/apple/axlearn/pull/1033
    # pinned_version date: March 6, 2025 (test succeeds at this commit)
    jax_pinned_053_fuji_v6e_8 = config.get_bite_tpu_config(
        tpu_version=TpuVersion.TRILLIUM,
        tpu_cores=8,
        tpu_zone=Zone.US_EAST5_B.value,
        runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
        jax_version='0.5.3',
        model_config='fuji-test-v1',
        pinned_version='35a189c15bdd06416be743cecde272693363ce3c',
        time_out_in_min=180,
        task_owner=test_owner.Maggie_Z,
    )

  with TaskGroup(
      group_id='bite_tpu_unittests', prefix_group_id=False
  ) as bite_unittests:
    # Trillium (v6e) with JAX 0.5.3
    config.get_bite_tpu_unittests_config(
        **trillium_conf,
        jax_version='0.5.3',
        **common,
    )
    # Trillium (v6e) with JAX 0.4.38
    config.get_bite_tpu_unittests_config(
        **trillium_conf,
        jax_version='0.4.38',
        **common,
    )
    # Trillium (v6e) with JAX nightly
    config.get_bite_tpu_unittests_config(
        **trillium_conf,
        **common,
    )
    # V5P with JAX 0.5.3
    config.get_bite_tpu_unittests_config(
        **v5p_conf,
        jax_version='0.5.3',
        **common,
    )
    # V5P with JAX 0.4.38
    config.get_bite_tpu_unittests_config(
        **v5p_conf,
        jax_version='0.4.38',
        **common,
    )
    # V5P with JAX nightly
    config.get_bite_tpu_unittests_config(
        **v5p_conf,
        **common,
    )
    # V5E with JAX nightly
    config.get_bite_tpu_unittests_config(
        **v5e_conf,
        **common,
    )
