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

"""A DAG to run MaxText inference benchmarks with nightly version."""

import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import maxtext_inference_gce_config
from dags.multipod.configs.common import SetupMode, Platform


# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_inference",
    schedule=SCHEDULED_TIME,
    tags=["inference_team", "maxtext", "nightly", "benchmark"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext-inference"
  test_models = {
      "llama2-7b": "llama2_7b",
      # "mistral": "test_mistral",
      # "gemma-2b": "gemma/2b/test_gemma",
      # "gpt3": "test_gpt3",
  }

  test_mode = SetupMode.NIGHTLY

  # TODO(yeandy) Setup peer network, and then change to CLOUD_TPU_INFERENCE_TEST
  # v5e_project_name = Project.CLOUD_TPU_INFERENCE_TEST.value
  # v5e_zone = Zone.US_WEST1_C.value
  v5e_project_name = Project.TPU_PROD_ENV_AUTOMATED.value
  v5e_zone = Zone.US_EAST1_C.value
  v5e_network = V5_NETWORKS
  v5e_subnetwork = V5E_SUBNETWORKS
  v5e_runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value

  for model, test_script in test_models.items():
    maxtext_nightly_1slice_v5e_8 = (
        maxtext_inference_gce_config.get_maxtext_inference_nightly_config(
            tpu_version=TpuVersion.V5E,
            tpu_cores=8,
            tpu_zone=v5e_zone,
            runtime_version=v5e_runtime_version,
            project_name=v5e_project_name,
            time_out_in_min=60,
            is_tpu_reserved=True,
            test_name=f"{test_name_prefix}-nightly-{model}",
            test_mode=test_mode,
            network=v5e_network,
            subnetwork=v5e_subnetwork,
        ).run()
    )

    maxtext_nightly_1slice_v5e_8
