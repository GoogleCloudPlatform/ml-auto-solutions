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

"""A small-scale DAG to run a single MaxText end-to-end test for quick validation."""


import datetime
from airflow import models
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config


with models.DAG(
    dag_id="maxtext_end_to_end",
    schedule=None,  # Set to None for manual triggering only
    tags=[
        "multipod_team",
        "maxtext",
        "small_scale_test",
        "TPU",
        "v5p-8",
    ],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"

  HF_TOKEN = models.Variable.get("HF_TOKEN", None)

  # We only keep ONE small test suite (Gemma3-4b on TPU v5p-8)
  test_models_tpu = {
      "gemma3-4b": {
          "owner": test_owner.HENGTAO_G,
          "commands": [
              "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh {{ ts_nodash }}",
              "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3.sh {{ ts_nodash }}",
          ],
      },
  }

  # Run ONLY the stable TPU test (no nightly, no multicluster CPU conversions)
  for model, test_config in test_models_tpu.items():
    model_cmds = (f"export HF_TOKEN={HF_TOKEN}",) + tuple(
        test_config["commands"]
    )
    
    stable_tpu = gke_config.get_gke_config(
        time_out_in_min=60,  # Increased timeout to match production and allow compilation
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=model_cmds,
        docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
        cluster=XpkClusters.TPU_V5P_8_CLUSTER_V2,
        test_owner=test_config["owner"],
    ).run()
