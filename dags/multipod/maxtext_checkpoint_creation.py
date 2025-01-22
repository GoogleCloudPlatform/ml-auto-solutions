# Copyright 2025 Google LLC
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

"""A DAG to run all MaxText checkpoint creation"""


import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common.quarantined_tests import QuarantineTests
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config
from xlml.utils import name_format

# Run once a day at 1 am UTC (5 pm PST)
SCHEDULED_TIME = "0 1 * * *" if composer_env.is_prod_env() else None
HF_TOKEN = models.Variable.get("HF_TOKEN", None)


with models.DAG(
    dag_id="maxtext_checkpoint_creation",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "mlscale_onduty"],
    start_date=datetime.datetime(2025, 1, 22),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"
  test_models_tpu = {
      "llama2-7b": {
          "script": "end_to_end/tpu/llama2/7b/1_test_llama2_7b.sh",
          "cluster": XpkClusters.CPU_N2_STANDARD_64_CLUSTER,
          "time_out_in_min": 60,
      },
      "llama2-70b": {
          "script": "end_to_end/tpu/llama2/70b/1_test_llama2_70b.sh",
          "cluster": XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER,
          "time_out_in_min": 360,
      },
      "llama3_1-8b": {
          "script": "end_to_end/tpu/llama3.1/8b/1_test_llama3.1_8b.sh",
          "cluster": XpkClusters.CPU_N2_STANDARD_64_CLUSTER,
          "time_out_in_min": 60,
      },
      "llama3_1-70b": {
          "script": "end_to_end/tpu/llama3.1/70b/1_test_llama3.1_70b.sh",
          "cluster": XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER,
          "time_out_in_min": 360,
      },
      "mistral-7b": {
          "script": "end_to_end/tpu/mistral/7b/1_test_mistral_7b.sh",
          "cluster": XpkClusters.CPU_N2_STANDARD_64_CLUSTER,
          "time_out_in_min": 60,
      },
      "gemma-2b": {
          "script": "end_to_end/tpu/gemma/2b/1_test_gemma_2b.sh",
          "cluster": XpkClusters.CPU_N2_STANDARD_64_CLUSTER,
          "time_out_in_min": 60,
      },
      "gemma-7b": {
          "script": "end_to_end/tpu/gemma/7b/1_test_gemma_7b.sh",
          "cluster": XpkClusters.CPU_N2_STANDARD_64_CLUSTER,
          "time_out_in_min": 60,
      },
  }
  for model, test_script in test_models_tpu.items():
    stable_tpu = gke_config.get_maxtext_cpu_end_to_end_gke_config(
        time_out_in_min=test_script["time_out_in_min"],
        test_name=f"{test_name_prefix}-{model}",
        run_model_cmds=(
            f"export HF_TOKEN={HF_TOKEN}",
            f"bash {test_script['script']}",
        ),
        docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK.value,
        cluster=test_script["cluster"],
        test_owner=test_owner.MOHIT_K,
    ).run()
