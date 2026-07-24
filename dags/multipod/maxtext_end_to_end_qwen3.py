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

"""A DAG to run end-to-end MaxText tests."""


import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common.quarantined_tests import QuarantineTests
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 7 * * *" if composer_env.is_prod_env() else None
HF_TOKEN = models.Variable.get("HF_TOKEN", None)


with models.DAG(
    dag_id="maxtext_end_to_end_qwen3",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "nightly",
        "mlscale_devx",
        "TPU",
        "v5p-8",
    ],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"

  test_models_tpu = {
      "qwen3-30b": {
          "owner": test_owner.HENGTAO_G,
          "time_out_in_min": 300,
          "stable_cluster": XpkClusters.TPU_V5P_128_CLUSTER,
          "nightly_cluster": XpkClusters.TPU_V5P_128_CLUSTER,
          "commands": [
              "git fetch origin && git checkout origin/yixuan-dev-dag",
              "export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)",
              "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3_to_mt.sh $RUN_ID",
              "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3.sh $RUN_ID",
              "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3_sft.sh $RUN_ID",
              "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3_rl.sh $RUN_ID",
              "bash tests/end_to_end/tpu/qwen3/30b/test_qwen3_to_hf.sh $RUN_ID",
          ],
      },
  }

  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  for model, test_config in test_models_tpu.items():
    model_cmds = (f"export HF_TOKEN={HF_TOKEN}",) + tuple(
        test_config["commands"]
    )
    stable_tpu = gke_config.get_gke_config(
        time_out_in_min=test_config.get("time_out_in_min", 60),
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=model_cmds,
        docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
        cluster=test_config.get("stable_cluster", XpkClusters.TPU_V5P_8_CLUSTER_V2),
        test_owner=test_config["owner"],
    ).run_with_quarantine(quarantine_task_group)
    nightly_tpu = gke_config.get_gke_config(
        time_out_in_min=test_config.get("time_out_in_min", 60),
        test_name=f"{test_name_prefix}-nightly-{model}",
        run_model_cmds=model_cmds,
        docker_image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
        cluster=test_config.get("nightly_cluster", XpkClusters.TPU_V5P_8_CLUSTER),
        test_owner=test_config["owner"],
    ).run_with_quarantine(quarantine_task_group)
    stable_tpu >> nightly_tpu

