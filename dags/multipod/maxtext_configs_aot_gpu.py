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

"""
A DAG to run AOT compilation tests for MaxText model configs.
"""
import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import GpuVersion, TpuVersion, Zone, DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 5 am UTC (9 pm PST / 10 pm PDT)
SCHEDULED_TIME = "45 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_configs_aot_gpu",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "nightly",
        "mlscale_devx",
        "GPU",
        "h100-mega-80gb-8",
    ],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
    concurrency=2,
) as dag:
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  # GPU AoT tests
  cmd = "bash src/MaxText/configs/a3/llama_2_7b/8vm.sh EXECUTABLE=train_compile M_COMPILE_TOPOLOGY=a3 M_COMPILE_TOPOLOGY_NUM_SLICES=8"
  stable_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
      time_out_in_min=300,
      test_name="maxtext-aot-a3-stable",
      run_model_cmds=(cmd,),
      num_slices=1,
      cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
      docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE_STACK.value,
      test_owner=test_owner.NUOJIN_C,
  ).run_with_quarantine(quarantine_task_group)
