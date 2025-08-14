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
SCHEDULED_TIME = "0 5 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_configs_aot",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "nightly",
        "mlscale_devx",
    ],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
    concurrency=2,
) as dag:
  # Testing configurations
  tpu_configs = {
      # accelerator: [(model_size, num_cores), ...],
      "v4": [("22b", 128), ("52b", 384)],
      "v5e": [("16b", 256), ("32b", 256), ("64b", 256), ("128b", 256)],
      "v5p": [
          ("32b", 128),
          ("64b", 128),
          ("128b", 256),
          ("128b", 512),
          ("256b", 1024),
          ("512b", 1024),
          ("1024b", 2048),
          ("1024b", 4096),
      ],
  }
  num_slices = [1, 2]
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]

  run_model_cmds_dict = {}
  for tpu, models in tpu_configs.items():
    run_model_cmds = []
    for model_size, num_cores in models:
      for n in num_slices:
        cmd = f"bash MaxText/configs/{tpu}/{model_size}.sh EXECUTABLE=train_compile M_COMPILE_TOPOLOGY={tpu}-{num_cores} M_COMPILE_TOPOLOGY_NUM_SLICES={n}"
        run_model_cmds.append(cmd)
    run_model_cmds_dict[tpu] = run_model_cmds

  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  for mode, image in docker_images:
    maxtext_v4_configs_test = gke_config.get_gke_config(
        time_out_in_min=60,
        test_name=f"maxtext-aot-v4-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v4"],
        docker_image=image.value,
        test_owner=test_owner.NUOJIN_C,
    ).run_with_quarantine(quarantine_task_group)

    maxtext_v5e_configs_test = gke_config.get_gke_config(
        time_out_in_min=60,
        test_name=f"maxtext-aot-v5e-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v5e"],
        docker_image=image.value,
        test_owner=test_owner.NUOJIN_C,
    ).run_with_quarantine(quarantine_task_group)

    maxtext_v5p_configs_test = gke_config.get_gke_config(
        time_out_in_min=60,
        test_name=f"maxtext-aot-v5p-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v5p"],
        docker_image=image.value,
        test_owner=test_owner.NUOJIN_C,
    ).run_with_quarantine(quarantine_task_group)

    (
        maxtext_v4_configs_test
        >> maxtext_v5e_configs_test
        >> maxtext_v5p_configs_test
    )

  # GPU AoT tests
  cmd = f"bash MaxText/configs/a3/llama_2_7b/16vm.sh EXECUTABLE=train_compile M_COMPILE_TOPOLOGY=a3 M_COMPILE_TOPOLOGY_NUM_SLICES=16"
  stable_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
      time_out_in_min=300,
      test_name=f"maxtext-aot-a3-stable",
      run_model_cmds=(cmd,),
      num_slices=1,
      cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
      docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE_STACK.value,
      test_owner=test_owner.NUOJIN_C,
  ).run_with_quarantine(quarantine_task_group)
