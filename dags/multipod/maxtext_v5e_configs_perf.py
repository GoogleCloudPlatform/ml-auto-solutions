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
A DAG to run perf tests for MaxText model configs on v5e.
"""
import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, Project, XpkClusters, DockerImage
from dags.common.model_configs import MaxTextV5eModelConfigs
from dags.multipod.configs import maxtext_sweep_gke_config
from dags.multipod.configs.common import SetupMode
from xlml.apis import metric_config

# Run once a day at 4 am UTC (8 pm PST / 9 pm PDT)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None
DOCKER_IMAGES = [
    (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK),
    (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
]
QUANTIZATION_SWEEP = {"M_QUANTIZATION": ["", "int8"]}
BASE_OUTPUT_DIRECTORY = "gs://runner-maxtext-logs"

with models.DAG(
    dag_id="maxtext_v5e_configs_perf",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "nightly",
        "mlscale_perfx",
    ],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
) as dag:
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )
  for mode, image in DOCKER_IMAGES:
    for model in MaxTextV5eModelConfigs:
      base_run_model_cmds = [
          "bash preflight.sh",
          f"python3 benchmarks/benchmark_runner.py on-device --base_output_directory={BASE_OUTPUT_DIRECTORY} --model_name={model.value} --libtpu_type=maxtext-docker --num_steps=15",
      ]
      maxtext_sweep_gke_test = (
          maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
              test_owner=test_owner.RAYMOND_Z,
              dataset_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              composer_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              dataset_name=metric_config.DatasetOption.XLML_DATASET,
              cluster=XpkClusters.TPU_V5E_256_CLUSTER,
              time_out_in_min=360,
              base_output_directory=BASE_OUTPUT_DIRECTORY,
              num_slices=[1, 2],
              docker_image=image.value,
              run_name_prefix=f"maxtext-{model.name.lower()}-{mode.value}",
              base_run_model_cmds=base_run_model_cmds,
              sweep_params=QUANTIZATION_SWEEP,
          )
      )

      chain_num = 4
      prev = maxtext_sweep_gke_test[0].run_with_name_gen_and_quarantine(
          quarantine_task_group
      )
      for i in range(1, len(maxtext_sweep_gke_test)):
        curr = maxtext_sweep_gke_test[i].run_with_name_gen_and_quarantine(
            quarantine_task_group
        )
        if i % chain_num != 0:
          prev >> curr
        prev = curr


# Run once a day at 10 am UTC (2 am PST / 3 am PDT)
PATHWAYS_SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="pathways_maxtext_v5e_configs_perf",
    schedule=None,
    tags=["multipod_team", "maxtext", "stable", "nightly", "mlscale_perfx"],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
) as dag:
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )
  for mode, image in DOCKER_IMAGES:
    for model in MaxTextV5eModelConfigs:
      base_run_model_cmds = [
          f"python3 benchmarks/benchmark_runner.py on-device --base_output_directory={BASE_OUTPUT_DIRECTORY} --model_name={model.value} --libtpu_type=maxtext-docker --num_steps=15 --use_pathways=True",
      ]
      maxtext_sweep_gke_test = (
          maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
              test_owner=test_owner.RAYMOND_Z,
              dataset_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              composer_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              dataset_name=metric_config.DatasetOption.XLML_DATASET,
              cluster=XpkClusters.TPU_V5E_256_CLUSTER,
              time_out_in_min=360,
              base_output_directory=BASE_OUTPUT_DIRECTORY,
              num_slices=[1, 2],
              docker_image=image.value,
              run_name_prefix=f"p-maxtext-{model.name.lower()}-{mode.value}",
              base_run_model_cmds=base_run_model_cmds,
              sweep_params=QUANTIZATION_SWEEP,
          )
      )

      chain_num = 4
      prev = maxtext_sweep_gke_test[0].run_with_name_gen_and_quarantine(
          quarantine_task_group, use_pathways=True
      )
      for i in range(1, len(maxtext_sweep_gke_test)):
        curr = maxtext_sweep_gke_test[i].run_with_name_gen_and_quarantine(
            quarantine_task_group, use_pathways=True
        )
        if i % chain_num != 0:
          prev >> curr
        prev = curr
