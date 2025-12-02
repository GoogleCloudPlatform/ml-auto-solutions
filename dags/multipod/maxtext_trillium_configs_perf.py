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
A DAG to run perf tests for MaxText model configs on v6e.
"""
import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, Project, XpkClusters, DockerImage
from dags.common.model_configs import MaxTextTrilliumModelConfigs
from dags.multipod.configs import maxtext_sweep_gke_config
from dags.multipod.configs.common import SetupMode
from xlml.apis import metric_config, mlcompass

# Run once a day at 3 am UTC (7 pm PST / 8 pm PDT)
CONIFGS_SCHEDULED_TIME = "30 10 * * *" if composer_env.is_prod_env() else None
DOCKER_IMAGES = [
    (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK),
    (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
    (SetupMode.TPU_RECIPES, DockerImage.MAXTEXT_JAX_052_RECIPES_012),
]
BASE_OUTPUT_DIRECTORY = "gs://runner-maxtext-logs"

# Use stable-stack-candidate instead of stable-stack
# int8 needs aqtp>=0.8.3, flash attention needs jax>=0.5.3
need_stable_candidate_set = {
    MaxTextTrilliumModelConfigs.MIXTRAL_8X7B_DROPPED_INT8,
    MaxTextTrilliumModelConfigs.DEEPSEEK_V3_EP16,
}
moe_set = {
    MaxTextTrilliumModelConfigs.MIXTRAL_8X7B_DROPLESS,
    MaxTextTrilliumModelConfigs.MIXTRAL_8X7B_DROPPED,
    MaxTextTrilliumModelConfigs.MIXTRAL_8X7B_DROPPED_INT8,
    MaxTextTrilliumModelConfigs.DEEPSEEK_V3_EP16,
}

with models.DAG(
    dag_id="maxtext_trillium_configs_perf",
    schedule=CONIFGS_SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "nightly",
        "mlscale_perfx",
        "TPU",
        "v6e-256",
    ],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
) as dag:
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )
  all_tests = []
  for mode, image in DOCKER_IMAGES:
    for model in MaxTextTrilliumModelConfigs:
      # No tpu-recipe for DeepSeek v3
      if (
          model == MaxTextTrilliumModelConfigs.DEEPSEEK_V3_EP16
          and image == DockerImage.MAXTEXT_JAX_052_RECIPES_012
      ):
        continue
      if (
          model in need_stable_candidate_set
          and image == DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK
      ):
        image = DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK_CANDIDATE

      base_run_model_cmds = [
          f"python3 -m benchmarks.benchmark_runner on-device --base_output_directory={BASE_OUTPUT_DIRECTORY} --model_name={model.value} --libtpu_type=maxtext-docker --num_steps=15",
      ]
      num_slices = (
          [2]
          if model == MaxTextTrilliumModelConfigs.LLAMA3_1_405B_8192
          or model == MaxTextTrilliumModelConfigs.DEEPSEEK_V3_EP16
          else [1, 2]
      )
      # Enable profile config to extract metrics for MoE
      enable_profile_config = True if model in moe_set else False

      maxtext_sweep_gke_test = (
          maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
              test_owner=test_owner.AIRFLOW,
              dataset_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              composer_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              dataset_name=metric_config.DatasetOption.XLML_DATASET,
              cluster=XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
              time_out_in_min=360,
              base_output_directory=BASE_OUTPUT_DIRECTORY,
              num_slices=num_slices,
              docker_image=image.value,
              run_name_prefix=f"maxtext-{model.name.lower()}-{mode.value}",
              base_run_model_cmds=base_run_model_cmds,
              sweep_params={},
              enable_profile_config=enable_profile_config,
          )
      )
      all_tests += maxtext_sweep_gke_test

  # Add dependencies between the tests so they are not all launched at once
  mlcompass_scheduler = mlcompass.Scheduler()
  chain_num = 4
  prev = all_tests[0].run_with_name_gen_and_quarantine(quarantine_task_group)
  mlcompass_scheduler.register(prev)
  for i in range(1, len(all_tests)):
    curr = all_tests[i].run_with_name_gen_and_quarantine(quarantine_task_group)
    mlcompass_scheduler.register(curr)
    if i % chain_num != 0:
      prev >> curr
    prev = curr
