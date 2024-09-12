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
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, ClusterName, DockerImage
from dags.multipod.configs import maxtext_sweep_gke_config
from dags.multipod.configs.common import SetupMode
from xlml.apis import metric_config

# Run once a day at 4 am UTC (8 pm PST / 9 pm PDT)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None
MODEL_CONFIGS = [
    "gpt3_175b",
]
DOCKER_IMAGES = [
    (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE),
    (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
]
QUANTIZATION_SWEEP = {"M_QUANTIZATION": ["", "int8"]}
BASE_OUTPUT_DIRECTORY = "gs://runner-maxtext-logs"

with models.DAG(
    dag_id="maxtext_trillium_configs_perf",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
) as dag:
  for mode, image in DOCKER_IMAGES:
    for model in MODEL_CONFIGS:
      base_run_model_cmds = [
          f"bash MaxText/configs/trillium/{model}.sh OUTPUT_PATH={BASE_OUTPUT_DIRECTORY} DATASET_PATH=gs://max-datasets-rogue",
      ]
      maxtext_sweep_gke_test = (
          maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
              test_owner=test_owner.RAYMOND_Z,
              project_name=Project.TPU_PROD_ENV_LARGE_ADHOC.value,
              dataset_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              composer_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
              dataset_name=metric_config.DatasetOption.XLML_DATASET,
              cluster_name=ClusterName.BODABORG_V6E_256_EUROPE_WEST4_A.value,
              tpu_zone=Zone.EUROPE_WEST4_A.value,
              time_out_in_min=360,
              base_output_directory=BASE_OUTPUT_DIRECTORY,
              tpu_version=TpuVersion.TRILLIUM,
              tpu_cores=256,
              num_slices=[1, 2],
              docker_image=image.value,
              run_name_prefix=f"maxtext-{model}-{mode.value}",
              base_run_model_cmds=base_run_model_cmds,
              sweep_params=QUANTIZATION_SWEEP,
          )
      )

      chain_num = 4
      prev = maxtext_sweep_gke_test[0].run_with_run_name_generation()
      for i in range(1, len(maxtext_sweep_gke_test)):
        curr = maxtext_sweep_gke_test[i].run_with_run_name_generation()
        if i % chain_num != 0:
          prev >> curr
        prev = curr
