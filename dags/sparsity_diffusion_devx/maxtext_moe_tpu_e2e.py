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

"""A DAG to run end-to-end MoE tests."""


import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env, test_owner
from dags.quarantined_tests import QuarantineTests
from dags.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config
from xlml.utils import name_format

# Run once a day at 5 am UTC (9 pm PST)
SCHEDULED_TIME = "0 5 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_moe_tpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "sparsity_diffusion_devx",
        "multipod_team",
        "maxtext",
        "tpu",
        "stable",
        "nightly",
        "mlscale_onduty",
    ],
    start_date=datetime.datetime(2024, 11, 14),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  multicluster_test_models = {
      "mixtral-8x7b": [
          {
              "script_name": "tpu/mixtral/8x7b/1_test_mixtral",
              "cluster": XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER,
              "time_out_in_min": 240,
          },
          {
              "script_name": "tpu/mixtral/8x7b/2_test_mixtral",
              "cluster": XpkClusters.TPU_V4_128_CLUSTER,
              "time_out_in_min": 90,
          },
      ],
      "mixtral-8x22b": [
          {
              "script_name": "tpu/mixtral/8x22b/1_test_mixtral",
              "cluster": XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER,
              "time_out_in_min": 360,
          },
          {
              "script_name": "tpu/mixtral/8x22b/2_test_mixtral",
              "cluster": XpkClusters.TPU_V4_128_CLUSTER,
              "time_out_in_min": 60,
          },
      ],
  }

  def convert_checkpoint_and_run_training(
      test_group_id,
      test_name_prefix,
      type,
      docker_image,
      model,
      test_scripts_details,
  ):
    with TaskGroup(group_id=test_group_id, prefix_group_id=False) as group:
      test_name = f"{test_name_prefix}-{type}-{model}"
      shared_gcs_location = name_format.generate_gcs_folder_location.override(
          task_id=f"{test_group_id}_generate_gcs_folder_location"
      )(
          gcs_subfolder,
          test_group_id,
      )
      conversion_cpu = gke_config.get_maxtext_cpu_end_to_end_gke_config(
          time_out_in_min=test_scripts_details[0]["time_out_in_min"],
          test_name=test_name,
          run_model_cmds=(
              f"export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash end_to_end/{test_scripts_details[0]['script_name']}.sh",
          ),
          docker_image=docker_image,
          test_owner=test_owner.RAN_R,
          cluster=test_scripts_details[0]["cluster"],
      ).run(gcs_location=shared_gcs_location)
      training_tpu = gke_config.get_gke_config(
          time_out_in_min=test_scripts_details[1]["time_out_in_min"],
          test_name=test_name,
          run_model_cmds=(
              f"export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash end_to_end/{test_scripts_details[1]['script_name']}.sh",
          ),
          docker_image=docker_image,
          test_owner=test_owner.RAN_R,
          cluster=test_scripts_details[1]["cluster"],
      ).run(gcs_location=shared_gcs_location)
      return conversion_cpu, training_tpu

  docker_image = {
      "stable": DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK.value,
      "nightly": DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
  }
  tests = []
  for model, test_scripts_details in multicluster_test_models.items():
    gcs_subfolder = f"{test_owner.Team.SPARSITY_DIFFUSION_DEVX.value}/maxtext"
    for type in docker_image.keys():
      test_group_id = "chained_tests" + "_" + model + "_" + type
      if QuarantineTests.is_quarantined(test_group_id):
        with quarantine_task_group:
          mode_cpu, mode_tpu = convert_checkpoint_and_run_training(
              test_group_id,
              test_name_prefix,
              type,
              docker_image[type],
              model,
              test_scripts_details,
          )
      else:
        mode_cpu, mode_tpu = convert_checkpoint_and_run_training(
            test_group_id,
            test_name_prefix,
            type,
            docker_image[type],
            model,
            test_scripts_details,
        )
      tests.append(mode_cpu)
      tests.append(mode_tpu)

    # stable_cpu >> stable_tpu >> nightly_cpu >> nightly_tpu
    for i in range(len(tests) - 1):
      tests[i] >> tests[i + 1]
