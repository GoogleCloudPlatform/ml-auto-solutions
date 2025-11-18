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
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "30 4 * * *" if composer_env.is_prod_env() else None
HF_TOKEN = models.Variable.get("HF_TOKEN", None)


with models.DAG(
    dag_id="maxtext_end_to_end",
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
      "llama2-7b": {
          "owner": test_owner.MOHIT_K,
          "commands": ["bash end_to_end/tpu/llama2/7b/test_llama2_7b.sh"],
      },
      "mistral-7b": {
          "owner": test_owner.MOHIT_K,
          "commands": ["bash end_to_end/tpu/mistral/7b/test_mistral-7b.sh"],
      },
      "gemma-2b": {
          "owner": test_owner.MOHIT_K,
          "commands": ["bash end_to_end/tpu/gemma/2b/test_gemma.sh"],
      },
      "gemma2-2b": {
          "owner": test_owner.HENGTAO_G,
          "commands": [
              "bash end_to_end/tpu/gemma2/2b/test_gemma2_to_mt.sh",
              "bash end_to_end/tpu/gemma2/2b/test_gemma2_to_hf.sh",
          ],
      },
      "gemma3-4b": {
          "owner": test_owner.HENGTAO_G,
          "commands": [
              "bash end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh",
              "bash end_to_end/tpu/gemma3/4b/test_gemma3_to_hf.sh",
          ],
      },
      "qwen3-4b": {
          "owner": test_owner.HENGTAO_G,
          "commands": [
              "bash end_to_end/tpu/qwen3/4b/test_qwen3_to_mt.sh",
              "bash end_to_end/tpu/qwen3/4b/test_qwen3_to_hf.sh",
          ],
      },
      "gpt3": {
          "owner": test_owner.MOHIT_K,
          "commands": ["bash end_to_end/tpu/test_gpt3.sh"],
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
        time_out_in_min=60,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=model_cmds,
        docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK_CANDIDATE.value,
        cluster=XpkClusters.TPU_V5P_8_CLUSTER,
        test_owner=test_config["owner"],
    ).run_with_quarantine(quarantine_task_group)
    nightly_tpu = gke_config.get_gke_config(
        time_out_in_min=60,
        test_name=f"{test_name_prefix}-nightly-{model}",
        run_model_cmds=model_cmds,
        docker_image=DockerImage.MAXTEXT_TPU_STABLE_STACK_NIGHTLY_JAX.value,
        cluster=XpkClusters.TPU_V5P_8_CLUSTER,
        test_owner=test_config["owner"],
    ).run_with_quarantine(quarantine_task_group)
    stable_tpu >> nightly_tpu

  multicluster_test_models = {
      "gemma-7b": [
          {
              "script_name": "tpu/gemma/7b/1_test_gemma",
              "cluster": XpkClusters.CPU_N2_STANDARD_64_CLUSTER,
              "time_out_in_min": 60,
          },
          {
              "script_name": "tpu/gemma/7b/2_test_gemma",
              "cluster": XpkClusters.TPU_V5P_8_CLUSTER,
              "time_out_in_min": 60,
          },
      ],
      "llama2-70b": [
          {
              "script_name": "tpu/llama2/70b/1_test_llama2_70b",
              "cluster": XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER,
              "time_out_in_min": 360,
          },
          {
              "script_name": "tpu/llama2/70b/2_test_llama2_70b",
              "cluster": XpkClusters.TPU_V5P_128_CLUSTER,
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
    with TaskGroup(group_id=test_group_id, prefix_group_id=True) as group:
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
          test_owner=test_owner.ANISHA_M,
          cluster=test_scripts_details[0]["cluster"],
      ).run(gcs_location=shared_gcs_location)
      training_tpu = gke_config.get_gke_config(
          time_out_in_min=test_scripts_details[1]["time_out_in_min"],
          test_name=test_name,
          run_model_cmds=(
              f"export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash end_to_end/{test_scripts_details[1]['script_name']}.sh",
          ),
          docker_image=docker_image,
          test_owner=test_owner.ANISHA_M,
          cluster=test_scripts_details[1]["cluster"],
      ).run(gcs_location=shared_gcs_location)
      return conversion_cpu, training_tpu

  docker_image = {
      "stable": DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK_CANDIDATE.value,
      "nightly": DockerImage.MAXTEXT_TPU_STABLE_STACK_NIGHTLY_JAX.value,
  }
  tests = []
  for model, test_scripts_details in multicluster_test_models.items():
    gcs_subfolder = f"{test_owner.Team.MULTIPOD.value}/maxtext"
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
