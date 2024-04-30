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
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, ClusterName
from dags.multipod.configs import gke_config
from airflow.utils.task_group import TaskGroup
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_end_to_end",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"
  test_models_tpu = {
      "llama2-7b": "tpu/test_llama2_7b",
      "mistral": "tpu/test_mistral",
      "gemma-2b": "tpu/gemma/2b/test_gemma",
      "gpt3": "tpu/test_gpt3",
  }

  test_models_gpu = {
      "llama2-7b-train-1node": ("MaxText/configs/a3/llama_2_7b/1vm.sh", 1),
      "llama2-7b-train-2node": ("MaxText/configs/a3/llama_2_7b/2vm.sh", 2),
      "llama2-7b": ("end_to_end/gpu/a3/test_llama2_7b.sh", 1),
  }

  for model, test_script in test_models_tpu.items():
    stable_tpu = gke_config.get_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(f"bash end_to_end/{test_script}.sh",),
        docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
        test_owner=test_owner.JON_B,
    ).run()
    nightly_tpu = gke_config.get_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"{test_name_prefix}-nightly-{model}",
        run_model_cmds=(f"bash end_to_end/{test_script}.sh",),
        docker_image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
        test_owner=test_owner.JON_B,
    ).run()
    stable_tpu >> nightly_tpu

  for model, (test_script, nnodes) in test_models_gpu.items():
    stable_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100,
        gpu_zone=Zone.US_CENTRAL1_C.value,
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-stable-{model}",
        test_script=test_script,
        num_slices=nnodes,
        cluster_name=ClusterName.A3_CLUSTER.value,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE.value,
        test_owner=test_owner.NINA_C,
    ).run()
    nightly_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100,
        gpu_zone=Zone.US_CENTRAL1_C.value,
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-nightly-{model}",
        test_script=test_script,
        num_slices=2,
        cluster_name=ClusterName.A3_CLUSTER.value,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_NIGHTLY.value,
        test_owner=test_owner.NINA_C,
    ).run()
    stable_tpu >> nightly_tpu >> stable_gpu >> nightly_gpu

  multicluster_test_models = {
      "gemma-7b": [
          {
              "script_name": "tpu/gemma/7b/1_test_gemma",
              "cpu_device_type": CpuVersion.N2_STANDARD,
              "cpu_zone": Zone.US_CENTRAL1_B.value,
              "cluster_name": ClusterName.CPU_N2_STANDARD_64.value,
              "time_out_in_min": 60,
          },
          {
              "script_name": "tpu/gemma/7b/2_test_gemma",
              "tpu_version": TpuVersion.V4,
              "tpu_cores": 16,
              "cluster_name": ClusterName.V4_16_MULTISLICE_CLUSTER.value,
              "tpu_zone": Zone.US_CENTRAL2_B.value,
              "time_out_in_min": 60,
          },
      ],
      "mixtral-8x7b": [
          {
              "script_name": "tpu/mixtral/8x7b/1_test_mixtral",
              "cpu_device_type": CpuVersion.M1_MEGAMEM,
              "cpu_zone": Zone.US_CENTRAL1_B.value,
              "cluster_name": ClusterName.CPU_M1_MEGAMEM_96.value,
              "time_out_in_min": 180,
          },
          {
              "script_name": "tpu/mixtral/8x7b/2_test_mixtral",
              "tpu_version": TpuVersion.V4,
              "tpu_cores": 128,
              "cluster_name": ClusterName.V4_128_MULTISLICE_CLUSTER.value,
              "tpu_zone": Zone.US_CENTRAL2_B.value,
              "time_out_in_min": 60,
          },
      ],
  }

  for model, test_scripts_details in multicluster_test_models.items():
    gcs_subfolder = f"{test_owner.Team.MULTIPOD.value}/maxtext"

    test_group_id = "chained_tests" + "_" + model + "_" + "stable"

    with TaskGroup(group_id=test_group_id, prefix_group_id=False) as group:
      shared_gcs_location = name_format.generate_gcs_folder_location.override(
          task_id=f"{test_group_id}_generate_gcs_folder_location"
      )(
          gcs_subfolder,
          test_group_id,
      )
      stable_cpu = gke_config.get_maxtext_cpu_end_to_end_gke_config(
          device_type=test_scripts_details[0]["cpu_device_type"],
          cpu_zone=test_scripts_details[0]["cpu_zone"],
          time_out_in_min=test_scripts_details[0]["time_out_in_min"],
          test_name=f"{test_name_prefix}-stable-{model}",
          run_model_cmds=(
              f"export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash end_to_end/{test_scripts_details[0]['script_name']}.sh",
          ),
          cluster_name=test_scripts_details[0]["cluster_name"],
          docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
          test_owner=test_owner.ANISHA_M,
      ).run(gcs_location=shared_gcs_location)
      stable_tpu = gke_config.get_gke_config(
          tpu_version=test_scripts_details[1]["tpu_version"],
          tpu_cores=test_scripts_details[1]["tpu_cores"],
          tpu_zone=test_scripts_details[1]["tpu_zone"],
          time_out_in_min=test_scripts_details[1]["time_out_in_min"],
          test_name=f"{test_name_prefix}-stable-{model}",
          run_model_cmds=(
              f"export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash end_to_end/{test_scripts_details[1]['script_name']}.sh",
          ),
          docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
          test_owner=test_owner.ANISHA_M,
          cluster_name=test_scripts_details[1]["cluster_name"],
      ).run(gcs_location=shared_gcs_location)

    test_group_id = "chained_tests" + "_" + model + "_" + "nightly"

    with TaskGroup(group_id=test_group_id, prefix_group_id=False) as group:
      shared_gcs_location = name_format.generate_gcs_folder_location.override(
          task_id=f"{test_group_id}_generate_gcs_folder_location"
      )(
          gcs_subfolder,
          test_group_id,
      )
      nightly_cpu = gke_config.get_maxtext_cpu_end_to_end_gke_config(
          device_type=test_scripts_details[0]["cpu_device_type"],
          cpu_zone=test_scripts_details[0]["cpu_zone"],
          time_out_in_min=test_scripts_details[0]["time_out_in_min"],
          test_name=f"{test_name_prefix}-nightly-{model}",
          run_model_cmds=(
              f"export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash end_to_end/{test_scripts_details[0]['script_name']}.sh",
          ),
          cluster_name=test_scripts_details[0]["cluster_name"],
          docker_image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
          test_owner=test_owner.ANISHA_M,
      ).run(gcs_location=shared_gcs_location)
      nightly_tpu = gke_config.get_gke_config(
          tpu_version=test_scripts_details[1]["tpu_version"],
          tpu_cores=test_scripts_details[1]["tpu_cores"],
          tpu_zone=test_scripts_details[1]["tpu_zone"],
          time_out_in_min=test_scripts_details[1]["time_out_in_min"],
          test_name=f"{test_name_prefix}-nightly-{model}",
          run_model_cmds=(
              f"export BASE_OUTPUT_PATH=$GCS_OUTPUT; bash end_to_end/{test_scripts_details[1]['script_name']}.sh",
          ),
          docker_image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
          test_owner=test_owner.ANISHA_M,
          cluster_name=test_scripts_details[1]["cluster_name"],
      ).run(gcs_location=shared_gcs_location)
      stable_cpu >> stable_tpu >> nightly_cpu >> nightly_tpu