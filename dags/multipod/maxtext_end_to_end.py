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
from dags.vm_resource import ClusterName, CpuVersion, DockerImage, GpuVersion, Project, TpuVersion, Zone
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
      "llama2-7b": "tpu/llama2/7b/test_llama2_7b",
      "mistral": "tpu/test_mistral",
      "gemma-2b": "tpu/gemma/2b/test_gemma",
      "gpt3": "tpu/test_gpt3",
  }

  timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
  train_base = (
      "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
      "python3 MaxText/train.py MaxText/configs/base.yml "
      "base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset "
      "steps=2 enable_checkpointing=false attention=dot_product"
  )
  decode_base = (
      "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
      "python3 MaxText/decode.py MaxText/configs/base.yml "
      "base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset "
      "steps=2 enable_checkpointing=false attention=dot_product "
      "max_target_length=128 per_device_batch_size=1"
  )
  test_models_gpu = {
      "train-c4-data": (f"{train_base} run_name=runner-{timestamp}-0", 1),
      "train-synthetic-data": (
          f"{train_base} run_name=runner-{timestamp}-1 dataset_type=synthetic",
          1,
      ),
      "train-flash": (
          f"{train_base} run_name=runner-{timestamp}-2 attention=cudnn_flash_te",
          1,
      ),
      "train-quarter-batch-size": (
          f"{train_base} run_name=runner-{timestamp}-3 per_device_batch_size=0.25 ici_tensor_parallelism=4",
          1,
      ),
      "train-int8": (
          f"{train_base} run_name=runner-{timestamp}-6 quantization=int8",
          1,
      ),
      "train-fp8": (
          f"{train_base} run_name=runner-{timestamp}-7 quantization=fp8",
          1,
      ),
      "decode": (f"{decode_base} run_name=runner-{timestamp}-4", 1),
      "decode-quarter-batch-size": (
          f"{decode_base} run_name=runner-{timestamp}-5 per_device_batch_size=.25 ici_tensor_parallelism=4",
          1,
      ),
      "generate-param-only-checkpoint": (
          "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
          f"bash end_to_end/test_generate_param_only_checkpoint.sh -r runner-{timestamp}-8 "
          "-o gs://runner-maxtext-logs -d gs://maxtext-dataset -i 4 -a dot_product",
          1,
      ),
      "generate-param-only-checkpoint-int8": (
          "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
          f"bash end_to_end/test_generate_param_only_checkpoint.sh -r runner-{timestamp}-9 "
          "-o gs://runner-maxtext-logs -d gs://maxtext-dataset -i 4 -q int8 -a dot_product",
          1,
      ),
      "grain-checkpoint-determinism": (
          "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
          "bash end_to_end/test_checkpointing.sh runner gs://runner-maxtext-logs "
          "gs://maxtext-dataset False c4-array_record dot_product",
          1,
      ),
      "checkpoint-compatibility": (
          "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
          "bash end_to_end/test_checkpoint_compatibility.sh runner "
          "gs://runner-maxtext-logs gs://maxtext-dataset dot_product",
          1,
      ),
      "llama2-7b-train-1node": ("bash MaxText/configs/a3/llama_2_7b/1vm.sh", 1),
      "llama2-7b-train-2node": ("bash MaxText/configs/a3/llama_2_7b/2vm.sh", 2),
      "llama2-7b": ("bash end_to_end/gpu/a3/test_llama2_7b.sh", 1),
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
    pinned_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100,
        gpu_zone=Zone.US_CENTRAL1_C.value,
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-pinned-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster_name=ClusterName.A3_CLUSTER.value,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.NINA_C,
    ).run()
    stable_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100,
        gpu_zone=Zone.US_CENTRAL1_C.value,
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster_name=ClusterName.A3_CLUSTER.value,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE.value,
        test_owner=test_owner.NINA_C,
    ).run()
    pinned_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100_MEGA,
        gpu_zone=Zone.AUSTRALIA_SOUTHEAST1_C.value,
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-pinned-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster_name=ClusterName.A3PLUS_CLUSTER.value,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.NINA_C,
    ).run()
    stable_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100_MEGA,
        gpu_zone=Zone.AUSTRALIA_SOUTHEAST1_C.value,
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster_name=ClusterName.A3PLUS_CLUSTER.value,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE.value,
        test_owner=test_owner.NINA_C,
    ).run()
    pinned_a3_gpu >> stable_a3_gpu >> pinned_a3plus_gpu >> stable_a3plus_gpu

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
              "time_out_in_min": 240,
          },
          {
              "script_name": "tpu/mixtral/8x7b/2_test_mixtral",
              "tpu_version": TpuVersion.V4,
              "tpu_cores": 128,
              "cluster_name": ClusterName.V4_128_MULTISLICE_CLUSTER.value,
              "project_name": Project.CLOUD_TPU_MULTIPOD_DEV.value,
              "tpu_zone": Zone.US_CENTRAL2_B.value,
              "time_out_in_min": 60,
          },
      ],
      "mixtral-8x22b": [
          {
              "script_name": "tpu/mixtral/8x22b/1_test_mixtral",
              "cpu_device_type": CpuVersion.M1_MEGAMEM,
              "cpu_zone": Zone.US_CENTRAL1_B.value,
              "cluster_name": ClusterName.CPU_M1_MEGAMEM_96.value,
              "time_out_in_min": 360,
          },
          {
              "script_name": "tpu/mixtral/8x22b/2_test_mixtral",
              "tpu_version": TpuVersion.V5E,
              "tpu_cores": 256,
              "cluster_name": ClusterName.V5E_256_US_WEST_4_MULTISLICE_CLUSTER.value,
              "tpu_zone": Zone.US_WEST4_B.value,
              "time_out_in_min": 60,
          },
      ],
      "llama2-70b": [
          {
              "script_name": "tpu/llama2/70b/1_test_llama2_70b",
              "cpu_device_type": CpuVersion.M1_MEGAMEM,
              "cpu_zone": Zone.US_CENTRAL1_B.value,
              "cluster_name": ClusterName.CPU_M1_MEGAMEM_96.value,
              "time_out_in_min": 360,
          },
          {
              "script_name": "tpu/llama2/70b/2_test_llama2_70b",
              "tpu_version": TpuVersion.V4,
              "tpu_cores": 128,
              "cluster_name": ClusterName.V4_128_MULTISLICE_CLUSTER.value,
              "project_name": Project.CLOUD_TPU_MULTIPOD_DEV.value,
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

    test_group_id = "chained_tests" + "_" + model + "_" + "jax_stable_stack"

    with TaskGroup(group_id=test_group_id, prefix_group_id=False) as group:
      shared_gcs_location = name_format.generate_gcs_folder_location.override(
          task_id=f"{test_group_id}_generate_gcs_folder_location"
      )(
          gcs_subfolder,
          test_group_id,
      )
      (stable_cpu >> stable_tpu >> nightly_cpu >> nightly_tpu)
