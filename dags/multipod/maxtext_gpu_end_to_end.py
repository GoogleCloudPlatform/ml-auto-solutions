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
from dags.vm_resource import XpkClusters, CpuVersion, DockerImage, GpuVersion, Project, TpuVersion, Zone
from dags.multipod.configs import gke_config
from airflow.utils.task_group import TaskGroup
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_gpu_end_to_end",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"

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

  for model, (test_script, nnodes) in test_models_gpu.items():
    pinned_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-pinned-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.NINA_C,
    ).run()
    stable_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE.value,
        test_owner=test_owner.NINA_C,
    ).run()
    pinned_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-pinned-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.NINA_C,
    ).run()
    stable_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=300,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE.value,
        test_owner=test_owner.NINA_C,
    ).run()
    pinned_a3_gpu >> stable_a3_gpu >> pinned_a3plus_gpu >> stable_a3plus_gpu
