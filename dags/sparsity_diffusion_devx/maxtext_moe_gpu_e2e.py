# Copyright 2025 Google LLC
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

"""A DAG to run end-to-end MoE tests on GPU."""


import datetime
from airflow import models
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config

# Run once a day at 11 am UTC (3 am PST)
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_moe_gpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "sparsity_diffusion_devx",
        "multipod_team",
        "maxtext",
        "gpu",
        "stable",
        "nightly",
        "mlscale_onduty",
    ],
    start_date=datetime.datetime(2024, 12, 11),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"

  timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
  train_base = (
      "python3 MaxText/train.py MaxText/configs/base.yml model_name=mixtral-8x7b "
      "base_output_directory=gs://runner-maxtext-logs dataset_path=gs://maxtext-dataset "
      "steps=2 per_device_batch_size=1 hardware=gpu dataset_type=synthetic attention=cudnn_flash_te "
      "remat_policy=full use_iota_embed=True capacity_factor=1.0 "
      "reuse_example_batch=1 enable_checkpointing=False megablox=False "
      "weight_dtype=bfloat16 ici_expert_parallelism=-1 ici_fsdp_parallelism=1"
  )
  test_models_gpu = {
      "mixtral-8x7b-1node": (
          f"{train_base} run_name=runner-{timestamp}-1",
          1,
      ),
      "mixtral-8x7b-2node": (
          f"{train_base} run_name=runner-{timestamp}-2",
          2,
      ),
  }

  for model, (test_script, nnodes) in test_models_gpu.items():
    pinned_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=60,
        test_name=f"{test_name_prefix}-pinned-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.MICHELLE_Y,
    ).run()
    pinned_a3_gpu
