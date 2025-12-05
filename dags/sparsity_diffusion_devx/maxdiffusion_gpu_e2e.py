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

"""A DAG to run end-to-end MaxDiffusion GPU tests."""


import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import Project, TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, XpkClusters
from dags.sparsity_diffusion_devx.configs import gke_config as config
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "45 3 * * *" if composer_env.is_prod_env() else None

# TODO: Expand smoke test coverage to include, inference, metrics check, and models (Flux, Wan)
with models.DAG(
    dag_id="maxdiffusion_gpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "jax_models_and_performance",
        "multipod_team",
        "maxdiffusion",
        "mlscale_devx",
        "GPU",
        "h100-80gb-8",
        "h100-mega-80gb-8",
        "a3mega",
    ],
    start_date=datetime.datetime(2024, 9, 12),
    catchup=False,
) as dag:
  maxdiffusion_test_configs = {
      "a3": [1, 2],
      "a3plus": [1, 2],
  }
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )
  current_datetime = config.get_current_datetime()
  for accelerator, slices in maxdiffusion_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      maxdiffusion_sdxl_test_gpu = config.get_gpu_gke_test_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=60,
          run_model_cmds=(
              f"NVTE_FUSED_ATTN=1 python -m src.maxdiffusion.train_sdxl src/maxdiffusion/configs/base_xl.yml hardware=gpu "
              f" train_new_unet=true train_text_encoder=false "
              f"cache_latents_text_encoder_outputs=true max_train_steps=200 weights_dtype=bfloat16 "
              f"per_device_batch_size=1 attention='cudnn_flash_te' "
              f"activations_dtype=bfloat16 weights_dtype=bfloat16 "
              f"max_train_steps=200 run_name=sdxl-gpu enable_profiler=True "
              f"run_name={slice_num}slice-V{cluster.device_version}_{cores}-maxdiffusion-jax-stable-stack-{current_datetime} "
              f"output_dir={gcs_bucket.BASE_OUTPUT_DIR}/maxdiffusion/automated/maxdiffusion_sdxl/{current_datetime}",
          ),
          test_name=f"maxd-sdxl-{accelerator}-{slice_num}x",
          docker_image=DockerImage.MAXDIFFUSION_GPU_JAX_STABLE.value,
          test_owner=test_owner.KUNJAN_P,
      ).run_with_quarantine(quarantine_task_group)
      maxdiffusion_sdxl_test_gpu
