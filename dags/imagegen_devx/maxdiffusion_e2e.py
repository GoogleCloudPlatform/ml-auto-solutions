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

"""A DAG to run end-to-end MaxText JAX Stable Stack tests."""


import datetime
from airflow import models
from dags import composer_env, test_owner, gcs_bucket
from dags.vm_resource import Project, TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, XpkClusters
from dags.imagegen_devx.configs import gke_config as config
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxdiffusion_e2e",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxdiffusion"],
    start_date=datetime.datetime(2024, 9, 12),
    catchup=False,
) as dag:
  maxdiffusion_test_configs = {
      # accelerator: list of slices to test
      "v6e-256": [1, 2],
      "v4-8": [1, 2],
  }
  current_datetime = config.get_current_datetime()
  for accelerator, slices in maxdiffusion_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      maxdiffusion_sdxl_test = config.get_gke_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=60,
          run_model_cmds=(
              f"JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
              f"pip install . && python src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml "
              f"pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0 "
              f"revision=refs/pr/95 activations_dtype=bfloat16 weights_dtype=bfloat16 "
              f"dataset_name=gs://jfacevedo-maxdiffusion-v5p/pokemon-datasets/pokemon-gpt4-captions_xl resolution=1024 per_device_batch_size=1 "
              f"jax_cache_dir=gs://jfacevedo-maxdiffusion/cache_dir/ max_train_steps=20 attention=flash run_name=sdxl-fsdp-v5p-64-ddp enable_profiler=True "
              f"run_name={slice_num}slice-V{cluster.device_version}_{cores}-maxdiffusion-jax-stable-stack-{current_datetime} "
              f"output_dir={gcs_bucket.BASE_OUTPUT_DIR}/maxdiffusion/automated/maxdiffusion_sdxl/{current_datetime}",
          ),
          test_name=f"maxd-sdxl-{accelerator}-{slice_num}x",
          docker_image=DockerImage.MAXDIFFUSION_TPU_JAX_STABLE_STACK.value,
          test_owner=test_owner.PARAM_B,
      ).run()
      maxdiffusion_sdxl_nan_test = config.get_gke_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=60,
          run_model_cmds=(
              f"JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
              f"pip install . && bash end_to_end/tpu/test_sdxl_training_loss.sh "
              f"OUTPUT_DIR={gcs_bucket.BASE_OUTPUT_DIR}/maxdiffusion/automated/maxd-sdxl-nan/{current_datetime} "
              f"RUN_NAME={slice_num}slice-V{cluster.device_version}_{cores}-maxdiffusion-jax-stable-stack-{current_datetime} "
              f"STEPS=20 "
              f"LOSS_THRESHOLD=100",
          ),
          test_name=f"maxd-sdxl-nan-{accelerator}-{slice_num}x",
          docker_image=DockerImage.MAXDIFFUSION_TPU_JAX_STABLE_STACK.value,
          test_owner=test_owner.PARAM_B,
      ).run()
      maxdiffusion_sdxl_test >> maxdiffusion_sdxl_nan_test
