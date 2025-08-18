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

"""A DAG to run end-to-end JAX Stable Stack TPU tests."""


import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import Project, TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, XpkClusters
from dags.sparsity_diffusion_devx.configs import gke_config as config
from dags.multipod.configs.common import SetupMode
from xlml.utils import name_format
from xlml.apis import metric_config

# Run once a day at 3 am UTC (7 pm PST)
SCHEDULED_TIME = "0 3 * * *" if composer_env.is_prod_env() else None
BASE_OUTPUT_DIRECTORY = gcs_bucket.BASE_OUTPUT_DIR


with models.DAG(
    dag_id="jax_ai_image_tpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "jax_models_and_performance",
        "multipod_team",
        "maxtext",
        "maxdiffusion",
        "tpu",
        "jax-stable-stack",
        "mlscale_devx",
        "v5-8",
        "v6e-256",
    ],
    start_date=datetime.datetime(2024, 6, 7),
    catchup=False,
) as dag:
  current_datetime = config.get_current_datetime()
  maxtext_test_configs = {
      # accelerator: list of slices to test
      "v5-8": [1, 2],
      "v6e-256": [1],
  }
  maxdiffusion_test_configs = {
      # accelerator: list of slices to test
      "v5-8": [1, 2],
      "v6e-256": [1],
  }

  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  maxtext_docker_images = [
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_STABLE_STACK_NIGHTLY_JAX),
  ]

  maxdiffusion_docker_images = [
      (
          SetupMode.NIGHTLY,
          DockerImage.MAXDIFFUSION_TPU_STABLE_STACK_NIGHTLY_JAX,
      ),
  ]

  for accelerator, slices in maxtext_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      for mode, image in maxtext_docker_images:
        maxtext_jax_stable_stack_test = config.get_gke_config(
            num_slices=slice_num,
            cluster=cluster,
            time_out_in_min=60,
            run_model_cmds=(
                f"JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
                f"python -m MaxText.train MaxText/configs/base.yml run_name={slice_num}slice-V{cluster.device_version}_{cores}-maxtext-jax-stable-stack-{current_datetime} "
                "steps=30 per_device_batch_size=1 max_target_length=4096 model_name=llama2-7b "
                "enable_checkpointing=false attention=dot_product remat_policy=minimal_flash use_iota_embed=true scan_layers=false "
                "dataset_type=synthetic async_checkpointing=false "
                f"base_output_directory={gcs_bucket.BASE_OUTPUT_DIR}/maxtext/jax-stable-stack/automated/{current_datetime}",
            ),
            test_name=f"maxtext-jax-stable-stack-{mode.value}",
            docker_image=image.value,
            test_owner=test_owner.ROHAN_B,
        ).run_with_quarantine(quarantine_task_group)

  for accelerator, slices in maxdiffusion_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      for mode, image in maxdiffusion_docker_images:
        maxdiffusion_jax_stable_stack_test = config.get_gke_config(
            num_slices=slice_num,
            cluster=cluster,
            time_out_in_min=60,
            run_model_cmds=(
                f"JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
                f"pip install . && python src/maxdiffusion/train_sdxl.py src/maxdiffusion/configs/base_xl.yml "
                f"pretrained_model_name_or_path=gs://maxdiffusion-github-runner-test-assets/checkpoints/models--stabilityai--stable-diffusion-xl-base-1.0 "
                f"revision=refs/pr/95 activations_dtype=bfloat16 weights_dtype=bfloat16 "
                f"dataset_name=gs://jfacevedo-maxdiffusion-v5p/pokemon-datasets/pokemon-gpt4-captions_sdxl resolution=1024 per_device_batch_size=1 "
                f"jax_cache_dir=gs://jfacevedo-maxdiffusion/cache_dir/ max_train_steps=20 attention=flash enable_profiler=True "
                f"run_name={slice_num}slice-V{cluster.device_version}_{cores}-maxdiffusion-jax-stable-stack-{current_datetime} "
                f"output_dir={gcs_bucket.BASE_OUTPUT_DIR}/maxdiffusion-jax-stable-stack-{mode.value}-{accelerator}-{slice_num}/automated/{current_datetime}",
            ),
            test_name=f"maxdiffusion-jax-ai-image-{mode.value}",
            docker_image=image.value,
            test_owner=test_owner.ROHAN_B,
        ).run_with_quarantine(quarantine_task_group)
