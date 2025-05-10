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
import os
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import Project, TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, XpkClusters
from dags.sparsity_diffusion_devx.configs import gke_config as config
from xlml.apis import metric_config
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None

BASE_OUTPUT_DIRECTORY = gcs_bucket.BASE_OUTPUT_DIR


with models.DAG(
    dag_id="maxdiffusion_tpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "sparsity_diffusion_devx",
        "multipod_team",
        "maxdiffusion",
        "mlscale_devx",
    ],
    start_date=datetime.datetime(2024, 9, 12),
    catchup=False,
) as dag:
  maxdiffusion_test_configs = {
      # accelerator: list of slices to test
      "v6e-256": [1, 2],
      "v4-8": [1, 2],
  }
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  # The concrete run_name will be generated at runtime in `run_with_name_gen_and_quarantine`
  # and passed to the underlying maxdiffusion trainer script via the environment variable
  # JOBSET_NAME.
  #
  # Also note that the accelerator type, core counts, and slice num will be automatically
  # added by the name gen.
  sdxl_base_output_dir = (
      f"{BASE_OUTPUT_DIRECTORY}/maxdiffusion/automated/maxdiffusion_sdxl"
  )
  sdxl_run_name_prefix = f"maxd-sdxl-jax-stable-stack"
  sdxl_tensorboard_summary_config = metric_config.SummaryConfig(
      file_location=sdxl_base_output_dir,
      aggregation_strategy=metric_config.AggregationStrategy.MEDIAN,
      use_regex_file_location=True,
  )
  sdxl_nan_base_output_dir = (
      f"{BASE_OUTPUT_DIRECTORY}/maxdiffusion/automated/maxd-sdxl-nan"
  )
  sdxl_nan_run_name_prefix = f"maxd-sdxl-nan-jax-stable-stack"
  sdxl_nan_tensorboard_summary_config = metric_config.SummaryConfig(
      file_location=sdxl_nan_base_output_dir,
      aggregation_strategy=metric_config.AggregationStrategy.MEDIAN,
      use_regex_file_location=True,
  )

  for accelerator, slices in maxdiffusion_test_configs.items():
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
              f"jax_cache_dir=gs://jfacevedo-maxdiffusion/cache_dir/ max_train_steps=20 attention=flash enable_profiler=True "
              f"run_name='' "
              f"output_dir={sdxl_base_output_dir}",
          ),
          test_name=sdxl_run_name_prefix,
          docker_image=DockerImage.MAXDIFFUSION_TPU_JAX_STABLE_STACK.value,
          test_owner=test_owner.PARAM_B,
          tensorboard_summary_config=sdxl_tensorboard_summary_config,
      ).run_with_name_gen_and_quarantine(
          quarantine_task_group,
          run_name_env="JOBSET_NAME",
          nested_run_name_in_tb_file_location=False,
      )

      maxdiffusion_sdxl_nan_test = config.get_gke_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=60,
          run_model_cmds=(
              f"JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
              f"pip install . && bash end_to_end/tpu/test_sdxl_training_loss.sh "
              f"OUTPUT_DIR={sdxl_nan_base_output_dir} "
              f"RUN_NAME='' "
              f"STEPS=20 "
              f"LOSS_THRESHOLD=100",
          ),
          test_name=sdxl_nan_run_name_prefix,
          docker_image=DockerImage.MAXDIFFUSION_TPU_JAX_STABLE_STACK.value,
          test_owner=test_owner.PARAM_B,
          tensorboard_summary_config=sdxl_nan_tensorboard_summary_config,
      ).run_with_name_gen_and_quarantine(
          quarantine_task_group,
          run_name_env="JOBSET_NAME",
          nested_run_name_in_tb_file_location=False,
      )
      maxdiffusion_sdxl_test >> maxdiffusion_sdxl_nan_test
