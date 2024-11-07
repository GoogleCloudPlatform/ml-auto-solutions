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
from dags import composer_env, test_owner, gcs_bucket
from dags.vm_resource import Project, TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, XpkClusters
from dags.imagegen_devx.configs import gke_config as config
from xlml.utils import name_format

# Run once a day at 3 am UTC (7 pm PST)
SCHEDULED_TIME = "0 3 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="jax_stable_tpu_stack_e2e",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "maxdiffusion", "axlearn", "tpu"  "jax-stable-stack"],
    start_date=datetime.datetime(2024, 6, 7),
    catchup=False,
) as dag:
  current_datetime = config.get_current_datetime()
  maxtext_test_configs = {
      # accelerator: list of slices to test
      "v4-16": [1, 2],
      "v6e-256": [1],
  }
  maxdiffusion_test_configs = {
      # accelerator: list of slices to test
      "v4-8": [1],
      "v6e-256": [1],
  }
  axlearn_test_configs = {
      # accelerator: list of slices to test
      "v4-16": [1, 2],
  }

  for accelerator, slices in maxtext_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      maxtext_jax_stable_stack_test = config.get_gke_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=60,
          run_model_cmds=(
              f"JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
              f"python MaxText/train.py MaxText/configs/base.yml run_name={slice_num}slice-V{cluster.device_version}_{cores}-maxtext-jax-stable-stack-{current_datetime} "
              "steps=30 per_device_batch_size=1 max_target_length=4096 model_name=llama2-7b "
              "enable_checkpointing=false attention=dot_product remat_policy=minimal_flash use_iota_embed=true scan_layers=false "
              "dataset_type=synthetic async_checkpointing=false "
              f"base_output_directory={gcs_bucket.BASE_OUTPUT_DIR}/maxtext/jax-stable-stack/automated/{current_datetime}",
          ),
          test_name=f"maxtext-jax-stable-stack-{accelerator}-{slice_num}x",
          docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK.value,
          test_owner=test_owner.PARAM_B,
      ).run()

  for accelerator, slices in maxdiffusion_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      maxdiffusion_jax_stable_stack_test = config.get_gke_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=60,
          run_model_cmds=(
              f"JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
              f"pip install . && python src/maxdiffusion/train.py src/maxdiffusion/configs/base_2_base.yml "
              f"run_name={slice_num}slice-V{cluster.device_version}_{cores}-maxdiffusion-jax-stable-stack-{current_datetime} "
              f"output_dir={gcs_bucket.BASE_OUTPUT_DIR}/maxdiffusion/jax-stable-stack/automated/{current_datetime}",
          ),
          test_name=f"maxdiffusion-jax-stable-stack-{accelerator}-{slice_num}x",
          docker_image=DockerImage.MAXDIFFUSION_TPU_JAX_STABLE_STACK.value,
          test_owner=test_owner.PARAM_B,
      ).run()

  for accelerator, slices in axlearn_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      axlearn_jax_stable_stack_test = config.get_gke_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=180,
          run_model_cmds=(
              "JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true TPU_SLICE_BUILDER_DUMP_CHIP_FORCE=true TPU_SLICE_BUILDER_DUMP_ICI=true JAX_FORCE_TPU_INIT=true ENABLE_TPUNETD_CLIENT=true && "
              "cd axlearn && python -m axlearn.common.launch_trainer_main "
              f"--module=text.gpt.c4_trainer --config=fuji-test-v1 "
              f"--trainer_dir={gcs_bucket.BASE_OUTPUT_DIR}/bite/jax-stable-stack/automated/{current_datetime} "
              f"--data_dir={gcs_bucket.AXLEARN_DIR} --jax_backend=tpu ",
          ),
          test_name=f"axlearn-jax-stable-stack-{accelerator}-{slice_num}x",
          docker_image=DockerImage.AXLEARN_TPU_JAX_STABLE_STACK.value,
          test_owner=test_owner.PARAM_B,
      ).run()
