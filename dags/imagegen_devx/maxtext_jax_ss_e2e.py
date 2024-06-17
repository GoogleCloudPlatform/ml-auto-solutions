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
from dags.vm_resource import Project, TpuVersion, CpuVersion, Zone, DockerImage, GpuVersion, ClusterName
from dags.imagegen_devx.configs import jax_ss_config as config
from xlml.utils import name_format

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_jax_ss_e2e",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "jax-ss"],
    start_date=datetime.datetime(2024, 6, 7),
    catchup=False,
) as dag:
  test_configs = {
      # accelerator: list of slices to test
      "v4-16": [1, 2],
      "v5-8": [1, 2],
  }
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext/jax-ss/automated/{config.get_current_datetime}"
  )
  for accelerator, slices in test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    for slice_num in slices:
     run_name = f"{slice_num}slice-V{config.tpu_versions[accelerator]}_{cores}-maxtext-jax-ss-{config.get_current_datetime}"
     maxtext_jax_ss_test = config.get_gke_jax_ss_config(
          tpu_version=config.tpu_versions[accelerator],
          tpu_cores=cores,
          num_slices=slice_num,
          cluster_name=config.cluster_names[accelerator].value,
          tpu_zone=config.tpu_zones[accelerator].value,
          project_name=config.project_names[accelerator].value,
          time_out_in_min=60,
          run_model_cmds = (
            f"python MaxText/train.py MaxText/configs/base.yml run_name={run_name} "
            "steps=30 per_device_batch_size=1 max_target_length=4096 model_name=llama2-7b "
            "enable_checkpointing=false attention=dot_product remat_policy=minimal_flash use_iota_embed=true scan_layers=false "
            "dataset_type=synthetic async_checkpointing=false "
            f"base_output_directory={base_output_directory}",
          ),
          test_name=f"maxtext-jax-ss-{accelerator}-{slice_num}x",
          docker_image=DockerImage.MAXTEXT_TPU_JAX_SS.value,
          test_owner=test_owner.PARAM_B,
      ).run()
