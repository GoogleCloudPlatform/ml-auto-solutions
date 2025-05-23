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

"""
A DAG to run MaxText multi-tier checkpointing tests.
"""
import datetime
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode  # Run once a day at 10 am UTC (2 am PST)
from dags.common import test_owner

SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_multi_tier_checkpointing_recover",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "multi_tier_checkpointing_recover",
    ],
    start_date=datetime.datetime(2025, 5, 15),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.MTC_BUCKET}/maxtext_multi_tier_checkpointing_recover"
  )
  dataset_path = gcs_bucket.MTC_BUCKET
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]
  test_configs = {
      # accelerator: list of slices to test
      "v5p-8": [2],
  }
  clusters = {
      # accelerator: cluster name
      "v5p-8": XpkClusters.TPU_V5P_8_CLUSTER,
  }

  for mode, image in docker_images:
    for accelerator, slices in test_configs.items():
      for slice_num in slices:
        command = (
            "bash end_to_end/test_mtc_phase_2_save_path.sh"
            f" multitiercheckpointing-{slice_num}x-{accelerator}"
            f" {base_output_directory} {dataset_path}",
        )
        maxtext_save_checkpoint = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name="maxtext-multi-tier-checkpointing-recover",
            run_model_cmds=command,
            docker_image=image.value,
            test_owner=test_owner.JACKY_F,
        ).run_with_interruption(
            ramdisk_directory="local",
            xpk_branch="main",
            skip_post_process=True,
            mtc_enabled=True,
        )

        clean_cmd = (f"rm -rf /local/*",)
        clean_ramdisk_one = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name="clean-ramdisk",
            run_model_cmds=clean_cmd,
            docker_image=image.value,
            test_owner=test_owner.JACKY_F,
        ).run(
            ramdisk_directory="local",
            xpk_branch="main",
            skip_post_process=True,
            mtc_enabled=True,
        )

        (maxtext_save_checkpoint >> clean_ramdisk_one)
