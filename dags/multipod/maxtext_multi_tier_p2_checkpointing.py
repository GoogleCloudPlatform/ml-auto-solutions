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
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode  # Run once a day at 10 am UTC (2 am PST)

SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_muti_tier_p2_checkpointing",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "multi_tier_checkpointing_p2",
        "nightly",
    ],
    start_date=datetime.datetime(2025, 4, 17),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_multi_tier_p2_checkpointing"
  )
  dataset_path = gcs_bucket.MAXTEXT_DIR
  docker_images = [
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]
  test_configs = {
      # accelerator: list of slices to test
      "v6e-16": [3],
  }
  clusters = {
      # accelerator: cluster name
      "v6e-16": XpkClusters.TPU_V6E_16_IN_MEM_CLUSTER,
  }

  for mode, image in docker_images:
    for accelerator, slices in test_configs.items():
      for slice_num in slices:
        command = (
            "bash end_to_end/test_mtc_phase_2_save_path.sh"
            f" multi_tier_checkpointing-{slice_num}x-{accelerator}"
            f" {base_output_directory} {dataset_path}",
        )
        maxtext_v6e_chkpt_save_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name="maxtext-multi-tier-checkpointing-p2-save",
            run_model_cmds=command,
            docker_image=image.value,
            test_owner=test_owner.ABHINAV_S,
        ).run(ramdisk_directory="local", mtc_enabled=True)

        command = "rm -rf /local/*"
        ramdisk_single_slice_cleanup = gke_config.get_gke_config(
            num_slices=1,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name="maxtext-multi-tier-checkpointing-p2-emulate-disruption",
            run_model_cmds=command,
            docker_image=image.value,
            test_owner=test_owner.ABHINAV_S,
        ).run(ramdisk_directory="local", mtc_enabled=True)
        command = (
            "bash end_to_end/test_mtc_phase_2_save_path.sh"
            f" multi_tier_checkpointing-{slice_num}x-{accelerator}"
            f" {base_output_directory} {dataset_path}",
        )

        maxtext_v6e_chkpt_restore_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name="maxtext-multi-tier-checkpointing-p2-restore",
            run_model_cmds=command,
            docker_image=image.value,
            test_owner=test_owner.ABHINAV_S,
        ).run(ramdisk_directory="local", mtc_enabled=True)

        (
            maxtext_v6e_chkpt_save_test
            >> ramdisk_single_slice_cleanup
            >> maxtext_v6e_chkpt_restore_test
        )
