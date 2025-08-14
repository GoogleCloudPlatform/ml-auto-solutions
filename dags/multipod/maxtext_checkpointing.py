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

"""
A DAG to run MaxText checkpointing tests.
"""
import datetime
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_checkpointing",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "nightly",
        "mlscale_devx",
    ],
    start_date=datetime.datetime(2024, 3, 1),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_checkpointing"
  dataset_path = gcs_bucket.MAXTEXT_DIR
  current_time = datetime.datetime.now()
  current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK_CANDIDATE),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_STABLE_STACK_NIGHTLY_JAX),
  ]

  for mode, image in docker_images:
    for chkpt_mode in ["sync", "async"]:
      async_checkpointing = chkpt_mode == "async"
      run_name = f"checkpointing-{mode.value}-{chkpt_mode}-{current_datetime}"
      command = (
          "bash end_to_end/test_checkpointing.sh "
          f" {run_name} {base_output_directory} {dataset_path}"
          f" true tfds autoselected {async_checkpointing}",
      )
      maxtext_v4_configs_test = gke_config.get_gke_config(
          num_slices=1,
          cluster=XpkClusters.TPU_V5P_8_CLUSTER,
          time_out_in_min=60,
          test_name=f"maxtext-checkpointing-{mode.value}-{chkpt_mode}",
          run_model_cmds=command,
          docker_image=image.value,
          test_owner=test_owner.SURBHI_J,
      ).run()
