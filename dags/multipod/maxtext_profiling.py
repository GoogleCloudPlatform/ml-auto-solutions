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
A DAG to run MaxText profiling tests.
"""
import datetime
from airflow import models
from dags import composer_env, test_owner, gcs_bucket
from dags.vm_resource import TpuVersion, Zone, DockerImage, ClusterName
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from xlml.apis import gcp_config, metric_config, task, test_config

# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_profiling",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 3, 25),
    catchup=False,
    concurrency=2,
) as dag:
  current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  base_output_directory = gcs_bucket.XLML_OUTPUT_DIR
  dataset_path = gcs_bucket.MAXTEXT_DIR
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_JAX_STABLE),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_JAX_NIGHTLY),
  ]

  profiling_command = f'''
  echo "Start profiling tests....."
  cd maxtext
  RUN_NAME='profiling_tests_{current_time}'
  python3 MaxText/train.py MaxText/configs/base.yml run_name=\$RUN_NAME base_output_directory={base_output_directory} dataset_path={dataset_path} enable_profiler=true steps=20
  gsutil cp -R {base_output_directory}/\$RUN_NAME/tensorboard .
  # Uninstall both stable and nightly versions of tensorboard_plugin_profile
  pip3 uninstall -y tbp-nightly && pip3 uninstall -y tensorboard_plugin_profile
  # profiler test with stable tensorboard_plugin_profile
  pip3 install tensorboard_plugin_profile
  python3 MaxText/tests/profiler_test.py
  # profiler test with tbp-nightly
  pip3 uninstall -y tensorboard_plugin_profile && pip3 install tbp-nightly
  python3 MaxText/tests/profiler_test.py
  exit 0
  '''

  for mode, image in docker_images:
    maxtext_v4_configs_test = gke_config.get_gke_config(
          tpu_version=TpuVersion.V4,
          tpu_cores=8,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          test_name=f"maxtext-profiling-v4-{mode.value}",
          run_model_cmds=profiling_command,
          docker_image=image.value,
          test_owner=test_owner.SURBHI_J,
      ).run()
