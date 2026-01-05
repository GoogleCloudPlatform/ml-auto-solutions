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
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, DockerImage
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 2 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_profiling",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "stable",
        "nightly",
        "mlscale_devx",
        "TPU",
        "v4-8",
    ],
    start_date=datetime.datetime(2024, 3, 1),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_profiling"
  dataset_path = gcs_bucket.MAXTEXT_DIR
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]

  for mode, image in docker_images:
    profiling_cmds = (
        f"export RUN_NAME=profiling_{mode.value}_$(date +%Y-%m-%d-%H-%M-%S)",
        "python3 -m MaxText.train MaxText/configs/base.yml"
        f" run_name=$RUN_NAME base_output_directory={base_output_directory}"
        f" dataset_path={dataset_path} profiler=xplane steps=20",
        f"gcloud storage cp --recursive {base_output_directory}/$RUN_NAME/tensorboard .",
    )
    maxtext_v4_configs_test = gke_config.get_gke_config(
        time_out_in_min=60,
        test_name=f"maxtext-profiling-{mode.value}",
        run_model_cmds=profiling_cmds,
        docker_image=image.value,
        test_owner=test_owner.BRANDEN_V,
    ).run()
