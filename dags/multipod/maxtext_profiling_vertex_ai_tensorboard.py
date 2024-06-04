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
A DAG to test MaxText Automatic Profile Upload to Vertex AI Tensorboard.
"""
import datetime
from airflow import models
from dags import composer_env, test_owner, gcs_bucket
from dags.vm_resource import TpuVersion, Zone, DockerImage
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_profiling_vertex_ai_tensorboard",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable", "nightly", "vertex_ai"],
    start_date=datetime.datetime(2024, 6, 1),
    catchup=False,
    concurrency=2,
) as dag:
    base_output_directory = (
        f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_vertex_ai_tensorboard"
    )
    dataset_path = gcs_bucket.MAXTEXT_DIR
    docker_images = [
        (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE),
        (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
    ]

    for mode, image in docker_images:
        profiling_in_vertex_ai_tb_cmds = (
            f"export RUN_NAME=vertex_ai_{mode.value}_$(date +%Y-%m-%d-%H-%M-%S)",
            "python3 MaxText/train.py MaxText/configs/base.yml"
            f" run_name=$RUN_NAME base_output_directory={base_output_directory}"
            f" dataset_path={dataset_path} profiler=xplane steps=10",
            "gsutil ls gs://cloud-ai-platform-*/tensorboard-*/$EXPERIMENT_NAME",
        )
        profiling_in_vertex_ai_tb_test = gke_config.get_gke_config(
            tpu_version=TpuVersion.V4,
            tpu_cores=8,
            tpu_zone=Zone.US_CENTRAL2_B.value,
            time_out_in_min=240,
            test_name=f"profiling-vertex-ai-tensorboard-{mode.value}",
            run_model_cmds=profiling_in_vertex_ai_tb_cmds,
            docker_image=image.value,
            test_owner=test_owner.SURBHI_J,
        ).run(use_vertex_tensorboard=True)
