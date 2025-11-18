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
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode

SCHEDULED_TIME = None

with models.DAG(
    dag_id="maxtext_profiling_vertex_ai_tensorboard",
    schedule=SCHEDULED_TIME,
    tags=["maxtext", "TPU", "v4-8", "v4-16"],
    start_date=datetime.datetime(2024, 6, 1),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_vertex_ai_tensorboard"
  )
  dataset_path = gcs_bucket.MAXTEXT_DIR
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_NIGHTLY),
  ]
  test_configs = {
      # accelerator: list of slices to test
      "v4-8": [1],
      "v4-16": [1, 2],
  }
  clusters = {
      # accelerator: cluster name
      "v4-8": XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
      "v4-16": XpkClusters.TPU_V4_16_CLUSTER,
  }

  for mode, image in docker_images:
    for accelerator, slices in test_configs.items():
      for slice_num in slices:
        current_time = datetime.datetime.now()
        current_datetime = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        profiling_in_vertex_ai_tb_cmds = (
            f"export RUN_NAME=vertex-ai-{mode.value}-{slice_num}x-{accelerator}-{current_datetime}",
            "python3 -m MaxText.train MaxText/configs/base.yml"
            f" run_name=$RUN_NAME base_output_directory={base_output_directory}"
            f" dataset_path={dataset_path} profiler=xplane steps=10",
            "gsutil ls gs://cloud-ai-platform-*/tensorboard-*/$EXPERIMENT_NAME",
        )
        profiling_in_vertex_ai_tb_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=240,
            test_name=f"profiling-vertex-ai-tensorboard-{mode.value}",
            run_model_cmds=profiling_in_vertex_ai_tb_cmds,
            docker_image=image.value,
            test_owner=test_owner.SURBHI_J,
        ).run(use_vertex_tensorboard=True)
