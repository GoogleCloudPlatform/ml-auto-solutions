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

"""A DAG to run end-to-end JAX Stable Stack tests for GCP GPUs."""


import datetime
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.sparsity_diffusion_devx.configs import gke_config as config
from xlml.utils import name_format

# Run once a day at 3 am UTC (7 pm PST)
SCHEDULED_TIME = "0 3 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="project_bite_gpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "gpu",
        "axlearn",
        "bite",
    ],
    start_date=datetime.datetime(2024, 11, 12),
    catchup=False,
) as dag:
  current_datetime = config.get_current_datetime()

  axlearn_test_configs = {
      # accelerator: list of slices to test
      "a3plus": [1, 2],
  }

  for accelerator, slices in axlearn_test_configs.items():
    cores = accelerator.rsplit("-", maxsplit=1)[-1]
    cluster = config.clusters[accelerator]
    for slice_num in slices:
      maxtext_jax_stable_stack_test = config.get_gpu_gke_test_config(
          num_slices=slice_num,
          cluster=cluster,
          time_out_in_min=300,
          run_model_cmds=(
              "cd axlearn && "
              "XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 TF_FORCE_GPU_ALLOW_GROWTH=true "
              "python -m axlearn.common.launch_trainer_main "
              f"--module=text.gpt.c4_trainer --config=fuji-test-v1 "
              f"--trainer_dir={gcs_bucket.BASE_OUTPUT_DIR}/bite/gpu/jax-stable-stack/automated/{current_datetime} "
              f"--data_dir={gcs_bucket.AXLEARN_DIR} --jax_backend=gpu ",
          ),
          test_name=f"axlearn-jax-nightly-{accelerator}-{slice_num}x",
          docker_image=DockerImage.AXLEARN_GPU_JAX_NIGHTLY.value,
          test_owner=test_owner.Maggie_Z,
      ).run()
