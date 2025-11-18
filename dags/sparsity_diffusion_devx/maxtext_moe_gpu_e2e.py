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

"""A DAG to run end-to-end MoE tests on GPU."""


import datetime

from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config


# Run once a day at 11 am UTC (3 am PST)
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None

SCANNED_CHECKPOINT = "gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_nightly-2025-01-09-05-00-18//8x7b/scanned_ckpt/0/items"
UNSCANNED_CKPT_PATH = "gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_nightly-2025-01-09-05-00-18//unscanned_ckpt/checkpoints/0/items"


def run_maxtext_tests():
  test_name_prefix = "maxtext"

  test_models_gpu = {
      "mixtral-8x7b-1node": (
          f"SCANNED_CHECKPOINT={SCANNED_CHECKPOINT} \
            UNSCANNED_CKPT_PATH={UNSCANNED_CKPT_PATH} \
            BASE_OUTPUT_PATH=$GCS_OUTPUT \
            bash end_to_end/gpu/mixtral/test_8x7b.sh",
          1,
      ),
      "mixtral-8x7b-2node": (
          f"SCANNED_CHECKPOINT={SCANNED_CHECKPOINT} \
            UNSCANNED_CKPT_PATH={UNSCANNED_CKPT_PATH} \
            BASE_OUTPUT_PATH=$GCS_OUTPUT \
            bash end_to_end/gpu/mixtral/test_8x7b.sh",
          2,
      ),
  }
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  for model, (test_script, nnodes) in test_models_gpu.items():
    candidate_a3plus_gpu = (
        gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
            time_out_in_min=90,
            test_name=f"{test_name_prefix}-candidate-{model}",
            run_model_cmds=(test_script,),
            num_slices=nnodes,
            cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
            docker_image=DockerImage.MAXTEXT_GPU_JAX_AI_CANDIDATE_IMAGE.value,
            test_owner=test_owner.MICHELLE_Y,
        ).run_with_quarantine(quarantine_task_group)
    )
    stable_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=90,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE_STACK.value,
        test_owner=test_owner.MICHELLE_Y,
    ).run_with_quarantine(quarantine_task_group)
    candidate_a3plus_gpu >> stable_a3plus_gpu


with models.DAG(
    dag_id="maxtext_moe_gpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "jax_models_and_performance",
        "multipod_team",
        "maxtext",
        "gpu",
        "stable",
        "nightly",
        "mlscale_devx",
        "h100-mega-80gb-8",
        "a3mega",
    ],
    start_date=datetime.datetime(2024, 12, 11),
    catchup=False,
) as dag:
  run_maxtext_tests()
