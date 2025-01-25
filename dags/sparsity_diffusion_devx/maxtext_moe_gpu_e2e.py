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
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config

# Run once a day at 11 am UTC (3 am PST)
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None
SCANNED_CHECKPOINT = "gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_nightly-2025-01-09-05-00-18//8x7b/scanned_ckpt/0/items"
UNSCANNED_CKPT_PATH = "gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_nightly-2025-01-09-05-00-18//unscanned_ckpt/checkpoints/0/items"


with models.DAG(
    dag_id="maxtext_moe_gpu_e2e",
    schedule=SCHEDULED_TIME,
    tags=[
        "sparsity_diffusion_devx",
        "multipod_team",
        "maxtext",
        "gpu",
        "stable",
        "nightly",
    ],
    start_date=datetime.datetime(2024, 12, 11),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext"

  timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

  test_models_gpu = {
      "mixtral-8x7b-1node": (
          f"SCANNED_CHECKPOINT={SCANNED_CHECKPOINT} \
            UNSCANNED_CKPT_PATH={UNSCANNED_CKPT_PATH} \
            bash end_to_end/gpu/test_mixtral.sh",
          1,
      ),
      "mixtral-8x7b-1node": (
          f"SCANNED_CHECKPOINT={SCANNED_CHECKPOINT} \
            UNSCANNED_CKPT_PATH={UNSCANNED_CKPT_PATH} \
            bash end_to_end/gpu/test_mixtral.sh",
          2,
      ),
  }

  for model, (test_script, nnodes) in test_models_gpu.items():
    # TODO(b/392001211): Add these once the bug is fixed
    # pinned_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
    #     time_out_in_min=60,
    #     test_name=f"{test_name_prefix}-pinned-{model}",
    #     run_model_cmds=(test_script,),
    #     num_slices=nnodes,
    #     cluster=XpkClusters.GPU_A3_CLUSTER,
    #     docker_image='gcr.io/supercomputer-testing/yooh/maxtext-pinned',
    #     test_owner=test_owner.MICHELLE_Y,
    # ).run()
    # pinned_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
    #     time_out_in_min=300,
    #     test_name=f"{test_name_prefix}-pinned-{model}",
    #     run_model_cmds=(test_script,),
    #     num_slices=nnodes,
    #     cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
    #     docker_image='gcr.io/supercomputer-testing/yooh/maxtext-pinned',
    #     test_owner=test_owner.MICHELLE_Y,
    # ).run()
    stable_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=60,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3_CLUSTER,
        # docker_image='gcr.io/supercomputer-testing/yooh/maxtext_gpu_jax_stable_stack_0.4.35',
        docker_image='gcr.io/supercomputer-testing/yooh/maxtext-stable-stack',
        test_owner=test_owner.MICHELLE_Y,
    ).run()
    stable_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=60,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
        # docker_image='gcr.io/supercomputer-testing/yooh/maxtext_gpu_jax_stable_stack_0.4.35',
        docker_image='gcr.io/supercomputer-testing/yooh/maxtext-stable-stack',
        test_owner=test_owner.MICHELLE_Y,
    ).run()
    stable_a3_gpu >> stable_a3plus_gpu
    # TODO(b/392001211): replace with this once the bug is fixed
    # pinned_a3_gpu >> pinned_a3plus_gpu >> stable_a3_gpu >> stable_a3plus_gpu
