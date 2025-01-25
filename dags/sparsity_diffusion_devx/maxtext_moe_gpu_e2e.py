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
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage, Project
from dags.multipod.configs import gke_config
from xlml.utils import gke


# Run once a day at 11 am UTC (3 am PST)
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None

# Number of nodes on A3 cluster to be scaled up to
A3_NUM_NODES = 3

SCANNED_CHECKPOINT = "gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_nightly-2025-01-09-05-00-18//8x7b/scanned_ckpt/0/items"
UNSCANNED_CKPT_PATH = "gs://ml-auto-solutions/output/sparsity_diffusion_devx/maxtext/chained_tests_mixtral-8x7b_nightly-2025-01-09-05-00-18//unscanned_ckpt/checkpoints/0/items"


def configure_project_and_cluster(project: str, cluster_name: str, zone: str):
  region = gke.zone_to_region(zone)

  gcloud_command = (
      f"gcloud config set project {project}",
      "sudo chown -R airflow:airflow /home/airflow/composer_kube_config",
      f"gcloud container clusters get-credentials {cluster_name}"
      f"  --region {region}",
  )
  return gcloud_command


def resize_a3_cluster(cluster_name: str, zone: str, num_nodes: int):
  region = gke.zone_to_region(zone)
  node_pool = f"{cluster_name}-np-0"

  gcloud_command = (
      f"gcloud container clusters resize {cluster_name}"
      f"  --quiet --region {region}"
      f"  --node-pool {node_pool}"
      f"  --num-nodes {num_nodes}",
  )
  return gcloud_command


def wait_for_cluster_ready():
  kubectl_command = (
      "kubectl wait --for=condition=Ready nodes --all --timeout=5m",
  )
  return kubectl_command


@task
def scale_up_a3_cluster():
  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(
                    Project.SUPERCOMPUTER_TESTING.value,
                    XpkClusters.GPU_A3_CLUSTER.name,
                    XpkClusters.GPU_A3_CLUSTER.zone,
                )
                + resize_a3_cluster(
                    XpkClusters.GPU_A3_CLUSTER.name,
                    XpkClusters.GPU_A3_CLUSTER.zone,
                    A3_NUM_NODES,
                )
                + wait_for_cluster_ready()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


@task
def scale_down_a3_cluster():
  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(
                    Project.SUPERCOMPUTER_TESTING.value,
                    XpkClusters.GPU_A3_CLUSTER.name,
                    XpkClusters.GPU_A3_CLUSTER.zone,
                )
                + resize_a3_cluster(
                    XpkClusters.GPU_A3_CLUSTER.name,
                    XpkClusters.GPU_A3_CLUSTER.zone,
                    0,
                )
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


def run_maxtext_tests(dag: models.DAG):
  test_name_prefix = "maxtext"

  test_models_gpu = {
      "mixtral-8x7b-1node": (
          f"SCANNED_CHECKPOINT={SCANNED_CHECKPOINT} \
            UNSCANNED_CKPT_PATH={UNSCANNED_CKPT_PATH} \
            bash end_to_end/gpu/test_mixtral.sh",
          1,
      ),
      "mixtral-8x7b-2node": (
          f"SCANNED_CHECKPOINT={SCANNED_CHECKPOINT} \
            UNSCANNED_CKPT_PATH={UNSCANNED_CKPT_PATH} \
            bash end_to_end/gpu/test_mixtral.sh",
          2,
      ),
  }

  for model, (test_script, nnodes) in test_models_gpu.items():
    pinned_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=90,
        test_name=f"{test_name_prefix}-pinned-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.MICHELLE_Y,
    ).run()
    pinned_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=90,
        test_name=f"{test_name_prefix}-pinned-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.MICHELLE_Y,
    ).run()
    stable_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=90,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE_STACK.value,
        test_owner=test_owner.MICHELLE_Y,
    ).run()
    stable_a3plus_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        time_out_in_min=90,
        test_name=f"{test_name_prefix}-stable-{model}",
        run_model_cmds=(test_script,),
        num_slices=nnodes,
        cluster=XpkClusters.GPU_A3PLUS_CLUSTER,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_STABLE_STACK.value,
        test_owner=test_owner.MICHELLE_Y,
    ).run()
    pinned_a3_gpu >> pinned_a3plus_gpu >> stable_a3_gpu >> stable_a3plus_gpu


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
  with TaskGroup(group_id="scale_up", dag=dag) as scale_up:
    scale_up_a3_cluster()

  with TaskGroup(
      group_id="run_tests", dag=dag, prefix_group_id=False
  ) as run_tests:
    run_maxtext_tests(dag)

  with TaskGroup(group_id="scale_down", dag=dag) as scale_down:
    scale_down_a3_cluster()

  scale_up >> run_tests >> scale_down
