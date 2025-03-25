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

"""DAGs to run hypercomputer recipes"""

import datetime
import sys
import os
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags import composer_env
from dags.map_reproducibility.utils.common_utils import get_nemo_metrics_cmds
from dags.map_reproducibility.utils.common_utils import configure_project_and_cluster
from dags.map_reproducibility.utils.common_utils import install_helm_cmds
from dags.map_reproducibility.utils.common_utils import namespace_cmds
from dags.map_reproducibility.utils.common_utils import wait_for_jobs_cmds
from dags.map_reproducibility.utils.common_utils import copy_bucket_cmds_nemo
from dags.map_reproducibility.utils.common_utils import cleanup_cmds
from dags.map_reproducibility.utils.common_utils import git_cookie_authdaemon
from dags.map_reproducibility.utils.common_utils import clone_recipes_gob
from dags.map_reproducibility.utils.common_utils import helm_apply_cmds
from dags.map_reproducibility.utils.common_utils import get_nemo_metrics
from dags.map_reproducibility.utils.common_utils import get_bq_writer_repo
from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from dags.map_reproducibility.utils.common_utils import extract_run_details
from dags.map_reproducibility.utils.common_utils import extract_gpus
from dags.map_reproducibility.utils.common_utils import get_accelerator_type
from dags.map_reproducibility.utils.common_utils import get_pre_workload_cmds
from dags.map_reproducibility.utils.common_utils import get_gpu_recipe_cmd
from dags.map_reproducibility.utils.common_utils import get_bq_writer_path
from dags.map_reproducibility.utils.common_utils import get_recipe_repo_path
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.common_utils import get_two_node_cmds
from dags.map_reproducibility.utils.common_utils import get_docker_image

MODEL_ID = "mixtral-8x7b"
METRICS_MODEL_ID = "mixtral-7b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3ultra"
FRAMEWORK = "nemo"

SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None

VALUE_YAML_PATH = (
    f"training/{HYPERCOMPUTER}/{MODEL_ID}/nemo-pretraining-gke/values.yaml"
)
CLUSTER, CLUSTER_REGION = get_cluster(HYPERCOMPUTER)
SOFTWARE_ID = "pytorch_nemo"
IMAGE_VERSION = "nemo24.07"
DOCKER_IMAGE = get_docker_image(HYPERCOMPUTER, FRAMEWORK)
KUEUE_NAME = "a3-ultra"


@task
def run_aotc_workload():
  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                git_cookie_authdaemon()
                + clone_recipes_gob()
                + get_bq_writer_repo()
            ),
        ],
        cwd=tmpdir,
    )

    recipe_repo_root = get_recipe_repo_path(tmpdir)
    bq_writer_repo_root = get_bq_writer_path(tmpdir)

    num_gpus = extract_gpus(recipe_repo_root, VALUE_YAML_PATH)
    num_gpus_temp = 256
    config_yaml_path = f"src/frameworks/{HYPERCOMPUTER}/nemo-configs/{MODEL_ID}-{num_gpus_temp}gpus-a3u-{PRECISION}.yaml"
    full_config_yaml_path = os.path.join(recipe_repo_root, config_yaml_path)

    (
        global_batch_size,
        optimizer,
        precision,
        seq_length,
        max_steps,
    ) = extract_run_details(recipe_repo_root, config_yaml_path)

    accelerator_type = get_accelerator_type(HYPERCOMPUTER)
    print(
        f"batch size: {global_batch_size}, num gpus: {num_gpus},  precision: {precision}, seq length: {seq_length}, max steps: {max_steps}"
    )

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(CLUSTER, CLUSTER_REGION)
                + get_gpu_recipe_cmd(
                    HYPERCOMPUTER, MODEL_ID, FRAMEWORK, recipe_repo_root
                )
                + install_helm_cmds()
                + namespace_cmds()
                + get_pre_workload_cmds(MODEL_ID, FRAMEWORK)
                + helm_apply_cmds(
                    FRAMEWORK,
                    HYPERCOMPUTER,
                    full_config_yaml_path,
                    recipe_repo_root,
                    DOCKER_IMAGE,
                    cluster_name=CLUSTER,
                    kueue_name=KUEUE_NAME,
                    additional_cmds=get_two_node_cmds(),
                )
                + wait_for_jobs_cmds()
                + copy_bucket_cmds_nemo(
                    recipe_repo_root,
                    hypercomputer=HYPERCOMPUTER,
                )
                + get_nemo_metrics_cmds(
                    global_batch_size,
                    num_gpus,
                    PRECISION,
                    METRICS_MODEL_ID,
                    accelerator_type,
                    tmpdir,
                    freq="daily",
                )
                + cleanup_cmds()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    average_step_time, mfu = get_nemo_metrics(tmpdir)

    write_run(
        model_id=MODEL_ID,
        hardware_id=HYPERCOMPUTER,
        software_id=SOFTWARE_ID,
        number_of_nodes=num_gpus / 8,
        number_of_chips=num_gpus,
        container_image_name=IMAGE_VERSION,
        global_batch_size=global_batch_size,
        precision=precision,
        optimizer=optimizer,
        seq_length=seq_length,
        median_step_time=average_step_time,
        e2e_time=0,
        number_of_steps=1,
        mfu=mfu,
        tokens_per_second=1,
        writer_path=bq_writer_repo_root,
        comment="Two node and single step tests",
        is_test=False,
    )


with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_two_node_{FRAMEWORK}",
    schedule=SCHEDULED_TIME,
    tags=[
        "reproducibility",
        "experimental",
        "xlml",
        "regressiontests",
        "a3ultra",
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  run_aotc_workload()
