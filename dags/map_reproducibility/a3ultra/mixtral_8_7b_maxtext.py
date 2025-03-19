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

"""DAGs to run Aotc reproducibility benchmarks."""

import datetime
import os
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags import composer_env
from dags.map_reproducibility.utils.common_utils import configure_project_and_cluster
from dags.map_reproducibility.utils.common_utils import install_helm_cmds
from dags.map_reproducibility.utils.common_utils import namespace_cmds
from dags.map_reproducibility.utils.common_utils import wait_for_jobs_cmds
from dags.map_reproducibility.utils.common_utils import cleanup_cmds
from dags.map_reproducibility.utils.common_utils import git_cookie_authdaemon
from dags.map_reproducibility.utils.common_utils import clone_recipes_gob
from dags.map_reproducibility.utils.common_utils import helm_apply_cmds
from dags.map_reproducibility.utils.common_utils import get_bq_writer_repo
from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from dags.map_reproducibility.utils.common_utils import extract_gpus
from dags.map_reproducibility.utils.common_utils import get_pre_workload_cmds
from dags.map_reproducibility.utils.common_utils import get_gpu_recipe_cmd
from dags.map_reproducibility.utils.common_utils import get_bq_writer_path
from dags.map_reproducibility.utils.common_utils import get_recipe_repo_path
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.common_utils import get_scheduled_time
from dags.map_reproducibility.utils.common_utils import get_docker_image
from dags.map_reproducibility.utils.common_utils import calculate_maxtext_metrics
from dags.map_reproducibility.utils.common_utils import copy_bucket_cmds_maxtext


MODEL_ID = "mixtral-8x7b"
METRICS_MODEL_ID = "mixtral-7b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3ultra"
FRAMEWORK = "maxtext"
VALUE_YAML_PATH = (
    f"training/{HYPERCOMPUTER}/{MODEL_ID}/maxtext-pretraining-gke/values.yaml"
)

SCHEDULED_TIME = (
    get_scheduled_time(HYPERCOMPUTER, MODEL_ID, FRAMEWORK)
    if composer_env.is_prod_env()
    else None
)

SOFTWARE_ID = "jax_maxtext"
CLUSTER, CLUSTER_REGION = get_cluster(HYPERCOMPUTER)
IMAGE_VERSION = "maxtext-nightly"
DOCKER_IMAGE = get_docker_image(HYPERCOMPUTER, FRAMEWORK)
KUEUE_NAME = "a3-ultra"

OPTIMIZER = "adam"
SEQUENCE_LENGTH = 2048
NUM_STEPS = 30
BATCH_SIZE_PER_DEVICE = 5


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
    config_yaml_path = f"src/frameworks/{HYPERCOMPUTER}/maxtext-configs/{MODEL_ID}-{num_gpus}gpus-a3u-{PRECISION}.yaml"
    full_config_yaml_path = os.path.join(recipe_repo_root, config_yaml_path)

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
                )
                + wait_for_jobs_cmds()
                + copy_bucket_cmds_maxtext(
                    tmpdir, recipe_repo_root=recipe_repo_root
                )
                + cleanup_cmds()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    log_location = os.path.join(tmpdir, "tflog/metrics")

    mfu, step_time = calculate_maxtext_metrics(log_location, HYPERCOMPUTER)

    print(f"mfu: {mfu}")
    print(f"step_time: {step_time}")

    write_run(
        model_id=MODEL_ID,
        hardware_id=HYPERCOMPUTER,
        software_id=SOFTWARE_ID,
        number_of_nodes=num_gpus / 8,
        number_of_chips=num_gpus,
        container_image_name=IMAGE_VERSION,
        global_batch_size=BATCH_SIZE_PER_DEVICE * num_gpus,
        precision=PRECISION,
        optimizer=OPTIMIZER,
        seq_length=SEQUENCE_LENGTH,
        median_step_time=step_time,
        e2e_time=step_time * NUM_STEPS,
        number_of_steps=NUM_STEPS,
        mfu=mfu,
        tokens_per_second=1,
        writer_path=bq_writer_repo_root,
        topology="",
        comment="Regression tests",
        is_test=False,
    )


with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}",
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
