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

"""Workload functions for AOTC reproducibility benchmarks."""

import os
import tempfile

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags.map_reproducibility.utils.common_utils import configure_project_and_cluster
from dags.map_reproducibility.utils.common_utils import install_helm_cmds
from dags.map_reproducibility.utils.common_utils import namespace_cmds
from dags.map_reproducibility.utils.common_utils import wait_for_jobs_cmds
from dags.map_reproducibility.utils.common_utils import cleanup_cmds
from dags.map_reproducibility.utils.common_utils import git_cookie_authdaemon
from dags.map_reproducibility.utils.common_utils import clone_recipes_gob, clone_internal_recipes_gob
from dags.map_reproducibility.utils.common_utils import helm_apply_cmds_internal_run
from dags.map_reproducibility.utils.common_utils import get_bq_writer_repo
from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from dags.map_reproducibility.utils.common_utils import get_internal_pre_workload_cmds
from dags.map_reproducibility.utils.common_utils import get_gpu_recipe_cmd
from dags.map_reproducibility.utils.common_utils import get_bq_writer_path
from dags.map_reproducibility.utils.common_utils import get_recipe_repo_path, get_internal_recipe_repo_path
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.common_utils import get_internal_docker_image
from dags.map_reproducibility.utils.common_utils import calculate_maxtext_metrics
from dags.map_reproducibility.utils.common_utils import copy_bucket_cmds_maxtext
from dags.map_reproducibility.utils.common_utils import parse_internal_config_filename
from dags.map_reproducibility.utils.common_utils import parse_internal_config_content
from dags.map_reproducibility.utils.constants import Optimizer, KUEUE_NAME, NUM_STEPS


@task
def run_internal_aotc_workload(relative_config_yaml_path, test_run=False):
  """Runs the AOTC workload benchmark.

  Args:
    relative_config_yaml_path: Path to the config YAML relative to the repo root
  """
  # Parse config from filename
  config_yaml_name = relative_config_yaml_path.rsplit("/", maxsplit=1)[
      -1
  ].replace(".yaml", "")
  config = parse_internal_config_filename(config_yaml_name)

  # Get derived configuration
  cluster, cluster_region = get_cluster(config.HYPERCOMPUTER)
  docker_image = get_internal_docker_image(
      config.HYPERCOMPUTER, config.FRAMEWORK
  )
  values_name = f"{config.HYPERCOMPUTER}_{config.FRAMEWORK}_values"

  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                git_cookie_authdaemon()
                + clone_recipes_gob()
                + (() if test_run else clone_internal_recipes_gob())
                + get_bq_writer_repo()
            ),
        ],
        cwd=tmpdir,
    )

    recipe_repo_root = get_recipe_repo_path(tmpdir)
    bq_writer_repo_root = get_bq_writer_path(tmpdir)

    # Update paths now that we have the repo paths
    internal_recipe_repo_root = (
        "/home/airflow/gcs/dags/dags/map_reproducibility"
    )
    if not test_run:
      internal_recipe_repo_root = get_internal_recipe_repo_path(tmpdir)
    values_file_path = f"{internal_recipe_repo_root}/values/{values_name}.yaml"
    full_config_yaml_path = (
        f"{internal_recipe_repo_root}/{relative_config_yaml_path}"
    )
    print(f"values_file_path is {values_file_path}")
    print(f"full_config_yaml_path is {full_config_yaml_path}")

    # Parse the config content now that we have the file path
    more_config = parse_internal_config_content(full_config_yaml_path)

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(cluster, cluster_region)
                + get_gpu_recipe_cmd(
                    config.HYPERCOMPUTER,
                    config.MODEL_ID,
                    config.FRAMEWORK,
                    recipe_repo_root,
                )
                + install_helm_cmds()
                + namespace_cmds()
                + get_internal_pre_workload_cmds(
                    config.HELM_NAME_MODEL_ID,
                    config.FRAMEWORK,
                    config.IS_PGLE,
                )
                + helm_apply_cmds_internal_run(
                    config.FRAMEWORK,
                    config.HYPERCOMPUTER,
                    full_config_yaml_path,
                    recipe_repo_root,
                    values_file_path,
                    docker_image,
                    cluster_name=cluster,
                    kueue_name=KUEUE_NAME,
                    additional_cmds=f" --set workload.gpus={config.NUM_GPUS} ",
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

    mfu, step_time = calculate_maxtext_metrics(
        log_location, config.HYPERCOMPUTER
    )

    print(f"mfu: {mfu}")
    print(f"step_time: {step_time}")

    write_run(
        model_id=config.HELM_NAME_MODEL_ID,
        hardware_id=config.HYPERCOMPUTER,
        software_id=config.SOFTWARE_ID,
        number_of_nodes=config.NUM_GPUS / 8,
        number_of_chips=config.NUM_GPUS,
        container_image_name=docker_image,
        global_batch_size=more_config.BATCH_SIZE_PER_DEVICE * config.NUM_GPUS,
        precision=config.PRECISION,
        optimizer=Optimizer.ADAM,
        seq_length=more_config.SEQUENCE_LENGTH,
        median_step_time=step_time,
        e2e_time=step_time * NUM_STEPS,
        number_of_steps=NUM_STEPS,
        mfu=mfu,
        tokens_per_second=1,
        writer_path=bq_writer_repo_root,
        topology="",
        comment="internal recipes regression tests",
        is_test=False,
    )
