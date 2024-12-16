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
import sys
import os
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
import subprocess
from dags import composer_env
from dags.map_reproducibility.utils import get_metrics_cmds
from dags.map_reproducibility.utils import set_variables_cmds
from dags.map_reproducibility.utils import configure_project_and_cluster
from dags.map_reproducibility.utils import install_helm_cmds
from dags.map_reproducibility.utils import namespace_cmds
from dags.map_reproducibility.utils import wait_for_jobs_cmds
from dags.map_reproducibility.utils import copy_bucket_cmds
from dags.map_reproducibility.utils import cleanup_cmds
from dags.map_reproducibility.utils import git_cookie_authdaemon
from dags.map_reproducibility.utils import clone_gob
from dags.map_reproducibility.utils import helm_install_cmds
from dags.map_reproducibility.utils import get_metrics
from dags.map_reproducibility.utils import get_aotc_repo
from dags.map_reproducibility.utils import extract_python_path
from dags.map_reproducibility.benchmarkdb_utils import write_run
from dags.map_reproducibility.utils import extract_gpus


# Run once a day at 2 pm UTC (6 am PST)
SCHEDULED_TIME = "0 14 * * *" if composer_env.is_prod_env() else None


@task
def run_aotc_workload():
  gpu_recipe_cmd = (
      "export REPO_ROOT=`pwd`",
      "export RECIPE_ROOT="
      "$REPO_ROOT/training/a3mega/gpt3-175b/nemo-pretraining-gke",
      "cd $RECIPE_ROOT",
  )

  workload_cmds = (
      "CONFIG_FILE=$REPO_ROOT/src/frameworks/"
      "a3mega/nemo-configs/gpt3-175b-256gpus-fp8.yaml",
      "export JOB_NAME=gpt3-xlml-$NOW-175b-nemo",
  )

  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()
    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                set_variables_cmds()
                + configure_project_and_cluster()
                + git_cookie_authdaemon()
                + clone_gob()
                + gpu_recipe_cmd
                + install_helm_cmds()
                + namespace_cmds()
                + workload_cmds
                # + helm_install_cmds()
                # + wait_for_jobs_cmds()
                + copy_bucket_cmds()
                + get_metrics_cmds()
                # + cleanup_cmds()
                + get_aotc_repo()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    # # Extract COMPLETE_JOB_NAME from the output
    # bucket_name, file_name, python_path = extract_bucket_file_name(
    #     result.output
    # )

    # # Extract PYTHONPATH from the output
    # python_path = extract_python_path(result.output)
    python_base_path, python_path_to_bq_writer = extract_python_path(
        result.output.splitlines()[-1]
    )
    print(f"Base path in python: {python_base_path}")
    print(f"python to bq: {python_path_to_bq_writer}")

    yaml_file_path = "reproducible-benchmark-recipes/projects/gpu-recipes/training/a3mega/gpt3-175b/nemo-pretraining-gke/values.yaml"
    config_yaml_path = "reproducible-benchmark-recipes/projects/gpu-recipes/src/frameworks/a3mega/nemo-configs/gpt3-175b-256gpus-fp8.yaml"
    (
        number_of_nodes,
        global_batch_size,
        optimizer,
        precision,
        seq_length,
        max_steps,
    ) = extract_gpus(tmpdir, yaml_file_path, config_yaml_path)
    print(
        f"batch size: {global_batch_size}, number of nodes: {number_of_nodes}"
    )
    average_step_time, mfu = get_metrics(
        python_base_path
    )
    model_id = "gpt3-175b"
    hardware_id = "a3mega"
    software_id = "pytorch_nemo"
    number_of_chips = number_of_nodes * 8

    write_run(
        model_id=model_id,
        hardware_id=hardware_id,
        software_id=software_id,
        number_of_nodes=number_of_nodes,
        number_of_chips=number_of_chips,
        container_image_name="sample_docker",
        global_batch_size=global_batch_size,
        precision=precision,
        optimizer=optimizer,
        seq_length=seq_length,
        median_step_time=average_step_time,
        e2e_time=0,
        number_of_steps=max_steps,
        mfu=mfu,
        tokens_per_second=1,
        writer_path=python_path_to_bq_writer,
        topology="2X2",
        comment="Regression tests",
        is_test=True,
    )


with models.DAG(
    dag_id="reproducibility_nemo_gpt3_nighly_dag",
    schedule=SCHEDULED_TIME,
    tags=[
        "simple",
        "aotc",
        "nightly",
        "reproducibility",
        "experimental",
        "xlml",
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  run_aotc_workload()
