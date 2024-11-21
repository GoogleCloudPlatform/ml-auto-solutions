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
from airflow import models
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags import composer_env
from dags.map_reproducibility.aotc_reproducibility import get_metrics_cmds
from dags.map_reproducibility.aotc_reproducibility import set_variables_cmds
from dags.map_reproducibility.aotc_reproducibility import configure_project_and_cluster
from dags.map_reproducibility.aotc_reproducibility import install_helm_cmds
from dags.map_reproducibility.aotc_reproducibility import namespace_cmds
from dags.map_reproducibility.aotc_reproducibility import wait_for_jobs_cmds
from dags.map_reproducibility.aotc_reproducibility import copy_bucket_cmds
from dags.map_reproducibility.aotc_reproducibility import cleanup_cmds
from dags.map_reproducibility.aotc_reproducibility import git_cookie_authdaemon
from dags.map_reproducibility.aotc_reproducibility import clone_gob

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

    helm_cmds = (
        "CONFIG_FILE=$REPO_ROOT/src/frameworks"
        "/nemo-configs/gpt3-175b-256gpus-fp8.yaml",
        "export JOB_NAME=gpt3-xlml-$NOW-175b-nemo",
        " helm install -f values.yaml "
        "--namespace default "
        "--set namespace=default"
        " --set-file nemo_config"
        "=$CONFIG_FILE"
        " --set workload.image"
        "=us-central1-docker.pkg.dev/"
        "supercomputer-testing/gunjanjalori/nemo_test/nemo_workload:24.07"
        " --set workload.gcsBucketForDataCataPath=$BUCKET_NAME"
        " $JOB_NAME $REPO_ROOT/src/helm-charts/nemo-training",
    )

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
                + helm_cmds
                + wait_for_jobs_cmds()
                + copy_bucket_cmds()
                + get_metrics_cmds()
                + cleanup_cmds()
            ),
        ],
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


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
