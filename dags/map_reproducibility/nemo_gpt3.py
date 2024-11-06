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


@task
def run_aotc_workload():

    set_variables = (
        "export PROJECT=supercomputer-testing",
        "export CLUSTER=a3plus-benchmark",
        "export CLUSTER_REGION=australia-southeast1",
        "NOW=$(date +%s)",
        "NAMESPACE=xlml-regression-tests-gpt3",
    )

    set_project_command = (
        "gcloud config set project $PROJECT",
        "sudo chown -R airflow:airflow /home/airflow/composer_kube_config",
        "gcloud container clusters get-credentials "
        "$CLUSTER --region $CLUSTER_REGION",
    )

    gpu_recipe_cmd = (
        "git clone https://github.com/ai-hypercomputer/gpu-recipes.git",
        "cd gpu-recipes",
        "export REPO_ROOT=`git rev-parse --show-toplevel`",
        "export RECIPE_ROOT="
        "$REPO_ROOT/training/a3mega/gpt3-175b/nemo-pretraining-gke",
        "cd $RECIPE_ROOT",
    )

    install_helm_cmd = (
        "curl -fsSL -o get_helm.sh "
        "https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3",
        "chmod 700 get_helm.sh",
        "./get_helm.sh",
    )

    namespace_cmds = (
        "helm plugin install https://github.com/thomastaylor312/helm-namespace",
        "kubectl config set-context helm --namespace xlml-regression-tests-gpt3",
        "sudo apt install -y nfs-common",
        "kubectl apply -f https://raw.githubusercontent.com/appscode/"
        "third-party-tools/master/storage/nfs/artifacts/nfs-server.yaml",
    )

    helm_cmds = (
        " helm namespace install -f values.yaml "
        " --set-file nemo_config"
        "=$REPO_ROOT/src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml"
        " --set workload.image"
        "=us-central1-docker.pkg.dev/"
        "supercomputer-testing/gunjanjalori/nemo_test/nemo_workload:24.07"
        " --set workload.gcsBucketForDataCataPath=gunjanjalori-testing-xlml"
        " gpt3-xlml-$NOW-175b-nemo $REPO_ROOT/src/helm-charts/nemo-training",
    )

    hook = SubprocessHook()
    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                set_variables
                + set_project_command
                + gpu_recipe_cmd
                + install_helm_cmd
                + namespace_cmds
                + helm_cmds
            ),
        ],
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


with models.DAG(
    dag_id="reproducibility_nemo_gpt3_nighly_dag",
    schedule=None,
    tags=["simple", "aotc", "nightly", "reproducibility", "experimental", "xlml"],
    start_date=datetime.datetime(2024, 10, 22),
    catchup=False,
) as dag:
    run_aotc_workload()
