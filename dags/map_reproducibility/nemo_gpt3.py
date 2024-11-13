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
      "export BUCKET_NAME=gunjanjalori-testing-xlml",
      "export JOB_NAME=gpt3-xlml-$NOW-175b-nemo",
  )

  set_project_command = (
      "gcloud config set project $PROJECT",
      "sudo chown -R airflow:airflow /home/airflow/composer_kube_config",
      "gcloud container clusters get-credentials "
      "$CLUSTER --region $CLUSTER_REGION",
  )

  gpu_recipe_cmd = (
      "git clone https://github.com/gunjanj007/gpu-recipes.git",
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
      "kubectl get pods",
      "kubectl config view | grep namespace",
      "kubectl config set-context --current --namespace=default",
      "kubectl config set-context heml --namespace=default",
      "kubectl get pods --namespace=defaults"
  )

  helm_cmds = (
    "CONFIG_FILE=$REPO_ROOT/src/frameworks"
    "/nemo-configs/gpt3-175b-256gpus-fp8.yaml",
    " helm install -f values.yaml "
    "--namespace default "
    "--set namespace=default"
    " --set-file nemo_config"
    "=$CONFIG_FILE"
    " --set workload.image"
    "=us-central1-docker.pkg.dev/"
    "supercomputer-testing/gunjanjalori/nemo_test/nemo_workload:24.07"
    " --set workload.gcsBucketForDataCataPath=gunjanjalori-testing-xlml "
    " --set workload.arguments='{trainer.max_steps=5}' "
    " $JOB_NAME $REPO_ROOT/src/helm-charts/nemo-training",
  )

  wait_for_job = (
    "echo 'will wait for job to start running'",
    "kubectl wait --for=condition=running job/$JOB_NAME"
    " --namespace=default --timeout=10m",
    "echo 'will wait for jobs to finish'",
    "kubectl wait --for=condition=complete "
    "job/$JOB_NAME --namespace=default --timeout=100m",
  )

  copy_bucket_contents = (
    "COMPLETE_JOB_NAME=$(gcloud storage ls "
    "gs://$BUCKET_NAME/nemo-experiments/ | grep $JOB_NAME)",
    "echo 'copying from $COMPLETE_JOB_NAME'",
    "cd $REPO_ROOT/src/utils/training_metrics",
    "gcloud storage cp ${COMPLETE_JOB_NAME}"
    "dllogger/rank-0/dllogger.json .",
  )

  get_metrics = (
    "python3 process_training_results.py --file"
    " dllogger.json --batch_size 2048 "
    "--num_accelerators 256 "
    "--precision fp8  "
    "--model_type gpt3-175b "
    "--accelerator_type h100 ",
  )

  cleanup = (
     "kubectl get pods "
     "--no-headers=true | awk '{print $1}' "
     "| grep $JOB_NAME |  xargs kubectl delete pods",
     "helm uninstall $JOB_NAME",
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
        + wait_for_job
        # + wait_for_bucket
        + copy_bucket_contents
        + get_metrics
        + cleanup
      ),
    ],
  )
  assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


with models.DAG(
    dag_id="reproducibility_nemo_gpt3_nighly_dag",
    schedule=None,
    tags=[
        "simple",
        "aotc",
        "nightly",
        "reproducibility",
        "experimental",
        "xlml"],
    start_date=datetime.datetime(2024, 10, 22),
    catchup=False,
) as dag:
  run_aotc_workload()
