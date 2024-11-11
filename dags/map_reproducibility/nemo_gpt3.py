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
      "export BUCKET_NAME=gunjanjalori-testing-xlml"
      # "NAMESPACE=xlml-regression-tests-gpt3",
      "JOB_NAME=gpt3-xlml-$NOW-175b-nemo"
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
      # "helm plugin install https://github.com/thomastaylor312/helm-namespace",
      # TODO(gunjanjalori): add a check for if already exists.
      "kubectl get pods",
      "kubectl config view | grep namespace",
      # "kubectl create namespace xlml-regression-tests-gpt3",
      "kubectl config set-context helm --namespace=default",
      # "kubectl config set-context helm --namespace xlml-regression-tests-gpt3",
      # "sudo apt install nfs-common",
      # "kubectl apply -f https://raw.githubusercontent.com/appscode/"
      # "third-party-tools/master/storage/nfs/artifacts/nfs-server.yaml",
  )

  helm_cmds = (
    "echo 'can I create?'",
    "kubectl auth can-i create configmaps",
    "echo 'can I create namespaces?'",
    "kubectl auth can-i create namespaces",
    "CONFIG_FILE=$REPO_ROOT/src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml",
    # "kubectl create configmap $JOB_NAME --from-file=$CONFIG_FILE",
    "echo $CONFIG_FILE",
    "echo 'mid'",
    "echo $REPO_ROOT/src/frameworks/nemo-configs/gpt3-175b-256gpus-fp8.yaml",
    " helm install -f values.yaml "
    "--namespace default "
    # "--create-namespace "
    "--set namespace=default"
    " --set-file nemo_config"
    "=$CONFIG_FILE"
    " --set workload.image"
    "=us-central1-docker.pkg.dev/"
    "supercomputer-testing/gunjanjalori/nemo_test/nemo_workload:24.07"
    " --set workload.gcsBucketForDataCataPath=reproducibility-demo"
    " $JOB_NAME $REPO_ROOT/src/helm-charts/nemo-training",
    "kubectl get pods | grep $JOB_NAME",
    # "kubectl create configmap $JOB_NAME --from-file=$CONFIG_FILE",
  )

  # wait_for_job = (
  #   "echo 'pwd'",
  #   "pwd",
  #   "ls",
  #   "echo 'ls'",
  #   "kubectl wait --for=condition=complete job/$JOB_NAME",

  #   # wait till bucket has the run
  #   "while true; do"
  #   "OBJECTS=$(gsutil ls gs://$BUCKET_NAME)/nemo-experiments/"
  #   "JOB_NAME"
  #   "if [[ -n "$OBJECTS" ]]; then"
  #   "echo 'Bucket $BUCKET_NAME has objects. Proceeding...'"
  #   "break"
  #   "else"
  #   "echo 'Bucket $BUCKET_NAME is empty. Waiting...'"
  #   "sleep 360" # Wait for 360 seconds before checking again
  #   "fi"
  #   "done"
  # )


  get_results = (
    # copy logs from bucket
    # "CURRENT_DIR=$(pwd)"
    # "FILES=$(ls -U $CURRENT_DIR | head -n 2)"
    "CURRENT_DIR=$(pwd)",
    "FIRST_TWO_FOLDERS=$(echo $CURRENT_DIR | cut -d/ -f2,3)",
    "echo $FIRST_TWO_FOLDERS",
    "JOB_NAME=gunjanjalori-llama-3-70b-128-nemo-1729727194-fve3",
    "COMPLETE_JOB_NAME=$(gcloud storage ls gs://reproducibility-demo/nemo-experiments | grep $JOB_NAME)",
    "echo $COMPLETE_JOB_NAME",
    "echo 'copying'",
    "gcloud storage cp $COMPLETE_JOB_NAMEdllogger/rank-0/dllogger.json "
    "/$FIRST_TWO_FOLDERS/",

    # get metrics
    "cd $REPO_ROOT/src/utils/training_metrics",
    "python3 process_training_results.py --file"
    " /$FIRST_TWO_FOLDERS/dllogger.json --batch_size 2048 "
    "--num_accelerators 256 "
    "--precision fp8  "
    "--model_type gpt3-175b "
    "--accelerator_type h100 ",
  )

  # get_results = (
  #   # copy logs from bucket
  #   "CURRENT_DIR=$(pwd)",
  #   "FIRST_TWO_FOLDERS=$(echo $CURRENT_DIR | cut -d/ -f2,3)",
  #   "COMPLETE_JOB_NAME=$(ls -d | grep $JOB_NAME)",
  #   "gcloud storage cp gs://$BUCKET_NAME/nemo-experiments/"
  #   "$COMPLETE_JOB_NAME/dllogger/rank-0/dllogger.json "
  #   "/$FIRST_TWO_FOLDERS/",

  #   # get metrics
  #   "cd $REPO_ROOT/src/utils/training_metrics"
  #   "python3 process_training_results.py --file"
  #   " /$FIRST_TWO_FOLDERS/dllogger.json --batch_size 2048"
  #   "--num_accelerators 256 "
  #   "--precision fp8  "
  #   "--model_type gpt3-175b "
  #   "--accelerator_type h100"
  # )

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
        # + wait_for_job
        # + get_results
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
