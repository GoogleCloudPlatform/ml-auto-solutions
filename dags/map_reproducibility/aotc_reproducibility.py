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

"Bash helper commands for AOTC artifacts"

import os


def set_variables_cmds():
  set_variables = (
      "export PROJECT=supercomputer-testing",
      "export CLUSTER=a3plus-benchmark",
      "export CLUSTER_REGION=australia-southeast1",
      "NOW=$(date +%s)",
      "export BUCKET_NAME=regression-testing-xlml",
  )
  return set_variables


def configure_project_and_cluster():
  set_project_command = (
      "gcloud config set project $PROJECT",
      "sudo chown -R airflow:airflow /home/airflow/composer_kube_config",
      "gcloud container clusters get-credentials "
      "$CLUSTER --region $CLUSTER_REGION",
  )
  return set_project_command


# This is required to get auth to access
# internal GoB repo
def git_cookie_authdaemon():
  auth_cmds = (
      "git clone https://gerrit.googlesource.com/gcompute-tools",
      "echo 'trying to run git-cookie-authdaemon'",
      "./gcompute-tools/git-cookie-authdaemon",
  )
  return auth_cmds


def clone_gob():
  gob_clone_cmds = (
      "echo 'trying to clone GoB repo from outside'",
      "git clone https://ai-hypercomputer-benchmarks.googlesource.com/"
      "reproducible-benchmark-recipes",
      "cd reproducible-benchmark-recipes/projects",
      "cd gpu-recipes",
  )
  return gob_clone_cmds


def install_helm_cmds():
  install_helm_cmd = (
      "curl -fsSL -o get_helm.sh "
      "https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3",
      "chmod 700 get_helm.sh",
      "./get_helm.sh",
  )
  return install_helm_cmd


# By default the composer environment overwrites the
# namespaces to airflow namespaces.
# In order to prevent that it is necessary explicitly
# change the namespace to default.
def namespace_cmds():
  namespace = (
      "kubectl config view | grep namespace",
      "kubectl config set-context --current --namespace=default",
      "kubectl config set-context helm --namespace=default",
  )
  return namespace


def helm_install_cmds():
  helm_cmds = (
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
  return helm_cmds


def wait_for_jobs_cmds():
  wait_for_job = (
      "echo 'will wait for job to start running'",
      "kubectl wait --for=condition=running job/$JOB_NAME"
      " --namespace=default --timeout=10m",
      "echo 'will wait for jobs to finish'",
      "kubectl wait --for=condition=complete "
      "job/$JOB_NAME --namespace=default --timeout=100m",
  )
  return wait_for_job


def copy_bucket_cmds():
  copy_bucket_contents = (
      "COMPLETE_JOB_NAME=$(gcloud storage ls "
      "gs://$BUCKET_NAME/nemo-experiments/ | grep $JOB_NAME)",
      "echo 'copying from' ",
      "echo $COMPLETE_JOB_NAME",
      "cd $REPO_ROOT/src/utils/training_metrics",
      "gcloud storage cp ${COMPLETE_JOB_NAME}"
      "dllogger/rank-0/dllogger.json .",
  )
  return copy_bucket_contents


def get_metrics_cmds():
  # TODO(gunjanj007): get these parameters from the recipe
  get_metrics = (
      "python3 process_training_results.py --file"
      " dllogger.json --batch_size 2048 "
      "--num_accelerators 256 "
      "--precision fp8  "
      "--model_type gpt3-175b "
      "--accelerator_type h100 ",
  )
  return get_metrics


def cleanup_cmds():
  cleanup = (
      "kubectl get pods "
      "--no-headers=true | awk '{print $1}' "
      "| grep $JOB_NAME |  xargs kubectl delete pods",
      "helm uninstall $JOB_NAME",
  )
  return cleanup
