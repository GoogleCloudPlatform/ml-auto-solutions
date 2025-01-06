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

import re
import os
from google.cloud import storage
import yaml

PROJECT = "supercomputer-testing"
BUCKET_NAME = "regression-testing-xlml"


# This is required to get auth to access
def git_cookie_authdaemon():
  auth_cmds = (
      "git clone https://gerrit.googlesource.com/gcompute-tools",
      "echo 'trying to run git-cookie-authdaemon'",
      # Check if the daemon is already running
      "if (( $(pgrep -f 'gcompute-tools/git-cookie-authdaemon' | wc -l)>1 )) ; then "  # greater than one because one would be the main job
      "  echo 'git-cookie-authdaemon is already running'; "
      "else "
      "  ./gcompute-tools/git-cookie-authdaemon || echo 'Error running git-cookie-authdaemon'; "  # Run if not running
      "fi",
      "sleep 2",
      "ps aux | grep git-cookie-authdaemon",
  )
  return auth_cmds


def clone_recipes_gob():
  gob_clone_cmds = (
      "echo 'trying to clone GoB repo from outside'",
      "git clone https://ai-hypercomputer-benchmarks.googlesource.com/"
      "reproducible-benchmark-recipes",
  )
  return gob_clone_cmds


def get_aotc_repo():
  gob_clone_cmds = (
      "echo 'trying to clone GoB aotc repo'",
      "git clone https://cmcs-perf-tooling-internal.googlesource.com/"
      "benchmark-automation",
  )
  return gob_clone_cmds


def configure_project_and_cluster(cluster: str, cluster_region: str):
  set_project_command = (
      f"gcloud config set project {PROJECT}",
      "sudo chown -R airflow:airflow /home/airflow/composer_kube_config",
      "gcloud container clusters get-credentials "
      f"{cluster} --region {cluster_region}",
  )
  return set_project_command


def get_gpu_recipe_cmd(hypercomputer, model_id, framework, recipe_repo_root):
  gpu_recipe_cmd = (
      "cd reproducible-benchmark-recipes/projects/gpu-recipes",
      "export RECIPE_ROOT="
      f"{recipe_repo_root}/training/{hypercomputer}/{model_id}/{framework}-pretraining-gke",
      "cd $RECIPE_ROOT",
  )
  return gpu_recipe_cmd


def get_pre_workload_cmds(model_id, framework):
  prepare_workload_cmds = (
      "NOW=$(date +%s)",
      f"export JOB_NAME=gunjanjalori-{model_id}-xlml-$NOW-{framework}",
  )
  return prepare_workload_cmds


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


def helm_apply_cmds(
    framework: str,
    hypercomputer: str,
    config_file,
    recipe_repo_root,
    docker_image,
):
  gcs_cmd = ""
  if hypercomputer == "a3ultra":
    gcs_cmd = f" --set volumes.gcsMounts[0].bucketName={BUCKET_NAME}"
  else:
    gcs_cmd = f" --set workload.gcsBucketForDataCataPath={BUCKET_NAME}"
  helm_cmds = (
      " helm install -f values.yaml "
      "--namespace default "
      "--set namespace=default"
      " --set-file nemo_config"
      f"={config_file}"
      " --set workload.image"
      f"={docker_image} "
      f"{gcs_cmd}"
      f" $JOB_NAME {recipe_repo_root}/src/helm-charts/{hypercomputer}/{framework}-training",
  )
  return helm_cmds


def wait_for_jobs_cmds():
  wait_for_job = (
      "echo 'will wait for jobs to finish'",
      "kubectl wait --for=condition=complete "
      "job/$JOB_NAME --namespace=default --timeout=100m",
  )
  return wait_for_job


def copy_bucket_cmds(recipe_repo_root):
  copy_bucket_contents = (
      "export COMPLETE_JOB_NAME=$(gcloud storage ls "
      f"gs://{BUCKET_NAME}/nemo-experiments/ | grep $JOB_NAME)",
      'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
      f"cd {recipe_repo_root}/src/utils/training_metrics",
      "gcloud storage cp ${COMPLETE_JOB_NAME}"
      "dllogger/rank-0/dllogger.json .",
  )
  return copy_bucket_contents


def get_metrics_cmds(
    batch_size, num_accelerators, precision, model_id, accelertator_type, temdir
):
  cmds = (
      f"METRICS_FILE={temdir}/metrics.txt",
      "python3 process_training_results.py --file"
      f" dllogger.json --batch_size {batch_size} "
      f"--num_accelerators {num_accelerators} "
      f"--precision {precision}  "
      f"--model_type {model_id} "
      f"--accelerator_type {accelertator_type} | "
      "gsutil cp - $METRICS_FILE",
  )
  return cmds


def cleanup_cmds():
  cleanup = (
      "kubectl get pods "
      "--no-headers=true | awk '{print $1}' "
      "| grep $JOB_NAME | xargs kubectl delete pods",
      "helm uninstall $JOB_NAME",
  )
  return cleanup


def get_metrics(temdir):
  file_content = ""
  with open(temdir + "/metrics.txt", "r", encoding="utf-8") as file:
    file_content = file.read()

  # Parse the metrics (adjust based on your file format)
  lines = file_content.splitlines()
  average_step_time = float(lines[0].split(": ")[1])
  tflops_per_accelerator = float(lines[1].split(": ")[1])
  mfu = float(lines[2].split(": ")[1])

  print(f"Average Step Time: {average_step_time}")
  print(f"TFLOPS/Accelerator: {tflops_per_accelerator}")
  print(f"MFU: {mfu}")

  return average_step_time, mfu


def extract_gpus(tmpdir, yaml_file):
  gpus = None
  try:
    yaml_file_path = os.path.join(tmpdir, yaml_file)
    with open(yaml_file_path, "r", encoding="utf-8") as file:
      config = yaml.safe_load(file)
      gpus = config.get("workload", {}).get("gpus")
  except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error: {e}")
    return None

  return gpus


def extract_run_details(root, config_path):
  batch_size = None
  optimizer = None

  try:
    config_path = os.path.join(root, config_path)
    with open(config_path, "r", encoding="utf-8") as file:
      config = yaml.safe_load(file)
      batch_size = config.get("model", {}).get("global_batch_size")
      precision = config.get("trainer", {}).get("precision")
      optimizer = config.get("model", {}).get("optim", {}).get("name")
      seq_length = config.get("model", {}).get("data", {}).get("seq_length")
      max_steps = config.get("trainer", {}).get("max_steps")
  except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error: {e}")
    return None

  return batch_size, optimizer, precision, seq_length, max_steps


def get_accelerator_type(hypercomputer: str):
  if hypercomputer == "a3ultra":
    return "h200"
  elif hypercomputer == "a3mega":
    return "h100"
