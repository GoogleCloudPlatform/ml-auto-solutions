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
from xlml.utils import metric
from xlml.apis import metric_config

PROJECT = "supercomputer-testing"
BUCKET_NAME = "regression-testing-xlml"

MAX_TFLOP = {"a3ultra": 989, "a3mega": 989}


# This is required to get auth to access
def git_cookie_authdaemon():
  auth_cmds = (
      "git clone https://gerrit.googlesource.com/gcompute-tools",
      "echo 'trying to run git-cookie-authdaemon'",
      # Check if the daemon is already running
      "if (( $(ps aux | grep git-cookie-authdaemon | grep -v -E 'airflow|grep|bash' | wc -l)>0 )) ; then "  # greater than one because one would be the main job
      " echo 'git-cookie-authdaemon is already running' ",
      "else "
      " (./gcompute-tools/git-cookie-authdaemon >/dev/null 2>&1 &) ",  # Run if not running
      "sleep 4",
      "fi",
      "ps aux | grep git-cookie-authdaemon | grep -v -E 'airflow|grep|bash'",
  )
  return auth_cmds


def clone_recipes_gob():
  gob_clone_cmds = (
      "echo 'trying to clone GoB repo from outside'",
      "git clone https://ai-hypercomputer-benchmarks.googlesource.com/"
      "reproducible-benchmark-recipes",
  )
  return gob_clone_cmds


def get_bq_writer_repo():
  gob_clone_cmds = (
      "echo 'trying to clone GoB bq writer repo'",
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
      f"export JOB_NAME=imo-team-regr-test-{model_id}-$NOW-{framework}",
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
    aotc: bool = False,
    cluster_name: str = "a3plus-benchmark",
    kueue_name: str = "a3-ultra",
    additional_cmds: str = "",
):
  gcs_cmd = ""
  if hypercomputer == "a3ultra":
    if framework != "maxtext":
      gcs_cmd = f" --set queue={kueue_name}"
    gcs_cmd += f" --set volumes.gcsMounts[0].bucketName={BUCKET_NAME}"
  else:
    gcs_cmd = f" --set workload.gcsBucketForDataCataPath={BUCKET_NAME}"

  cluster_cmd = ""
  if framework == "nemo" and hypercomputer == "a3ultra":
    cluster_cmd = f" --set clusterName={cluster_name}"

  run_name_cmd = ""
  if framework == "maxtext":
    run_name_cmd = "--set workload.run_name=$JOB_NAME"

  set_aotc = ""
  if aotc:
    set_aotc = " --set-string workload.aotc=true "
  helm_cmds = (
      " helm install -f values.yaml "
      "--namespace default "
      "--set namespace=default"
      f" --set-file {framework}_config"
      f"={config_file}"
      " --set workload.image"
      f"={docker_image} "
      f"{cluster_cmd} {run_name_cmd} {gcs_cmd} {set_aotc}"
      f"{additional_cmds}"
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


def copy_bucket_cmds_nemo(recipe_repo_root, hypercomputer: str = "a3mega"):
  gcs_location = ""
  if hypercomputer == "a3ultra":
    gcs_location = f"gs://{BUCKET_NAME}/nemo-experiments/megatron_gpt/"
  else:
    gcs_location = f"gs://{BUCKET_NAME}/nemo-experiments/"

  copy_bucket_contents = (
      "export COMPLETE_JOB_NAME=$(gcloud storage ls "
      f"{gcs_location} | grep $JOB_NAME)",
      'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
      f"cd {recipe_repo_root}/src/utils/training_metrics",
      "gcloud storage cp ${COMPLETE_JOB_NAME}"
      "dllogger/rank-0/dllogger.json .",
  )
  return copy_bucket_contents


def copy_bucket_cmds_maxtext(tmpdir, recipe_repo_root):
  gcs_location = f"gs://{BUCKET_NAME}/maxtext/"

  cmds = (
      # "JOB_NAME=gunjanjalori-mixtral-8x7b-maxtext-1739253297",
      f"METRICS_FILE={tmpdir}/tflog/metrics",
      "export BUCKET_FOLDER=$(gcloud storage ls "
      f"{gcs_location} | grep $JOB_NAME)",
      'echo "BUCKET_FOLDER ${BUCKET_FOLDER}"',
      "export COMPLETE_JOB_NAME=$(gcloud storage ls "
      "${BUCKET_FOLDER}tensorboard/ | grep $JOB_NAME)",
      'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
      "export LOG_FILE=$(gcloud storage ls "
      "${COMPLETE_JOB_NAME} | grep events)",
      'echo "LOG_FILE ${LOG_FILE}"',
      "gcloud storage cp $LOG_FILE $METRICS_FILE",
  )
  return cmds


def calculate_maxtext_metrics(log_location: str, hardware: str = "a3ultra"):
  metrics, _ = metric.read_from_tb(log_location, None, None)
  step_time_metrics = metrics["perf/step_time_seconds"]
  avg_step_time = metric.aggregate_metrics(
      step_time_metrics, metric_config.AggregationStrategy.AVERAGE
  )

  tflop_per_device_per_sec_metrics = metrics["perf/per_device_tflops_per_sec"]
  avg_tflop_per_device_per_sec = metric.aggregate_metrics(
      tflop_per_device_per_sec_metrics,
      metric_config.AggregationStrategy.AVERAGE,
  )

  mfu = avg_tflop_per_device_per_sec / MAX_TFLOP[hardware]

  return mfu, avg_step_time


def get_nemo_metrics_cmds(
    batch_size,
    num_accelerators,
    precision,
    model_id,
    accelertator_type,
    temdir,
    freq: str = "weekly",
):
  step_cmd = ""
  if freq == "daily":
    step_cmd = "--start_step 0 --end_step 0 "
  cmds = (
      f"METRICS_FILE={temdir}/metrics.txt",
      "python3 process_training_results.py --file"
      f" dllogger.json --batch_size {batch_size} "
      f"--num_accelerators {num_accelerators} "
      f"--precision {precision}  "
      f"--model_type {model_id} "
      f"{step_cmd}"
      f"--accelerator_type {accelertator_type} | "
      "gsutil cp - $METRICS_FILE",
  )
  return cmds


def cleanup_cmds():
  cleanup = (
      "helm uninstall $JOB_NAME",
      "kubectl get pods "
      "--no-headers=true | awk '{print $1}' "
      "| grep $JOB_NAME | xargs kubectl delete pods",
      'echo "pods cleaned up"',
  )
  return cleanup


def get_nemo_metrics(temdir):
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


def get_bq_writer_path(tempdir):
  return os.path.join(tempdir, "benchmark-automation/benchmark_db_writer/src")


def get_recipe_repo_path(tmpdir):
  recipe_repo_root = os.path.join(
      tmpdir, "reproducible-benchmark-recipes/projects/gpu-recipes"
  )
  return recipe_repo_root


def get_cluster(hardware: str = "a3ultra"):
  if hardware == "a3mega":
    return "a3plus-benchmark", "australia-southeast1"
  if hardware == "a3ultra":
    return "a3ultra-bm-map-2", "europe-west1"


def get_scheduled_time(hardware: str, model: str, framework: str):
  """
  Returns a cron expression for the DAG schedule based on
  the given hardware, model, and framework.

  Each model runs on Thursday on a unique time so
  that we have free nodes for each.

  The alloted time for these tests is 6 pm - 10 pm PST on Thursday.
  6 PM pst -  0 2 * * 5
  10 PM pst - 0 6 * * 5

  Args:
      hardware: The hardware type (e.g., "a3ultra", "a3mega").
      model: The model ID (e.g., "mixtral-8x7b", "llama-3.1-70b").
      framework: The framework (e.g., "nemo", "maxtext").

  Returns:
      A cron expression string (e.g., "0 12 * * 4") or None
      if no schedule is defined
      for the given combination.
  """

  schedule_map = {
      "a3ultra": {
          "mixtral-8x7b": {
              "nemo": "0 3 * * 5",
              "maxtext": "0 2 * * 5",  # 6 PM PST on Thursday
          },
          "llama-3.1-70b": {
              "nemo": "0 4 * * 5",
              "maxtext": "0 5 * * 5",
          },
      },
      "a3mega": {
          "mixtral-8x7b": {
              "nemo": "0 4 * * 5",
              "maxtext": "0 3 * * 5",
          },
          "llama-3-70b": {
              "nemo": "0 2 * * 5",
              "maxtext": "0 5 * * 5",
          },
          "llama-3.1-70b": {
              "nemo": "0 2 * * 5",
              "maxtext": "0 4 * * 5",
          },
          "gpt3-175b": {
              "nemo": "0 4 * * 5",
          },
      },
  }

  if hardware in schedule_map:
    if model in schedule_map[hardware]:
      if framework in schedule_map[hardware][model]:
        return schedule_map[hardware][model][framework]

  return None  # Return None if no schedule is found for the given combination


def get_docker_image(hardware: str, framework: str):
  """
  Returns the appropriate Docker image based on the given hardware, model, and framework.

  Args:
      hardware: The hardware type (e.g., "a3ultra", "a3mega").
      framework: The framework (e.g., "nemo", "maxtext").

  Returns:
      A Docker image string or None if no image is defined for the given combination.
  """

  image_map = {
      "a3ultra": {
          "nemo": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.07-gib1.0.3-A3U",
          "maxtext": "us-central1-docker.pkg.dev/supercomputer-testing/gunjanjalori/maxtext-benchmark",
      },
      "a3mega": {
          "nemo": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo:nemo24.07-A3Mega",
          "maxtext": "us-central1-docker.pkg.dev/supercomputer-testing/gunjanjalori/maxtext-benchmark",
      },
  }

  if hardware in image_map:
    if framework in image_map[hardware]:
      return image_map[hardware][framework]

  return None  # Return None if no image is found for the given combination


def get_two_node_cmds(hypercomputer: str = "a3ultra"):
  cmd = ' --set workload.arguments="{trainer.max_steps=1}"  --set workload.gpus=16 '
  if hypercomputer == "a3mega":
    cmd += '--set workload.arguments="{model.pipeline_model_parallel_size=2}"'
  return cmd
