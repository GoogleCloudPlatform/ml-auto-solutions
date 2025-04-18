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
import tempfile
import yaml
import random
import string
import time
import subprocess

from google.cloud import storage
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from xlml.utils import metric
from xlml.apis import metric_config
from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from datetime import datetime, timezone
from dags import composer_env

PROJECT = "supercomputer-testing"
BUCKET_NAME = "regression-testing-xlml"

MAX_TFLOP = {"a3ultra": 989, "a3mega": 989, "a4": 2237}


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


def clone_internal_recipes_gob():
  gob_clone_cmds = (
      "echo 'trying to clone internal GoB repo'",
      "git clone https://jax3p-gpu-benchmarking.googlesource.com/"
      "internal-gpu-recipes",
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


def get_internal_pre_workload_cmds(job_name):
  prepare_workload_cmds = (f"export JOB_NAME={job_name}",)
  return prepare_workload_cmds


def get_internal_pre_workload_job_name(model_id, framework):
  helm_model_id = model_id.replace(".", "-")
  random_id = "".join(random.choices(string.ascii_lowercase, k=4))
  now = int(time.time())
  job_name = f"coreml-{helm_model_id}-{now}-{framework}-{random_id}"
  return job_name


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
    kueue_name: str = None,
    additional_cmds: str = "",
    num_steps: int = None,
):
  gcs_cmd = ""
  if hypercomputer in ("a3ultra", "a4"):
    if framework != "maxtext" and kueue_name:
      gcs_cmd = f" --set queue={kueue_name}"
    gcs_cmd += f" --set volumes.gcsMounts[0].bucketName={BUCKET_NAME}"
  else:
    gcs_cmd = f" --set workload.gcsBucketForDataCataPath={BUCKET_NAME}"

  if num_steps:
    additional_cmds += f" --set workload.steps={num_steps} "

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


def helm_apply_cmds_internal_run(
    framework: str,
    hypercomputer: str,
    config_file,
    recipe_repo_root,
    values_file_path,
    docker_image,
    aotc: bool = False,
    cluster_name: str = "a3plus-benchmark",
    kueue_name: str = "a3-ultra",
    additional_cmds: str = "",
    test_run=False,
):
  gcs_cmd = ""
  if framework == "maxtext":
    gcs_cmd += f" --set volumes.gcsMounts[0].bucketName={BUCKET_NAME} "

  if hypercomputer == "a3ultra":
    if framework != "maxtext":
      gcs_cmd += f" --set queue={kueue_name} "
  else:
    gcs_cmd += f" --set workload.gcsBucketForDataCataPath={BUCKET_NAME} "

  cluster_cmd = ""
  if framework == "nemo" and hypercomputer == "a3ultra":
    cluster_cmd = f" --set clusterName={cluster_name} "

  run_name_cmd = ""
  if framework == "maxtext":
    run_name_cmd = " --set workload.run_name=$JOB_NAME "

  set_aotc = ""
  if aotc:
    set_aotc = " --set-string workload.aotc=true "

  if test_run:
    helm_template_path = f"/home/airflow/gcs/dags/dags/map_reproducibility/helm-charts/{hypercomputer}/{framework}-training"
  else:
    helm_template_path = f"{recipe_repo_root}/src/helm-charts/{hypercomputer}/{framework}-training"

  print(f"helm_template_path is {helm_template_path}")

  helm_cmds = (
      f" helm install -f {values_file_path} "
      "--namespace default "
      "--set namespace=default"
      f" --set-file {framework}_config"
      f"={config_file}"
      " --set workload.image"
      f"={docker_image} "
      f"{cluster_cmd} {run_name_cmd} {gcs_cmd} {set_aotc}"
      f"{additional_cmds}"
      # f" $JOB_NAME {recipe_repo_root}/src/helm-charts/{hypercomputer}/{framework}-training",
      f" $JOB_NAME {helm_template_path}",
  )
  print("*******helm cmd is*******")
  print(helm_cmds)
  return helm_cmds


def wait_for_jobs_cmds():
  wait_for_job = (
      "kubectl get pods --selector=job-name=$JOB_NAME --namespace=default",
      "echo 'will wait for jobs to finish'",
      "kubectl wait --for=condition=complete "
      "job/$JOB_NAME --namespace=default --timeout=100m",
  )
  return wait_for_job


def internal_wait_for_jobs_cmds(timeout="100m"):
  timeout = str(timeout)
  if not timeout.endswith("m"):
    timeout += "m"
  wait_for_job = (
      "kubectl get pods --selector=job-name=$JOB_NAME --namespace=default",
      "echo 'will wait for jobs to finish'",
      "kubectl wait --for=condition=complete "
      f"job/$JOB_NAME --namespace=default --timeout={timeout}",
  )
  return wait_for_job


def get_job_gcs_bucket_folder(job_name):
  """
  Get the GCS bucket folder for a specific job.

  Args:
      bucket_name (str): The name of the GCS bucket
      job_name (str): The job name to search for

  Returns:
      str: The full path to the bucket folder containing the job
  """
  gcs_location = f"gs://{BUCKET_NAME}/maxtext/"
  bucket_folder_cmd = f"gcloud storage ls {gcs_location} | grep {job_name}"

  try:
    bucket_folder = (
        subprocess.check_output(bucket_folder_cmd, shell=True).decode().strip()
    )
    print(f"BUCKET_FOLDER: {bucket_folder}")
    return bucket_folder
  except subprocess.CalledProcessError as e:
    print(f"Error finding bucket folder: {e}")
    return None


def copy_bucket_cmds_nemo(recipe_repo_root, hypercomputer: str = "a3mega"):
  gcs_location = ""
  if hypercomputer in ("a3ultra", "a4"):
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

  print(f"metrics - {metrics}")
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
    two_node: bool = False,
):
  step_cmd = ""
  if two_node:
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
      # Attempt Helm uninstall first, continue even if it fails
      "helm uninstall $JOB_NAME -n default --wait || true ",
      # Give Helm resources time to fully clean up
      "echo 'Waiting 60 seconds for helm uninstall... '",
      "sleep 60 ",
      "echo 'Attempting regular job and pod deletion... '",
      # Track if job exists and attempt standard deletion if it does
      "JOB_EXISTS=false",
      "if kubectl get job $JOB_NAME &>/dev/null; then JOB_EXISTS=true; kubectl delete job/$JOB_NAME --grace-period=30; else echo 'Job not found, skipping regular deletion'; fi ",
      # Track if pods exist and attempt standard deletion if they do
      "PODS_EXIST=false",
      "if kubectl get pods -l job-name=$JOB_NAME 2>&1 | grep -q 'No resources found'; then echo 'No pods found, skipping deletion'; else PODS_EXIST=true; kubectl delete pods -l job-name=$JOB_NAME --grace-period=30; fi ",
      # Only wait if there were resources to delete
      "[ \"$JOB_EXISTS\" = true ] || [ \"$PODS_EXIST\" = true ] && { echo 'Waiting 30 seconds for kubectl graceful termination... '; sleep 30; } || echo 'No resources found, skipping wait period' ",
      # Attempt force deletion of job if it still exists now
      "if kubectl get job $JOB_NAME &>/dev/null; then echo 'Job still exists, using force deletion...'; kubectl delete job $JOB_NAME --force --grace-period=0; else echo 'No job to force delete'; fi ",
      # Attempt force deletion of pods if they existed before and still exist now
      "if ! kubectl get pods -l job-name=$JOB_NAME 2>&1 | grep -q 'No resources found'; then echo 'Pods still exist, using force deletion...'; kubectl delete pods -l job-name=$JOB_NAME --force --grace-period=0; else echo 'No pods to force delete'; fi ",
      "echo 'Cleanup completed'",
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


def get_internal_recipe_repo_path(tmpdir):
  recipe_repo_root = os.path.join(tmpdir, "internal-gpu-recipes")
  return recipe_repo_root


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
  elif hypercomputer == "a4":
    return "b200"


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
    return "gke-a3ultra-bm-map-3", "europe-west1"
  if hardware == "a4":
    return "gke-a4-map", "us-central1"


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
          "llama3-1-70b": {
              "nemo": "0 4 * * 5",
              "maxtext": "0 4 * * 5",
          },
          "llama3-1-405b": {
              "nemo": "0 5 * * 5",
              "maxtext": "0 5 * * 5",
          },
      },
      "a3mega": {
          "mixtral-8x7b": {
              "nemo": "0 4 * * 5",
              "maxtext": "0 3 * * 5",
          },
          "llama3-70b": {
              "nemo": "0 2 * * 5",
              "maxtext": "0 5 * * 5",
          },
          "llama3-1-70b": {
              "nemo": "0 2 * * 5",
              "maxtext": "0 4 * * 5",
          },
          "gpt3-175b": {
              "nemo": "0 4 * * 5",
          },
      },
      "a4": {
          "mixtral-8x7b": {
              "nemo": "0 2 * * 5",
          },
          "llama3-1-70b": {
              "nemo": "0 3 * * 5",
              "maxtext": "0 3 * * 5",
          },
          "llama3-1-405b": {
              "nemo": "0 4 * * 5",
              "maxtext": "0 4 * * 5",
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
      "a4": {
          "nemo": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.02-gib1.0.5-A4",
          "maxtext": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/jax-maxtext-gpu:jax0.5.1-cuda_dl25.02-rev1-maxtext-20150317",
      },
  }

  if hardware in image_map:
    if framework in image_map[hardware]:
      return image_map[hardware][framework]

  return None  # Return None if no image is found for the given combination


def get_internal_docker_image(hardware: str, framework: str):
  """
  Returns the appropriate Docker image based on the given hardware, model, and framework.

  Args:
      hardware: The hardware type (e.g., "a3ultra", "a3mega").
      framework: The framework (e.g., "nemo", "maxtext").

  Returns:
      A Docker image string or None if no image is defined for the given combination.
  """
  utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

  image_map = {
      "a3ultra": {
          "nemo": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.07-gib1.0.3-A3U",
          "maxtext": f"gcr.io/tpu-prod-env-multipod/maxtext_gpu_stable_stack_nightly_jax:{utc_date}",
      },
      "a3mega": {
          "nemo": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo:nemo24.07-A3Mega",
          "maxtext": f"gcr.io/tpu-prod-env-multipod/maxtext_gpu_stable_stack_nightly_jax:{utc_date}",
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


class Config:
  """
  A simple configuration class that allows dot notation access
  to dictionary keys.
  """

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def __repr__(self):
    return repr(self.__dict__)

  def __str__(self):
    return str(self.__dict__)


def parse_internal_config_filename(filename, config=None):
  """
  Parse configuration values embedded in the filename.

  Args:
      filename (str): Example: "a3ultra_llama2-7b_8gpus_fp16_maxtext_pgle.yaml"
      config (Config, optional): Existing Config object to update. If None, a new one is created.

  Returns:
      Config: Configuration object with dot notation access
  """
  parts = filename.split(".yaml")[0].split("_")

  hypercomputer = parts[0]
  model_id_raw = parts[1]
  model_id = model_id_raw.replace("llama", "llama-")
  num_gpus = int(parts[2].replace("gpus", ""))
  precision = parts[3]
  framework = parts[4]
  is_pgle = len(parts) >= 6 and parts[5] == "pgle"

  software_id = f"{'jax' if framework == 'maxtext' else 'pytorch'}_{framework}"

  filename_config = {
      "MODEL_ID": model_id,
      "HELM_NAME_MODEL_ID": model_id_raw.replace(".", "-"),
      "PRECISION": precision,
      "HYPERCOMPUTER": hypercomputer,
      "FRAMEWORK": framework,
      "SOFTWARE_ID": software_id,
      "NUM_GPUS": num_gpus,
      "IS_PGLE": is_pgle,
  }

  if config is None:
    return Config(**filename_config)
  else:
    config.__dict__.update(filename_config)
    return config


def parse_internal_config_content(yaml_path, config=None):
  """
  Parse the internal content of a config YAML file and update the existing config.

  Args:
      yaml_path (str): Path to the YAML file
      config (Config, optional): Existing Config object to update. If None, a new one is created.

  Returns:
      Config: Updated configuration object with dot notation access
  """
  try:
    with open(yaml_path, "r") as file:
      result = yaml.safe_load(file)

    if config is None:
      config = Config(**result)
    else:
      config.__dict__.update(result)

    print("******* configs are ********")
    print(config)

    return config
  except Exception as e:
    print(f"Unexpected error: {e}")
    raise e


@task
def run_nemo_workload(
    hypercomputer: str,
    model_id: str,
    framework: str,
    precision: str,
    metrics_model_id: str,
    num_gpus: int = None,
    num_steps: int = None,
    two_node: bool = False,
    kueue_name: str = None,
    config_model_name: str = None,
):
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
    value_yaml_path = f"training/{hypercomputer}/{model_id}/{framework}-pretraining-gke/values.yaml"

    num_gpus_file = extract_gpus(recipe_repo_root, value_yaml_path)

    if config_model_name:
      config_yaml_path = f"src/frameworks/{hypercomputer}/{framework}-configs/{config_model_name}"
    else:
      config_hardware = f"{'a3u-' if hypercomputer == 'a3ultra' else ''}"
      config_yaml_path = f"src/frameworks/{hypercomputer}/{framework}-configs/{model_id}-{num_gpus_file}gpus-{config_hardware}{precision}.yaml"
    full_config_yaml_path = os.path.join(recipe_repo_root, config_yaml_path)

    (
        global_batch_size,
        optimizer,
        precision,
        seq_length,
        num_steps,
    ) = extract_run_details(recipe_repo_root, config_yaml_path)

    accelerator_type = get_accelerator_type(hypercomputer)
    print(
        f"batch size: {global_batch_size}, num gpus: {num_gpus},  precision: {precision}, seq length: {seq_length}, num steps: {num_steps}"
    )

    additional_cmds = ""
    if two_node == True:
      additional_cmds += get_two_node_cmds(hypercomputer)

    if num_gpus:
      additional_cmds += f" --set workload.gpus={num_gpus} "
    else:
      num_gpus = num_gpus_file

    cluster, cluster_region = get_cluster(hypercomputer)
    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(cluster, cluster_region)
                + get_gpu_recipe_cmd(
                    hypercomputer, model_id, framework, recipe_repo_root
                )
                + install_helm_cmds()
                + namespace_cmds()
                + get_pre_workload_cmds(model_id, framework)
                + helm_apply_cmds(
                    framework,
                    hypercomputer,
                    full_config_yaml_path,
                    recipe_repo_root,
                    get_docker_image(hypercomputer, framework),
                    cluster_name=cluster,
                    kueue_name=kueue_name,
                    additional_cmds=additional_cmds,
                )
                + wait_for_jobs_cmds()
                + copy_bucket_cmds_nemo(
                    recipe_repo_root,
                    hypercomputer=hypercomputer,
                )
                + get_nemo_metrics_cmds(
                    global_batch_size,
                    num_gpus,
                    precision,
                    metrics_model_id,
                    accelerator_type,
                    tmpdir,
                    two_node=two_node,
                )
                + cleanup_cmds()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    average_step_time, mfu = get_nemo_metrics(tmpdir)
    if two_node:
      num_gpus = 16

    write_run(
        model_id=model_id,
        hardware_id=hypercomputer,
        software_id=get_software_id(framework),
        number_of_nodes=num_gpus / 8,
        number_of_chips=num_gpus,
        container_image_name=get_image_version(framework),
        global_batch_size=global_batch_size,
        precision=precision,
        optimizer=optimizer,
        seq_length=seq_length,
        median_step_time=average_step_time,
        e2e_time=0,
        number_of_steps=num_steps,
        mfu=mfu,
        tokens_per_second=1,
        writer_path=bq_writer_repo_root,
        topology="2X2",
        comment="Regression tests",
        is_test=(False if composer_env.is_prod_env() else True),
    )


@task
def run_maxtext_workload(
    hypercomputer: str,
    model_id: str,
    framework: str,
    precision: str,
    num_steps: int,
    batch_size_per_device: int,
    kueue_name: str,
    optimizer: str,
    sequence_length: int,
    helm_model_id: str,
    num_gpus: int = None,
    gpu_overide: bool = True,
):
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

    value_yaml_path = f"training/{hypercomputer}/{model_id}/{framework}-pretraining-gke/values.yaml"

    recipe_repo_root = get_recipe_repo_path(tmpdir)
    bq_writer_repo_root = get_bq_writer_path(tmpdir)

    num_gpus_in_file = extract_gpus(recipe_repo_root, value_yaml_path)
    gpu_helm_cmd = ""
    if num_gpus == None:
      num_gpus = num_gpus_in_file
    elif num_gpus != num_gpus_in_file:
      gpu_helm_cmd = f" --set workload.gpus={num_gpus} "

    if gpu_overide == False:
      num_gpus = num_gpus_in_file  # This is for two node tests, they'll use the same config of more nodes

    config_hardware = (
        f"{'a3u' if hypercomputer == 'a3ultra' else hypercomputer}"
    )
    config_yaml_path = f"src/frameworks/{hypercomputer}/maxtext-configs/{model_id}-{num_gpus}gpus-{config_hardware}-{precision}.yaml"
    full_config_yaml_path = os.path.join(recipe_repo_root, config_yaml_path)

    cluster, cluster_region = get_cluster(hypercomputer)
    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(cluster, cluster_region)
                + get_gpu_recipe_cmd(
                    hypercomputer, model_id, framework, recipe_repo_root
                )
                + install_helm_cmds()
                + namespace_cmds()
                + get_pre_workload_cmds(helm_model_id, framework)
                + helm_apply_cmds(
                    framework,
                    hypercomputer,
                    full_config_yaml_path,
                    recipe_repo_root,
                    get_docker_image(hypercomputer, framework),
                    cluster_name=cluster,
                    kueue_name=kueue_name,
                    additional_cmds=gpu_helm_cmd,
                    num_steps=num_steps,
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

    mfu, step_time = calculate_maxtext_metrics(log_location, hypercomputer)

    print(f"mfu: {mfu}")
    print(f"step_time: {step_time}")

    write_run(
        model_id=model_id,
        hardware_id=hypercomputer,
        software_id=get_software_id(framework),
        number_of_nodes=num_gpus / 8,
        number_of_chips=num_gpus,
        container_image_name=get_image_version(framework),
        global_batch_size=batch_size_per_device * num_gpus,
        precision=precision,
        optimizer=optimizer,
        seq_length=sequence_length,
        median_step_time=step_time,
        e2e_time=step_time * num_steps,
        number_of_steps=num_steps,
        mfu=mfu,
        tokens_per_second=-1,
        writer_path=bq_writer_repo_root,
        topology="",
        comment="Regression tests",
        is_test=(False if composer_env.is_prod_env() else True),
    )


def get_software_id(framework: str):
  if framework == "maxtext":
    return "jax_maxtext"
  elif framework == "nemo":
    return "pytorch_nemo"
  else:
    return None


def get_image_version(framework: str):
  if framework == "maxtext":
    return "maxtext_nightly"
  elif framework == "nemo":
    return "nemo24.07-A3U"
  else:
    return None
