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
import getpass
import logging

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from collections import namedtuple
from xlml.utils import metric
from xlml.apis import metric_config
from dags.map_reproducibility.utils import gcs_automation_utils
from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from datetime import datetime, timezone
from dags import composer_env
from google.cloud import storage
from typing import Optional, Set, Tuple


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


PROJECT = "supercomputer-testing"
BUCKET_NAME = "regression-testing-xlml"

MAX_TFLOP = {"a3ultra": 989, "a3mega": 989, "a4": 2237}


class Config:

  def __init__(self, **entries):
    for key, value in entries.items():
      if isinstance(value, dict):
        setattr(self, key, Config(**value))
      elif isinstance(value, list):
        setattr(
            self,
            key,
            [
                Config(**item) if isinstance(item, dict) else item
                for item in value
            ],
        )
      else:
        setattr(self, key, value)

  def __repr__(self):
    return f"{self.__class__.__name__}({self.__dict__})"

  def to_dict(self):
    result = {}
    for key, value in self.__dict__.items():
      if isinstance(value, Config):
        result[key] = value.to_dict()
      elif isinstance(value, list):
        result[key] = [
            v.to_dict() if isinstance(v, Config) else v for v in value
        ]
      else:
        result[key] = value
    return result


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


def configure_git(
    recipes_repo_change_refs: str = None,
    bq_writer_repo_change_refs: str = None,
    gcs_automation_repo_change_refs: str = None,
    username: str = None,
    email: str = None,
):
  """Set up git configs. This is currently used to merge the change reference.

  Args:
      recipes_repo_change_refs: The change reference of the recipe GOB repo.
      bq_writer_repo_change_refs: The change reference of the BQ writer repo.
      gcs_automation_repo_change_refs: The change reference of the GCS
      automation repo.
      username: The git account username.
      email: The git account email.

  Returns:
      A command to set up git configs.
  """
  if not any((
      recipes_repo_change_refs,
      bq_writer_repo_change_refs,
      gcs_automation_repo_change_refs,
  )):
    return ()

  cmds = (
      f"git config --global user.name {username}",
      f"git config --global user.email {email}",
  )
  return cmds


def clone_recipes_gob(
    change_refs: str = None,
    recipe_branch: str = False,
):
  gob_clone_cmds = (
      "echo 'trying to clone GoB repo from outside'",
      "git clone https://ai-hypercomputer-benchmarks.googlesource.com/"
      "reproducible-benchmark-recipes",
  )
  if recipe_branch:
    gob_clone_cmds += (
        f"(cd reproducible-benchmark-recipes && git checkout {recipe_branch})",
    )
  if change_refs:
    gob_clone_cmds += (
        "(cd reproducible-benchmark-recipes && git fetch "
        "https://ai-hypercomputer-benchmarks.googlesource.com/"
        f"reproducible-benchmark-recipes {change_refs} && "
        "git merge FETCH_HEAD)",
    )
  return gob_clone_cmds


def clone_internal_recipes_gob():
  gob_clone_cmds = (
      "echo 'trying to clone internal GoB repo'",
      "git clone https://jax3p-gpu-benchmarking.googlesource.com/"
      "internal-gpu-recipes",
  )
  return gob_clone_cmds


def get_bq_writer_repo(
    change_refs: str = None,
    gcs_results_generator: bool = False,
):
  gob_clone_cmds = (
      "echo 'trying to clone GoB bq writer repo'",
      "git clone https://cmcs-perf-tooling-internal.googlesource.com/"
      "benchmark-automation",
  )
  if change_refs:
    gob_clone_cmds += (
        "(cd benchmark-automation && git fetch "
        "https://cmcs-perf-tooling-internal.googlesource.com/"
        f"benchmark-automation {change_refs} && "
        "git merge FETCH_HEAD)",
    )
  if gcs_results_generator:
    gob_clone_cmds += ("(cd benchmark-automation && ./install_mantaray.sh)",)
  return gob_clone_cmds


def get_gcs_automation_repo(
    change_refs: str = None,
    gcs_results_generator: bool = False,
):
  if not gcs_results_generator:
    return ()
  gob_clone_cmds = (
      "echo 'trying to clone GCS automation repo'",
      "git clone https://tessellations.googlesource.com/benchmarks",
  )
  if change_refs:
    gob_clone_cmds += (
        "(cd benchmarks && git fetch "
        "https://tessellations.googlesource.com/benchmarks "
        f"{change_refs} && git merge FETCH_HEAD)",
    )
  gob_clone_cmds += (
      "(cd benchmarks/automation/run_results_generator && "
      "pip install --require-hashes -r requirements.txt)",
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


def get_gpu_recipe_cmd(
    hypercomputer,
    model_id,
    framework,
    recipe_repo_root,
    storage_product: str = None,
):
  gpu_recipe_cmd = (
      "cd reproducible-benchmark-recipes/projects/gpu-recipes",
      "export RECIPE_ROOT="
      f"{recipe_repo_root}/training/{hypercomputer}/{model_id}/"
      f"{framework}-pretraining-gke"
      f"{f'-{storage_product}' if storage_product else ''}",
      "cd $RECIPE_ROOT",
  )
  return gpu_recipe_cmd


def get_pre_workload_cmds(
    model_id,
    framework,
    user: str = None,
):
  prepare_workload_cmds = (
      "NOW=$(date +%s)",
      f"export JOB_NAME={f'{user}-' if user else ''}imo-"
      f"{model_id}-$NOW-{framework}",
  )
  return prepare_workload_cmds


def get_internal_pre_workload_cmds(job_name):
  prepare_workload_cmds = (f"export JOB_NAME={job_name}",)
  return prepare_workload_cmds


def get_internal_pre_workload_job_name(
    model_id, precision, num_gpus, framework, cluster, is_sample_run=False
):
  helm_model_id = model_id.replace(".", "-")
  random_id = "".join(random.choices(string.ascii_lowercase, k=4))
  now = int(time.time())
  job_name = f"cml-{helm_model_id}-{precision}-{num_gpus}-{cluster[:3]}-{framework[:1]}-{now}-{random_id}"
  if is_sample_run:
    # use 3 char max for user_name to make sure helm job is within 53 char
    job_name = f"{getpass.getuser()[:3]}-{job_name}"
  print(f"{'*' * 20}NAME: {job_name}")
  return job_name


def find_xprof_gcs_path(gcs_path):
  """
  Find the .xplane.pb file in the latest date blob from the specified GCS path.

  Args:
      gcs_path (str): Full GCS path in the format gs://bucket-name/folder/path/

  Returns:
      str: Path to the .xplane.pb file in the latest date blob
  """
  path_without_prefix = gcs_path.removeprefix("gs://")

  parts = path_without_prefix.split("/", 1)
  bucket_name = parts[0]
  print(f"Bucket name: {bucket_name}")

  prefix = parts[1] if len(parts) > 1 else ""

  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)

  # List all blobs in the bucket with the given prefix
  print(f"Prefix: {prefix}")
  blobs = list(bucket.list_blobs(prefix=prefix))

  # Look for .xplane.pb file in the latest directory
  xplane_pb_file = None
  for blob in blobs:
    if blob.name.endswith(".xplane.pb"):
      xplane_pb_file = blob.name
      break

  if not xplane_pb_file:
    print(f"No .xplane.pb file found in {gcs_path}")
    return None

  full_xplane_pb_file = f"gs://{bucket_name}/{xplane_pb_file}"
  print(f"Found .xplane.pb file: {full_xplane_pb_file}")
  return full_xplane_pb_file


def get_patheon_job_link(region, cluster_name, job_name):
  pantheon_link = f"https://pantheon.corp.google.com/kubernetes/job/{region}/{cluster_name}/default/{job_name}"
  print(f"{'*' * 20}LINK: {pantheon_link}")
  return pantheon_link


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
    logs_bucket: str = None,
):
  gcs_cmd = ""
  if hypercomputer in ("a3ultra", "a4"):
    if framework != "maxtext" and kueue_name:
      gcs_cmd = f" --set queue={kueue_name}"
    gcs_cmd += f" --set volumes.gcsMounts[0].bucketName={logs_bucket}"
  else:
    gcs_cmd = f" --set workload.gcsBucketForDataCataPath={logs_bucket}"

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
    kueue_name: str = None,
    additional_cmds: str = "",
    bucket_name=BUCKET_NAME,
):
  gcs_cmd = ""
  if hypercomputer in ("a3ultra", "a4"):
    if framework != "maxtext" and kueue_name:
      gcs_cmd = f" --set queue={kueue_name}"
    gcs_cmd += f" --set volumes.gcsMounts[0].bucketName={bucket_name}"
  else:
    gcs_cmd = f" --set workload.gcsBucketForDataCataPath={bucket_name}"

  cluster_cmd = ""
  if framework == "nemo" and hypercomputer == "a3ultra":
    cluster_cmd = f" --set clusterName={cluster_name} "

  run_name_cmd = ""
  if framework == "maxtext":
    run_name_cmd = " --set workload.run_name=$JOB_NAME "

  set_aotc = ""
  if aotc:
    set_aotc = " --set-string workload.aotc=true "

  helm_template_path = (
      f"{recipe_repo_root}/src/helm-charts/{hypercomputer}/{framework}-training"
  )

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


def helm_apply_cmds_workload(
    framework: str,
    hypercomputer: str,
    config_file: str,
    recipe_repo_root: str,
    workload_launcher: str,
    aotc: bool = False,
    kueue_name: Optional[str] = None,
    additional_cmds: str = "",
    num_steps: Optional[int] = None,
) -> tuple[str, ...]:
  """
  Generates the Helm install command string for a workload jobset.
  """
  cmd_parts = [
      "helm",
      "install",
      "-f",
      "values.yaml",
      "--namespace",
      "default",
      "--set",
      "namespace=default",
      # Maintaining original quoting for test compatibility:
      f'--set-file workload_config="{config_file}"',
      f'--set-file workload_launcher="{workload_launcher}"',
  ]

  if kueue_name:
    cmd_parts.append(f"--set queue={kueue_name}")

  # Determine workload arguments and GCS settings based on framework
  workload_args_value_parts = []
  gcs_part_for_non_maxtext = None

  if framework == "maxtext":
    workload_args_value_parts.append(
        f"base_output_directory=gs://{BUCKET_NAME}/maxtext-experiments"
    )
    workload_args_value_parts.append(
        "jax_distributed_initialization_timeout=600"
    )
    if num_steps is not None:
      workload_args_value_parts.append(f"steps={num_steps}")
  else:  # Other frameworks (e.g., nemo)
    if num_steps is not None:
      workload_args_value_parts.append(f"trainer.max_steps={num_steps}")
    # GCS command is typically added for non-maxtext frameworks
    gcs_part_for_non_maxtext = (
        f"--set volumes.gcsMounts[0].bucketName={BUCKET_NAME}"
    )

  if workload_args_value_parts:
    cmd_parts.append(
        f'--set workload.arguments[0]="{" ".join(workload_args_value_parts)}"'
    )

  if gcs_part_for_non_maxtext:
    cmd_parts.append(gcs_part_for_non_maxtext)

  if aotc:
    cmd_parts.append("--set-string workload.aotc=true")

  if additional_cmds:
    # additional_cmds is expected to be a string of pre-formatted Helm arguments
    cmd_parts.append(additional_cmds)

  # Add job name and chart path
  cmd_parts.append("$JOB_NAME")
  cmd_parts.append(f"{recipe_repo_root}/src/helm-charts/{hypercomputer}/jobset")

  # Join all parts with a single space and return as a single-element tuple
  return (" ".join(cmd_parts),)


def wait_for_jobs_cmds():
  wait_for_job = (
      "kubectl get pods --selector=job-name=$JOB_NAME --namespace=default",
      "echo 'will wait for jobs to finish'",
      "kubectl wait --for=condition=complete "
      "job/$JOB_NAME --namespace=default --timeout=100m",
  )
  return wait_for_job


def wait_for_jobsets_cmds(timeout: str = "100m"):
  """
  Generates kubectl commands to wait for a JobSet to complete.

  Args:
    timeout: The duration to wait for the JobSet to complete (e.g., "100m").

  Returns:
    A tuple of command strings.
  """
  wait_for_jobset = (
      'echo "Listing pods associated with JobSet $JOB_NAME:"',
      "kubectl get pods --selector=jobset.sigs.k8s.io/jobset-name=$JOB_NAME --namespace=default",
      'echo "Will wait for JobSet $JOB_NAME to finish..."',
      # The condition for JobSet completion is typically 'Completed'.
      f"kubectl wait --for=condition=Completed jobset/$JOB_NAME --namespace=default --timeout={timeout}",
      'echo "JobSet $JOB_NAME finished. Describing JobSet:"',
      "kubectl describe jobset $JOB_NAME --namespace=default",
      'echo "Final pod status for JobSet $JOB_NAME:"',
      "kubectl get pods --selector=jobset.sigs.k8s.io/jobset-name=$JOB_NAME --namespace=default",
  )
  return wait_for_jobset


def internal_wait_for_jobs_cmds(timeout="100m"):
  timeout = str(timeout)
  if not timeout.endswith("m"):
    timeout += "m"
  wait_for_job = (
      "kubectl describe job $JOB_NAME --namespace=default",
      "kubectl get pods --selector=job-name=$JOB_NAME --namespace=default",
      "echo 'will wait for jobs to finish'",
      f"kubectl wait --for=condition=complete job/$JOB_NAME --namespace=default --timeout={timeout}",
      "helm status $JOB_NAME --namespace=default",
      "kubectl describe job $JOB_NAME --namespace=default",
      "kubectl get pods --selector=job-name=$JOB_NAME --namespace=default",
  )
  print("**********wait cmd is*********")
  print(wait_for_job)
  return wait_for_job


def get_job_gcs_bucket_folder(
    job_name, bucket_name=BUCKET_NAME, framework="maxtext", cluster="a3ultra"
):
  """
  Retrieve the GCS bucket folder path for a specific job.

  Args:
    job_name (str): The job name to search for.
    bucket_name (str): Name of the GCS bucket.
    framework (str): Training framework ('maxtext' or 'nemo').
    cluster (str): Cluster name to determine the path.

  Returns:
    str | None: Full GCS path to the job folder, or None if not found.
  """
  if framework == "maxtext":
    gcs_location = f"gs://{bucket_name}/maxtext/"
  elif framework == "nemo":
    if cluster in ("a4", "a3ultra"):
      gcs_location = f"gs://{bucket_name}/nemo-experiments/megatron_gpt/"
    else:
      gcs_location = f"gs://{bucket_name}/nemo-experiments/"
  else:
    raise ValueError(f"Unsupported framework: {framework}")

  bucket_folder_cmd = f"gcloud storage ls {gcs_location} | grep {job_name}"
  print(f"[INFO] Running: {bucket_folder_cmd}")

  try:
    bucket_folder = (
        subprocess.check_output(bucket_folder_cmd, shell=True).decode().strip()
    )
    bucket_folder_prefix_removed = bucket_folder.removeprefix("gs://")
    pantheon_url = f"https://pantheon.corp.google.com/storage/browser/{bucket_folder_prefix_removed}"
    print(f"[INFO] Pantheon Link: {pantheon_url}")
    return bucket_folder
  except subprocess.CalledProcessError as e:
    print(f"[ERROR] Failed to locate bucket folder: {e}")
    return None


def copy_bucket_cmds_nemo(
    recipe_repo_root, hypercomputer: str = "a3mega", bucket_name=BUCKET_NAME
):
  gcs_location = ""
  if hypercomputer in ("a3ultra", "a4"):
    gcs_location = f"gs://{bucket_name}/nemo-experiments/megatron_gpt/"
  else:
    gcs_location = f"gs://{bucket_name}/nemo-experiments/"

  copy_bucket_contents = (
      "export COMPLETE_JOB_NAME=$(gcloud storage ls "
      f"{gcs_location} | grep $JOB_NAME)",
      'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
      f"cd {recipe_repo_root}/src/utils/training_metrics",
      "gcloud storage cp ${COMPLETE_JOB_NAME}"
      "dllogger/rank-0/dllogger.json .",
  )
  return copy_bucket_contents


def copy_bucket_cmds_maxtext(tmpdir, bucket_name=BUCKET_NAME):
  gcs_location = f"gs://{bucket_name}/maxtext/"

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


def copy_bucket_cmds_workload(
    recipe_repo_root: str, tmpdir: str, framework: str
) -> Tuple[str, ...]:
  gcs_location = ""
  if framework == "maxtext":
    gcs_location = f"gs://{BUCKET_NAME}/maxtext-experiments/"
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
  else:
    gcs_location = f"gs://{BUCKET_NAME}/nemo-experiments/"
    cmds = (
        "export COMPLETE_JOB_NAME=$(gcloud storage ls "
        f"{gcs_location} | grep $JOB_NAME)",
        'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
        f"cd {recipe_repo_root}/src/utils/training_metrics",
        "gcloud storage cp ${COMPLETE_JOB_NAME}"
        "dllogger/rank-0/dllogger.json .",
    )

  return cmds


def get_skip_steps_for_metrics_calculation(config: Config):
  """Extract the number of steps to skip for the profiler from config."""
  # case 1: profiler not enabled
  # skip 2 steps, this is the default skipping since the first 2 steps' metrics are not accurate
  if not hasattr(config, "profiler"):
    logger.info("Profiler not enabled, using default skip steps: 2")
    return 2

  # case 2: profiler enabled
  # skip first n steps for profiler
  base_skip_steps = getattr(config, "skip_first_n_steps_for_profiler", 1)

  # skip profiler steps also
  additional_skip_steps = getattr(config, "profiler_steps", 5)
  total_skip_steps = base_skip_steps + additional_skip_steps
  logger.info(
      f"Profiler enabled, skipping {total_skip_steps} steps (base: {base_skip_steps}, additional: {additional_skip_steps})"
  )
  return total_skip_steps


def calculate_maxtext_metrics(
    log_location: str, hardware: str = "a3ultra", skip_first=2, skip_last=2
):
  assert skip_last >= 0, "skip_last must be a non-negative integer"
  metrics, _ = metric.read_from_tb(log_location, None, None)

  print(f"metrics - {metrics}")
  step_time_metrics = metrics["perf/step_time_seconds"]

  # Calculate the sliced metrics based on skip values
  sliced_metrics = step_time_metrics[skip_first:-skip_last]

  # Check if the resulting metrics list is empty and raise an error if it is
  if not sliced_metrics:
    logger.error(
        f"Empty metrics list after applying skip_first={skip_first} and skip_last={skip_last}. Original metrics length: {len(step_time_metrics)}"
    )

  # Apply skip_first and skip_last when aggregating
  avg_step_time = metric.aggregate_metrics(
      sliced_metrics,
      metric_config.AggregationStrategy.AVERAGE,
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
    start_step: int = None,
    end_step: int = None,
):
  step_cmd = ""
  if two_node:
    step_cmd = "--start_step 0 --end_step 0 "
  if start_step and end_step:
    step_cmd = f"--start_step {start_step} --end_step {end_step} "
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


def cleanup_all_runs_cmds(cluster, cluster_region, prefix="cml-"):
  cleanup_cmds = (
      f"echo 'Getting credentials for cluster {cluster}...' && gcloud container clusters get-credentials {cluster} --region {cluster_region} --project {PROJECT} ",
      f"echo 'Uninstalling jobs with prefix {prefix}...' && JOBS=$(kubectl get job -n default | grep \"^{prefix}\" | awk '{{print $1}}') && if [ -n \"$JOBS\" ]; then echo \"$JOBS\" | xargs -L1 helm uninstall -n default; else echo 'No matching jobs found'; fi",
  )
  return cleanup_cmds


def cleanup_cmds():
  cleanup = (
      "kubectl config set-context --current --namespace=default ",
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
  print("**********cleanup cmd is*********")
  print(cleanup)
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


def extract_value_from_yaml(tmpdir, yaml_file, key="workload.image"):
  """
  Extract a value from a YAML file given a key using dot notation.

  Args:
      tmpdir (str): Temporary directory where the YAML file is located.
      yaml_file (str): Name of the YAML file.
      key (str): Key to extract, using dot notation (e.g., 'workload.image').

  Returns:
      The value associated with the key, or None if the key is not found or an error occurs.
  """
  try:
    yaml_file_path = os.path.join(tmpdir, yaml_file)
    with open(yaml_file_path, "r", encoding="utf-8") as file:
      config = yaml.safe_load(file)

    # Navigate through the dictionary using dot notation
    keys = key.split(".")
    value = config
    for k in keys:
      if isinstance(value, dict) and k in value:
        value = value[k]
      else:
        return None  # Key not found
    return value
  except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error: {e}")
    return None


def extract_run_details(
    root: str,
    config_path: str,
    model_id: str,
    software_id: str,
    hardware_id: str,
    storage_id: str = None,
    workload_manager: str = None,
    workload_type: str = None,
    hardware_num_chips: int = None,
    hardware_num_nodes: int = None,
    configs_env: str = None,
    configs_container_version: str = None,
    benchmark_type: str = None,
    gcs_metrics_bucket: str = None,
    source_bucket: str = None,
    cloud_region: str = None,
    cluster_name: str = None,
    gcsfuse_csi_driver: str = None,
):
  """Extract the workload setups from the configs file and populate the
  RunDetails named tuple.

  Args:
      root: The root path to the recipe repo.
      config_path: The path to the workload config file.
      model_id: The ID of the model used in the run (e.g., nemo, maxtext).
      software_id: The software ID used in the BQ table. Please see the
      IDs at `ml-workload-benchmarks.benchmark_dataset_v2.software_info`.
      hardware_id: The hardware ID used in the BQ table. Please see the
      IDs at `ml-workload-benchmarks.benchmark_dataset_v2.hardware_info`.
      storage_id: The storage ID used in the BQ table. Please see the
      IDs at `ml-workload-benchmarks.benchmark_dataset_v2.storage_info`.
      workload_manager: The workload manager used in the run.
      workload_type: The type of the workload, (e.g., system, emulated).
      hardware_num_chips: The number of chips used in the run.
      hardware_num_nodes: The number of nodes used in the run
      configs_env: The environment of the workload.
      configs_container_version: The container image used in the run.
      benchmark_type: The type of the benchmark (e.g., checkpointing,
      data_loading).
      gcs_metrics_bucket: The GCS bucket in which the metrics files are stored.
      source_bucket: The GCS bucket name where the dataset is pulled from.
      cloud_region: The compute region the benchmark was run in, i.e. us-west-4.
      cluster_name: The name of the cluster used for the benchmark run.
      gcsfuse_csi_driver: The container hash of the gcsfuse csi driver.
  Returns:
      A namedtuple which stores the run details.
  """
  RunDetails = namedtuple(
      "RunDetails",
      [
          "model_id",
          "software_id",
          "hardware_id",
          "storage_id",
          "workload_gbs",
          "workload_mbs",
          "workload_type",
          "workload_manager",
          "workload_precision",
          "workload_optimizer",
          "workload_sequence_length",
          "max_epochs",
          "max_steps",
          "checkpointing_async",
          "checkpointing_interval_every_n_steps",
          "checkpointing_file_format",
          "data_loader_num_workers",
          "hardware_num_chips",
          "hardware_num_nodes",
          "configs_env",
          "configs_container_version",
          "benchmark_type",
          "gcs_metrics_bucket",
          "source_bucket",
          "cloud_region",
          "cluster_name",
          "project_name",
          "gcsfuse_csi_driver",
          "result_success",
      ],
  )

  try:
    config_path = os.path.join(root, config_path)
    with open(config_path, "r", encoding="utf-8") as file:
      config = yaml.safe_load(file)
      run_details = RunDetails(
          model_id=model_id,
          software_id=software_id,
          hardware_id=hardware_id,
          storage_id=storage_id,
          workload_gbs=config.get("model", {}).get("global_batch_size"),
          workload_mbs=config.get("model", {}).get("micro_batch_size"),
          workload_type=workload_type,
          workload_manager=workload_manager,
          workload_precision=config.get("trainer", {}).get("precision"),
          workload_optimizer=config.get("model", {})
          .get("optim", {})
          .get("name"),
          workload_sequence_length=config.get("model", {})
          .get("data", {})
          .get("seq_length"),
          max_epochs=config.get("trainer", {}).get("max_epochs"),
          max_steps=config.get("trainer", {}).get("max_steps"),
          checkpointing_async=config.get("exp_manager", {})
          .get("checkpoint_callback_params", {})
          .get("async_save", {}),
          checkpointing_interval_every_n_steps=config.get("exp_manager", {})
          .get("checkpoint_callback_params", {})
          .get("every_n_train_steps", {}),
          checkpointing_file_format=config.get("model", {}).get(
              "dist_ckpt_format"
          ),
          data_loader_num_workers=config.get("model", {})
          .get("data", {})
          .get("num_workers"),
          hardware_num_chips=hardware_num_chips,
          hardware_num_nodes=hardware_num_nodes,
          configs_env=configs_env,
          configs_container_version=configs_container_version,
          benchmark_type=benchmark_type,
          gcs_metrics_bucket=gcs_metrics_bucket,
          cloud_region=cloud_region,
          cluster_name=cluster_name,
          source_bucket=source_bucket,
          project_name=PROJECT,
          gcsfuse_csi_driver=gcsfuse_csi_driver,
          result_success=True,
      )
    return run_details
  except (FileNotFoundError, yaml.YAMLError) as e:
    print(f"Error: {e}")
    return None


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


def get_gcs_automation_repo_path(tmpdir):
  return os.path.join(tmpdir, "benchmarks/automation/run_results_generator")


def get_cluster(hardware: str = "a3ultra"):
  if hardware == "a3mega":
    return "a3plus-benchmark", "australia-southeast1"
  if hardware == "a3ultra":
    return "gke-a3ultra-bm-map-3", "europe-west1"
  if hardware == "a4":
    return "gke-a4-shared", "us-central1"


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


def get_docker_image(
    hardware: str, framework: str, model_id: Optional[str] = None
):
  """
  Returns the appropriate Docker image based on the given hardware,framework and model.

  Args:
      hardware: The hardware type (e.g., "a3ultra", "a3mega").
      framework: The framework (e.g., "nemo", "maxtext").
      model_id: The model_id. Optional.

  Returns:
      A Docker image string or None if no image is defined for the given combination.
  """

  image_map = {
      "a3ultra": {
          "nemo": {
              "default": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.07-gib1.0.3-A3U",
              "llama3-1-405b": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.12-gib1.0.3-A3U",
          },
          "maxtext": {
              "default": "us-central1-docker.pkg.dev/supercomputer-testing/gunjanjalori/maxtext-benchmark"
          },
      },
      "a3mega": {
          "nemo": {
              "default": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo:nemo24.07-A3Mega"
          },
          "maxtext": {
              "default": "us-central1-docker.pkg.dev/supercomputer-testing/gunjanjalori/maxtext-benchmark"
          },
      },
      "a4": {
          "nemo": {
              "default": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.02-gib1.0.5-A4"
          },
          "maxtext": {
              "default": "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/jax-maxtext-gpu:jax0.5.1-cuda_dl25.02-rev1-maxtext-20150317"
          },
      },
  }

  if hardware in image_map:
    if framework in image_map[hardware]:
      if model_id:
        if model_id in image_map[hardware][framework]:
          return image_map[hardware][framework][model_id]
      return image_map[hardware][framework]["default"]
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
  # model_id = model_id_raw.replace("llama", "llama-")
  num_gpus = int(parts[2].replace("gpus", ""))
  precision = parts[3]
  framework = parts[4]
  is_pgle = len(parts) >= 6 and parts[5] == "pgle"

  software_id = f"{'jax' if framework == 'maxtext' else 'pytorch'}_{framework}"

  filename_config = {
      "MODEL_ID": model_id_raw,
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

    def recursive_merge(existing, new):
      for key, value in new.items():
        if isinstance(value, dict):
          sub = getattr(existing, key, Config())
          recursive_merge(sub, value)
          setattr(existing, key, sub)
        else:
          setattr(existing, key, value)

    if config is None:
      config = Config(**result)
    else:
      recursive_merge(config, result)

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
    user: str = None,
    git_name: str = None,
    git_email: str = None,
    storage_product: str = None,
    gcs_results_generator: bool = False,
    recipe_branch: str = None,
    recipes_repo_change_refs: str = None,
    bq_writer_repo_change_refs: str = None,
    gcs_automation_repo_change_refs: str = None,
    logs_bucket: str = None,
    gcs_source_bucket: str = None,
    gcs_metrics_bucket: str = None,
    workload_image: str = None,
    workload_type: str = None,
    benchmark_type: str = None,
    gcsfuse_csi_driver: str = None,
):
  """
  The DAG task to run and process the results of NeMo workloads.

  Args:
      hypercomputer: The type of the accelerator.
      model_id: The ID of the model used in the run.
      framework: The framework used in the run.
      precision: The precision used in the run (e.g., fp32, bf16).
      metrics_model_id: The model ID used in the BQ table. Please see the
      available IDs at `ml-workload-benchmarks.benchmark_dataset_v2.model_info`.
      config_model_name: The model name in the config file.
      num_gpus: The number of chips used in the run.
      num_steps: The number of steps taken in the run.
      kueue_name: The name of the kueue.
      user: The user who triggers the workload, this is only used for manual
      run.
      git_name: The git account username. This is used to download
      the change references.
      git_email: The github account email. This is used to download
      the change references.
      storage_product: The storage product used in the workload (e.g. gcs).
      gcs_results_generator: True if enabling GCS run results generator.
      recipe_branch: The branch name of the recipe repo (default: "main").
      recipes_repo_change_refs: The change reference of the recipe repo.
      bq_writer_repo_change_refs: The change reference of the BQ writer repo.
      gcs_automation_repo_change_refs: The change reference of the gcs
      automation repo.
      logs_bucket: The logs bucket.
      gcs_source_bucket: The GCS bucket name where the dataset is pulled from.
      gcs_metrics_bucket: The GCS bucket in which the metrics files are stored.
      workload_image: The frameowrk image used by the workload.
      workload_type: workload_type: The type of the workload,
      (e.g., system, emulated).
      benchmark_type: benchmark_type: The type of the benchmark
      (e.g., checkpointing, data_loading).
      gcsfuse_csi_driver: The container hash of the gcsfuse csi driver.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                git_cookie_authdaemon()
                + configure_git(
                    git_name,
                    git_email,
                    recipes_repo_change_refs,
                    bq_writer_repo_change_refs,
                    gcs_automation_repo_change_refs,
                )
                + clone_recipes_gob(
                    recipes_repo_change_refs,
                    recipe_branch,
                )
                + get_bq_writer_repo(
                    bq_writer_repo_change_refs,
                    gcs_results_generator=gcs_results_generator,
                )
                + get_gcs_automation_repo(
                    gcs_automation_repo_change_refs,
                    gcs_results_generator=gcs_results_generator,
                )
            ),
        ],
        cwd=tmpdir,
    )

    recipe_repo_root = get_recipe_repo_path(tmpdir)
    bq_writer_repo_root = get_bq_writer_path(tmpdir)
    gcs_automation_repo_root = get_gcs_automation_repo_path(tmpdir)
    value_yaml_path = (
        f"training/{hypercomputer}/{model_id}/{framework}-pretraining-gke"
        f"{f'-{storage_product}' if storage_product else ''}/values.yaml"
    )

    workload_image = (
        workload_image
        if workload_image
        else get_docker_image(hypercomputer, framework, model_id)
    )
    logs_bucket = logs_bucket if logs_bucket else BUCKET_NAME

    num_gpus_file = extract_gpus(recipe_repo_root, value_yaml_path)

    if config_model_name:
      config_yaml_path = f"src/frameworks/{hypercomputer}/{framework}-configs/{config_model_name}"
    else:
      config_hardware = f"{'a3u-' if hypercomputer == 'a3ultra' else ''}"
      config_yaml_path = f"src/frameworks/{hypercomputer}/{framework}-configs/{model_id}-{num_gpus_file}gpus-{config_hardware}{precision}.yaml"
    full_config_yaml_path = os.path.join(recipe_repo_root, config_yaml_path)

    accelerator_type = get_accelerator_type(hypercomputer)

    additional_cmds = ""
    if two_node == True:
      additional_cmds += get_two_node_cmds(hypercomputer)

    if num_gpus:
      additional_cmds += f" --set workload.gpus={num_gpus} "
    else:
      num_gpus = num_gpus_file

    cluster, cluster_region = get_cluster(hypercomputer)

    run_details = extract_run_details(
        root=recipe_repo_root,
        config_path=config_yaml_path,
        model_id=model_id,
        software_id=get_software_id(framework),
        hardware_id=hypercomputer,
        storage_id=get_storage_id(storage_product),
        workload_manager="GKE",
        workload_type=workload_type,
        hardware_num_chips=num_gpus,
        hardware_num_nodes=int(num_gpus / get_chips_per_node(hypercomputer)),
        configs_env=("prod" if composer_env.is_prod_env() else "dev"),
        configs_container_version=workload_image,
        benchmark_type=benchmark_type,
        gcs_metrics_bucket=gcs_metrics_bucket,
        source_bucket=gcs_source_bucket,
        cloud_region=cluster_region,
        cluster_name=cluster,
        gcsfuse_csi_driver=gcsfuse_csi_driver,
    )

    print(
        f"batch size: {run_details.workload_gbs}, "
        f"num gpus: {num_gpus}, "
        f"seq length: {run_details.workload_sequence_length}, "
        f"max steps: {run_details.max_steps}"
    )

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(cluster, cluster_region)
                + get_gpu_recipe_cmd(
                    hypercomputer,
                    model_id,
                    framework,
                    recipe_repo_root,
                    storage_product,
                )
                + install_helm_cmds()
                + namespace_cmds()
                + get_pre_workload_cmds(model_id, framework, user)
                + helm_apply_cmds(
                    framework,
                    hypercomputer,
                    full_config_yaml_path,
                    recipe_repo_root,
                    workload_image,
                    cluster_name=cluster,
                    kueue_name=kueue_name,
                    logs_bucket=logs_bucket,
                    additional_cmds=additional_cmds,
                )
                + wait_for_jobs_cmds()
                + copy_bucket_cmds_nemo(
                    recipe_repo_root,
                    hypercomputer=hypercomputer,
                    bucket_name=logs_bucket,
                )
                + get_nemo_metrics_cmds(
                    run_details.workload_gbs,
                    num_gpus,
                    precision,
                    metrics_model_id,
                    accelerator_type,
                    tmpdir,
                    two_node=two_node,
                )
                + gcs_automation_utils.gcs_automation_cmds(
                    gcs_results_generator=gcs_results_generator,
                    run_details=run_details,
                    logs_bucket=logs_bucket,
                    gcs_metrics_bucket=gcs_metrics_bucket,
                    recipe_repo_root=recipe_repo_root,
                    gcs_automation_repo_root=gcs_automation_repo_root,
                )
                + cleanup_cmds()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    if gcs_results_generator:
      return

    average_step_time, mfu = get_nemo_metrics(tmpdir)
    if two_node:
      num_gpus = 16

    write_run(
        model_id=model_id,
        hardware_id=hypercomputer,
        software_id=get_software_id(framework),
        number_of_nodes=num_gpus / 8,
        number_of_chips=num_gpus,
        container_image_name=get_image_version(framework, model_id),
        global_batch_size=run_details.workload_gbs,
        precision=precision,
        optimizer=run_details.workload_optimizer,
        seq_length=run_details.workload_sequence_length,
        median_step_time=average_step_time,
        e2e_time=0,
        number_of_steps=run_details.max_steps,
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
                + copy_bucket_cmds_maxtext(tmpdir)
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


@task
def run_workload(
    hypercomputer: str,
    model_id: str,
    framework: str,
    precision: str,
    metrics_model_id: str,
    workload_launcher: str,
    num_gpus: Optional[int] = None,
    num_steps: Optional[int] = None,
    kueue_name: str = None,
    config_model_name: str = None,
    optimizer: Optional[str] = None,
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

    workload_num_gpus = (
        num_gpus
        if num_gpus
        else extract_gpus(recipe_repo_root, value_yaml_path)
    )

    if config_model_name:
      config_yaml_path = f"src/frameworks/{hypercomputer}/{framework}-configs/{config_model_name}"
    else:
      config_yaml_path = f"src/frameworks/{hypercomputer}/{framework}-configs/{model_id}-{workload_num_gpus}gpus-{hypercomputer}-{precision}.yaml"

    full_config_yaml_path = os.path.join(recipe_repo_root, config_yaml_path)
    workload_launcher_path = f"src/launchers/{workload_launcher}"
    full_workload_launcher_path = os.path.join(
        recipe_repo_root, workload_launcher_path
    )
    if framework == "nemo":
      run_details = extract_run_details(
          root=recipe_repo_root,
          config_path=config_yaml_path,
          model_id=model_id,
          software_id=get_software_id(framework),
          hardware_id=hypercomputer,
      )
      global_batch_size = run_details.workload_gbs
      optimizer = run_details.workload_optimizer
      seq_length = run_details.workload_sequence_length
      config_num_steps = run_details.max_steps
    else:
      global_batch_size = (
          extract_value_from_yaml(
              recipe_repo_root, config_yaml_path, "per_device_batch_size"
          )
          * workload_num_gpus
      )
      seq_length = extract_value_from_yaml(
          recipe_repo_root, config_yaml_path, "max_target_length"
      )
    num_steps = num_steps if num_steps else config_num_steps
    accelerator_type = get_accelerator_type(hypercomputer)
    print(
        f"batch size: {global_batch_size}, num gpus: {workload_num_gpus}, seq length: {seq_length}, num steps: {num_steps}"
    )

    additional_cmds = ""

    if num_gpus:
      additional_cmds += f" --set workload.gpus={num_gpus} "

    cluster, cluster_region = get_cluster(hypercomputer)
    if framework == "nemo":
      metrics_cmd = get_nemo_metrics_cmds(
          global_batch_size,
          workload_num_gpus,
          precision,
          metrics_model_id,
          accelerator_type,
          tmpdir,
          two_node=workload_num_gpus == 16,
      )
    else:
      metrics_cmd = ()

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
                + helm_apply_cmds_workload(
                    framework,
                    hypercomputer,
                    full_config_yaml_path,
                    recipe_repo_root,
                    workload_launcher=full_workload_launcher_path,
                    kueue_name=kueue_name,
                    additional_cmds=additional_cmds,
                    num_steps=num_steps,
                )
                + wait_for_jobsets_cmds()
                + copy_bucket_cmds_workload(
                    recipe_repo_root=recipe_repo_root,
                    tmpdir=tmpdir,
                    framework=framework,
                )
                + metrics_cmd
                + cleanup_cmds()
            ),
        ],
        cwd=tmpdir,
    )

    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    if framework == "nemo":
      average_step_time, mfu = get_nemo_metrics(tmpdir)
    else:
      log_location = os.path.join(tmpdir, "tflog/metrics")
      mfu, average_step_time = calculate_maxtext_metrics(
          log_location, hypercomputer
      )
      print(f"mfu: {mfu}")
      print(f"step_time: {average_step_time}")

    write_run(
        model_id=model_id,
        hardware_id=hypercomputer,
        software_id=get_software_id(framework),
        number_of_nodes=workload_num_gpus / 8,
        number_of_chips=workload_num_gpus,
        container_image_name=extract_value_from_yaml(
            recipe_repo_root, value_yaml_path, "workload.image"
        ),
        global_batch_size=global_batch_size,
        precision=precision,
        optimizer=optimizer,
        seq_length=seq_length,
        median_step_time=average_step_time,
        e2e_time=0,
        number_of_steps=num_steps,
        mfu=mfu,
        tokens_per_second=global_batch_size * seq_length / average_step_time,
        writer_path=bq_writer_repo_root,
        topology="-",
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


def get_storage_id(storage_product: str):
  if storage_product == "gcs":
    return "gcsfuse"
  elif storage_product == "parallelstore":
    return "parallelstore"
  else:
    return None


def get_image_version(framework: str, model_id: Optional[str] = None):
  if framework == "maxtext":
    return "maxtext_nightly"
  elif framework == "nemo":
    if model_id == "llama3-1-405b":
      return "nemo24.12-A3U"
    else:
      return "nemo24.07-A3U"
  else:
    return None


def get_chips_per_node(hardware_id: str):
  match hardware_id:
    case "a3ultra" | "a3mega" | "a4":
      return 8
    case _:
      raise ValueError(f"Warning: {hardware_id} is not supported.")
