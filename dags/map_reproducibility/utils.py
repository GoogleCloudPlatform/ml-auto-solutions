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
from google.cloud import storage


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


def git_cookie_authdaemon():
  auth_cmds = (
      "git clone https://gerrit.googlesource.com/gcompute-tools",
      "echo 'trying to run git-cookie-authdaemon'",
      # Check if the daemon is already running
      "if pgrep -f git-cookie-authdaemon; then "
      "  echo 'git-cookie-authdaemon is already running'; "
      "else "
      "  ./gcompute-tools/git-cookie-authdaemon || echo 'Error running git-cookie-authdaemon'; "  # Run if not running
      "fi",
  )
  return auth_cmds


def clone_gob():
  gob_clone_cmds = (
      "echo 'trying to clone GoB repo from outside'",
      "git clone https://ai-hypercomputer-benchmarks.googlesource.com/"
      "reproducible-benchmark-recipes",
      "cd reproducible-benchmark-recipes/projects",
      "cd gpu-recipes",
      "pwd",
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
      "echo 'will wait for jobs to finish'",
      "kubectl wait --for=condition=complete "
      "job/$JOB_NAME --namespace=default --timeout=100m",
  )
  return wait_for_job


def copy_bucket_cmds():
  copy_bucket_contents = (
      "export COMPLETE_JOB_NAME=$(gcloud storage ls "
      "gs://$BUCKET_NAME/nemo-experiments/ | grep $JOB_NAME)",
      'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
      "cd $REPO_ROOT/src/utils/training_metrics",
      "gcloud storage cp ${COMPLETE_JOB_NAME}"
      "dllogger/rank-0/dllogger.json .",
  )
  return copy_bucket_contents


def get_metrics_cmds():
  # TODO(gunjanj007): get these parameters from the recipe
  get_metrics = (
      "METRICS_FILE=$COMPLETE_JOB_NAME/metrics.txt",
      "python3 process_training_results.py --file"
      " dllogger.json --batch_size 2048 "
      "--num_accelerators 256 "
      "--precision fp8  "
      "--model_type gpt3-175b "
      "--accelerator_type h100 | "
      "gsutil cp - $METRICS_FILE",
      'echo "METRICS_FILE=${METRICS_FILE}"',
  )
  return get_metrics


def get_aotc_repo():
  gob_clone_cmds = (
      "echo 'trying to clone GoB aotc repo'",
      "git clone https://cmcs-perf-tooling-internal.googlesource.com/"
      "benchmark-automation",
      "cd benchmark-automation/aotc/src",
      "export PYTHONPATH=$PWD",
      'echo "PYTHONPATH=$PYTHONPATH and METRICS_FILE=$METRICS_FILE"',
  )
  return gob_clone_cmds


def cleanup_cmds():
  cleanup = (
      "cd $REPO_ROOT",
      "cd ../../..",
      "kubectl get pods "
      "--no-headers=true | awk '{print $1}' "
      "| grep $JOB_NAME |  xargs kubectl delete pods",
      "helm uninstall $JOB_NAME",
  )
  return cleanup


def get_metrics_from_gcs(bucket_name, file_name):
  # Initialize GCS and BigQuery clients
  storage_client = storage.Client()

  # Get the bucket and file
  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(file_name)

  # Download the file content
  metrics_output = blob.download_as_string().decode("utf-8")

  # Parse the metrics (adjust based on your file format)
  lines = metrics_output.splitlines()
  average_step_time = float(lines[0].split(": ")[1])
  tflops_per_accelerator = float(lines[1].split(": ")[1])
  mfu = float(lines[2].split(": ")[1])

  print(f"Average Step Time: {average_step_time}")
  print(f"TFLOPS/Accelerator: {tflops_per_accelerator}")
  print(f"MFU: {mfu}")


def extract_bucket_file_name(last_line):
  metrics_file = None

  match = re.search(r"PYTHONPATH=(.*?)\s+METRICS_FILE=(.*)", last_line)
  if match:
    python_path = match.group(1)
    metrics_file = match.group(2)
    print(f"PYTHONPATH in python: {python_path}")
    print(f"METRICS_FILE: {metrics_file}")
  else:
    print("Error: Could not extract PYTHONPATH and METRICS_FILE")
  print(f"Metrics file name: {metrics_file}")
  if metrics_file:
    # Extract bucket_name and file_name
    bucket_name = re.search(r"gs://([^/]+)/", metrics_file).group(1)
    file_name = re.search(r"gs://[^/]+/(.+)", metrics_file).group(1)

    print(f"Bucket name: {bucket_name}")
    print(f"File name: {file_name}")
  else:
    print("Metrics file not found in the output.")

  return bucket_name, file_name, python_path


def extract_python_path(bash_result_output):
  python_path = None
  for line in bash_result_output.splitlines():
    if line.startswith("PYTHONPATH"):
      python_path = line.split("=", 1)[1]
      break

    print(f"Pyhon path name: {python_path}")

  return python_path
