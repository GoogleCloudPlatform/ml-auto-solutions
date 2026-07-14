# Copyright 2023 Google LLC
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

"""Common funcitons and tasks for MaxText Pathways DAGs"""

import re
import time

from absl import logging
from airflow.decorators import task
from google.cloud import logging as gcp_logging
from xlml.utils import gke, xpk


def generate_recipe_workload_id(dag_id: str) -> tuple[str, str]:
  """Generate a workload_id following the standard naming convention."""
  time.localtime()
  timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
  name = f"{dag_id[:10]}-{timestamp[:10]}"
  name = name[:40].replace("_", "-")

  return name


def generate_install_dependencies_commands() -> str:
  """Generate shell commands to install necessary dependencies in the Pod."""
  # fmt: off
  return " && ".join([
      # Update apt package list
      "sudo apt-get update",

      # Install kubectl
      "sudo apt-get install -y kubectl",

      # Install GKE auth plugin for cluster authentication
      "sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin -y",

      # Install xpk
      *xpk.get_xpk_setup_cmd("/root", xpk.MAIN_BRANCH),

      # Install dependencies for maxtext
      "pip install omegaconf",

      # Prepare environment for further pip installs
      "cd /deps",
      "export USER=root",
  ])
  # fmt: on


@task.python(multiple_outputs=True)
def get_dag_parameters(**context) -> dict:
  """Fetches and returns the DAG run's configuration parameters."""
  dag_params = context.get("params", {})

  return dag_params


@task.python(multiple_outputs=True)
def generate_derived_parameters(dag_params: dict, dag_id: str) -> dict:
  """Generates new parameters based on the initial DAG parameters."""
  derived_params = {}

  # Generate recipe workload_id.
  name = generate_recipe_workload_id(dag_id)
  derived_params["workload_id"] = name

  # Generate region by zone
  derived_params["region"] = gke.zone_to_region(dag_params["zone"])

  # Generate device_type.
  device_type = (
      dag_params["device_version"] + "-" + str(dag_params["core_count"])
  )
  derived_params["device_type"] = device_type

  # Confirm whether to use customized_model_name.
  if dag_params["selected_model_names"] == "customized_model_name":
    derived_params["selected_model_names"] = dag_params["customized_model_name"]

  if dag_params["elastic_type"] in ["Pause-resume", "Replica-resize"]:
    core_calc = dag_params["core_count"] // 4
    if dag_params["elastic_type"] == "Pause-resume":
      derived_params["elastic_min_slice_count"] = -1
      derived_params["topology"] = f"tpuv6e:4x{core_calc}"
      derived_params["num_elastic_slices"] = 1
    else:
      derived_params["elastic_min_slice_count"] = 1
      derived_params["topology"] = ",".join([f"tpuv6e:4x{core_calc}"] * 2)
      derived_params["num_elastic_slices"] = 2

  return derived_params


@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def check_gcp_logs_exist(
    project_id: str,
    cluster_name: str,
    workload_id: str,
    expect_log_contains: str,
    location: str,  # e.g., 'us-central1' or zone 'us-central1-a'
    expected_count: int = 1,
) -> bool:
  """
  Counts occurrences of a string pattern in GCP
  Cloud Logging for a specific workload.
  """
  # Initialize the GCP Logging Client
  client = gcp_logging.Client(project=project_id)

  log_filter = (
      f'resource.type="k8s_container" '
      f'resource.labels.cluster_name="{cluster_name}" '
      f'resource.labels.location="{location}" '
      'resource.labels.namespace_name="default" '
      f'resource.labels.pod_name=~"{workload_id}.*"'
  )

  logging.info(f"Querying GCP Logging with filter: {log_filter}")

  # Fetch the entries. (Adjust page_size based on log volume to optimize speed)
  entries = client.list_entries(filter_=log_filter, page_size=500)

  # Consolidate all log payloads into a single text body
  log_lines = []
  for entry in entries:
    payload = entry.payload

    if payload:
      if isinstance(payload, str):
        log_lines.append(payload)
      elif isinstance(payload, dict):
        message = payload.get("message") or payload.get("textPayload")
        if message:
          log_lines.append(str(message))
        else:
          log_lines.append(str(payload))

  full_logs_text = "\n".join(log_lines)

  if not full_logs_text:
    logging.info("No logs found yet in Cloud Logging for filter.")
    return False

  # Normalize input to a list if a single string is passed
  if isinstance(expect_log_contains, str):
    patterns = [expect_log_contains]
  else:
    patterns = expect_log_contains

  all_patterns_found = True
  for pattern in patterns:
    # re.escape matches your original literal string search logic
    log_matches = re.findall(re.escape(pattern), full_logs_text)
    log_count = len(log_matches)
    logging.info(f"Logs: '{pattern}' found {log_count} times, ")

    if log_count < expected_count:
      logging.info(
          f"Pattern '{pattern}' found {log_count} times, "
          f"which is less than expected_count ({expected_count})."
      )
      all_patterns_found = False
      break

  if all_patterns_found:
    logging.info(
        "All expected log patterns found successfully in GCP Cloud Logging."
    )
    return True

  logging.info("Waiting for matching log pattern in GCP...")
  return False
