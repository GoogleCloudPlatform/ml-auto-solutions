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

"""Utilities to run workloads with xpk (https://github.com/AI-Hypercomputer/xpk)."""

import os
import tempfile
import uuid
from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.hooks.subprocess import SubprocessHook
from kubernetes import client as k8s_client
from xlml.apis import metric_config
from xlml.utils import gke
from dags.common.vm_resource import GpuVersion
from airflow.providers.google.cloud.operators.gcs import GCSHook
import re
from google.cloud import logging as log_explorer
from datetime import datetime, timezone, timedelta
from typing import Optional

# b/411426745 - Setting branch to 0.4.1 till the depdency issue is resolved.
MAIN_BRANCH = "v0.4.1"
# Duration = past 7 days
LOGGING_URL_FORMAT = (
    "https://pantheon.corp.google.com/logs/query;"
    + "query=resource.type%3D%22k8s_container%22%0A"
    + "resource.labels.project_id%3D%22{project}%22%0A"
    + "resource.labels.location%3D%22{region}%22%0A"
    + "resource.labels.cluster_name%3D%22{cluster}%22%0A"
    + "resource.labels.namespace_name%3D%22default%22%0A"
    + "labels.k8s-pod%2Fjobset_sigs_k8s_io%2F"
    + "jobset-name%3D%22{workload_id}%22%20severity%3E%3DDEFAULT;"
    + "storageScope=project;duration=P7D?e=13803378&"
    + "mods=allow_workbench_image_override&project={project}"
)


@task
def list_log_entries(project_id: str, location: str, cluster_name: str, 
                     namespace: str = "default", pod_pattern: str = "*", 
                     container_name: Optional[str] = None, text_filter: Optional[str] = None,
                     start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> int:
    """
    List log entries for the specified Google Cloud project.
    This function connects to Google Cloud Logging, constructs a filter for Kubernetes container logs
    within a specific project, location, cluster, namespace, and pod name pattern, and retrieves log
    entries from the specified time range. It prints the timestamp, severity, resource information, and payload
    for each log entry found.
    Args:
        project_id: The Google Cloud project ID
        location: GKE cluster location
        cluster_name: GKE cluster name
        namespace: Kubernetes namespace (defaults to "default")
        pod_pattern: Pattern to match pod names (defaults to "*")
        container_name: Optional container name to filter logs
        text_filter: Optional comma-separated string to filter log entries by textPayload content
        start_time: Optional start time for log retrieval (defaults to 12 hours ago)
        end_time: Optional end time for log retrieval (defaults to now)
    
    Returns:
        int: Number of log entries found
    """

    # Create a Logging Client for the specified project
    logging_client = log_explorer.Client(project=project_id)

    # Set the time window for log retrieval: default to last 12 hours if not provided
    if end_time is None:
        end_time = datetime.now(timezone.utc)
    if start_time is None:
        start_time = end_time - timedelta(hours=12)

    # Format times as RFC3339 UTC "Zulu" format required by the Logging API
    start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Construct the log filter
    log_filter = (
        f'resource.labels.project_id="{project_id}" '
        f'resource.labels.location="{location}" '
        f'resource.labels.cluster_name="{cluster_name}" '
        f'resource.labels.namespace_name="{namespace}" '
        f'resource.labels.pod_name:"{pod_pattern}" '
        'severity>=DEFAULT '
        f'timestamp>="{start_time_str}" '
        f'timestamp<="{end_time_str}"'
    )

    # Add container name filter if provided
    if container_name:
        log_filter += f' resource.labels.container_name="{container_name}"'

    # Add text content filter if provided
    if text_filter:
        filter_terms = text_filter.split(',')  # Split by comma
        for term in filter_terms:
            log_filter += f' textPayload:"{term.strip()}"'

    # Retrieve log entries matching the filter
    logging.info(log_filter)
    entries = logging_client.list_entries(filter_=log_filter)
    entry_count = 0
    for entry in entries:
        entry_count += 1
        print(f"\n[{entry_count}] LOG ENTRY")
        print(f"├─ Timestamp: {entry.timestamp}")
        print(f"├─ Severity: {entry.severity}")
        print(f"├─ Resource: {entry.resource.type}")
        print(f"├─ Labels: {entry.resource.labels}")
        if entry.payload is not None:
            print(f"└─ Payload:")
            # Format payload with indentation
            payload_str = str(entry.payload)
            for line in payload_str.split('\n'):
                print(f"   {line}")
        print("-" * 80)

    print(f"\n{'='*80}")
    print(f"SUMMARY: {entry_count} log entries found")
    print(f"{'='*80}")

    if entry_count > 0:
      return True
    return False
                         

def get_xpk_setup_cmd(tmpdir, branch: str = MAIN_BRANCH):
  clone_branch = (
      f"git clone --branch {branch} https://github.com/AI-Hypercomputer/xpk"
      f" {tmpdir}/xpk"
  )
  cmds = [
      "set -xue",
      clone_branch,
      "pip install ruamel.yaml docker",
  ]
  return cmds


def is_valid_gpu_version(accelerator_type: str):
  if accelerator_type in [member.value for member in GpuVersion]:
    return True
  return False


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate a valid workload ID."""
  import re

  short_id = str(uuid.uuid4())[:8]
  # Remove all non-alphanumeric characters, and truncate to ensure the result
  # is less than 40 characters.
  short_benchmark = re.sub(r"[^a-zA-Z0-9-]+", "", benchmark_id)[:32]
  return f"{short_benchmark}{short_id}"


@task
def run_workload(
    task_id: str,
    cluster_project: str,
    zone: str,
    cluster_name: str,
    benchmark_id: str,
    workload_id: str,
    gcs_path: str,
    docker_image: str,
    accelerator_type: str,
    run_cmds: str,
    num_slices: int = 1,
    use_vertex_tensorboard: bool = False,
    use_pathways: bool = False,
    ramdisk_directory: str = "",  # Directory for enabling emergency checkpointing
    mtc_enabled: bool = False,  # It enables MTC phase-2 drivers
    xpk_branch: str = MAIN_BRANCH,
):
  """Run workload through xpk tool."""

  with tempfile.TemporaryDirectory() as tmpdir:
    if accelerator_type in [
        GpuVersion.XPK_H100.value,
        GpuVersion.XPK_H100_MEGA.value,
    ]:
      multi_keyword = "num-nodes"
    else:
      multi_keyword = "num-slices"

    create_field = "create-pathways" if use_pathways else "create"
    type_field = "tpu-type" if use_pathways else "device-type"

    workload_create_cmd = (
        f"python {tmpdir}/xpk/xpk.py workload {create_field}"
        f" --cluster={cluster_name} --workload={workload_id}"
        f" --command='{run_cmds}' --{type_field}={accelerator_type}"
        f" --{multi_keyword}={num_slices} --docker-image={docker_image}"
        f" --project={cluster_project} --zone={zone}"
        f" --env {metric_config.SshEnvVars.GCS_OUTPUT.name}={gcs_path}"
        " --restart-on-user-code-failure"
    )
    if ramdisk_directory:
      workload_create_cmd += f" --ramdisk-directory={ramdisk_directory}"
    if mtc_enabled:
      workload_create_cmd += " --mtc-enabled"

    # If using a valid GPU and the XPK branch is set to "main", then branch is switch to "v0.4.1".
    if is_valid_gpu_version(accelerator_type) and xpk_branch == MAIN_BRANCH:
      xpk_branch = "v0.4.1"

    cmds = get_xpk_setup_cmd(tmpdir, xpk_branch)
    if accelerator_type == GpuVersion.XPK_H100_MEGA.value:
      workload_create_cmd += " --scheduler=gke.io/topology-aware-auto"
    if use_vertex_tensorboard:
      workload_create_cmd += " --use-vertex-tensorboard"
      vertex_ai_dependency = (
          "pip install -U google-cloud-aiplatform cloud-accelerator-diagnostics"
      )
      cmds.append(vertex_ai_dependency)
    cmds.append(workload_create_cmd)
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={**os.environ, "KUBECONFIG": os.path.join(tmpdir, "xpk.conf")},
    )
    assert (
        result.exit_code == 0
    ), f"XPK command failed with code {result.exit_code}"


def _get_core_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.CoreV1Api:
  """Create a core API client for the given cluster."""
  client = gke.get_authenticated_client(project_id, region, cluster_name)

  # Initilize the client
  core_api = k8s_client.CoreV1Api(client)
  logging.info("Successful initilize k8s client from cluster response.")
  return core_api


def _list_workload_pods(
    core_api: k8s_client.CoreV1Api, workload_id: str
) -> k8s_client.V1PodList:
  """List all pods for the given workload."""
  logging.info(f"Getting pods for workload_id: {workload_id}")
  pods = core_api.list_namespaced_pod(
      label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      namespace="default",
  )
  return pods


def _get_batch_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.BatchV1Api:
  """Create a batch API client for the given cluster."""
  client = gke.get_authenticated_client(project_id, region, cluster_name)

  # Initilize the client
  batch_api = k8s_client.BatchV1Api(client)
  logging.info(
      "Successful initilize k8s batch api client from cluster response."
  )
  return batch_api


def _get_workload_job(
    batch_api: k8s_client.BatchV1Api, workload_id: str
) -> k8s_client.V1Job:
  """Get the job for a given workload."""
  logging.info(f"Getting job for workload_id: {workload_id}")
  jobs = batch_api.list_namespaced_job(
      label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      namespace="default",
  )
  if len(jobs.items) == 0:
    logging.info(f"Getting job for workload_id: {workload_id}")
    return None

  if len(jobs.items) > 1:
    logging.info(f"Got more than one job for workload_id: {workload_id}")
    for i, job in enumerate(jobs.items):
      logging.info(f"Job {i=}")
      logging.info(f"{job}")

  return jobs.items[0]


def _get_pods(
    core_api: k8s_client.CoreV1Api, namespace: str, 
) -> k8s_client.V1PodList:
  """List all pods for the given namespace."""
  pods = core_api.list_namespaced_pod(
      namespace=namespace,
  )
  return pods

  
@task.sensor(poke_interval=60, timeout=600, mode="reschedule")
def wait_for_workload_start(
    workload_id: str, project_id: str, region: str, cluster_name: str
) -> bool:
  """Check if the workload has started."""
  core_api = _get_core_api_client(project_id, region, cluster_name)
  pods = _list_workload_pods(core_api, workload_id)
  print(f"Found {len(pods.items)} pods for workload {workload_id}")
  return len(pods.items) > 0


@task.sensor(poke_interval=60, timeout=600, mode="reschedule")
def wait_for_workload_completion(
    workload_id: str, project_id: str, region: str, cluster_name: str
) -> bool:
  """Check the workload status."""
  core_api = _get_core_api_client(project_id, region, cluster_name)
  pods = _list_workload_pods(core_api, workload_id)

  if not pods.items:
    logging.info(f"No pods found for workload selector: {workload_id}.")

    # Pathways jobs delete all pods on failure so we must also check if the job
    # is complete
    batch_api = _get_batch_api_client(project_id, region, cluster_name)
    job = _get_workload_job(batch_api, workload_id)
    if job is None:
      logging.info(
          f"No pods or jobs were found for workload selector: {workload_id}"
      )
      return False

    if any(condition.type == "Failed" for condition in job.status.conditions):
      # Don't keep retrying if the job has failed
      raise AirflowFailException('Job has condition type: "Failed"')

    if any(condition.type == "Complete" for condition in job.status.conditions):
      logging.info(
          "No pods found but job is complete for workload selector:"
          f" {workload_id}"
      )
      return True

    return False

  if any(pod.status.phase in ["Pending", "Running"] for pod in pods.items):
    logging.info("At least one pod has yet to complete.")
    return False

  try:
    for pod in pods.items:
      if pod.status.phase == "Failed":
        # Don't keep retrying if the pod has failed
        raise AirflowFailException(f"Bad pod phase: {pod.status.phase}")
      elif pod.status.phase in ["Unknown"]:
        raise RuntimeError(f"Bad pod phase: {pod.status.phase}")
  finally:
    # TODO(jonbolin): log printing for GPUs, which have multiple containers
    if len(pod.spec.containers) == 1:
      # Print the logs of the last pod checked - either the first failed pod or
      # the last successful one.
      logs = core_api.read_namespaced_pod_log(
          name=pod.metadata.name, namespace=pod.metadata.namespace
      )
      logging.info(f"Logs for pod {pod.metadata.name}:")
      for line in logs.split("\n"):
        logging.info(line)
    url = LOGGING_URL_FORMAT.format(
        project=project_id,
        region=region,
        cluster=cluster_name,
        workload_id=workload_id,
    )
    logging.info(f"Link to logs: {url}")

  logging.info("All pod(s) phase are succeeded.")
  return True


@task(trigger_rule="all_done")
def clean_up_workload(
    workload_id: str,
    project_id: str,
    zone: str,
    cluster_name: str,
    xpk_branch: str = MAIN_BRANCH,
) -> bool:
  """Delete workload."""
  with tempfile.TemporaryDirectory() as tmpdir:
    workload_delete_cmd = (
        f"python {tmpdir}/xpk/xpk.py workload delete"
        f" --cluster={cluster_name} --workload={workload_id}"
        f" --project={project_id} --zone={zone}"
    )

    cmds = get_xpk_setup_cmd(tmpdir, xpk_branch)
    cmds.append(workload_delete_cmd)
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={**os.environ, "KUBECONFIG": os.path.join(tmpdir, "xpk.conf")},
    )
    assert (
        result.exit_code == 0
    ), f"XPK clean-up failed with code {result.exit_code}"


@task
def validate_saving_checkpoint(
    output_path: str
) -> bool:
  """Check the gcs bucket checkpointing files status."""
  hook = GCSHook()
  pattern = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<prefix>.+)$")
  m = pattern.match(output_path)
  bucket_name = m.group("bucket")
  prefix = m.group("prefix")
  logging.info(f"output_path:{output_path}")
  logging.info(f"bucket:{bucket_name}")
  logging.info(f"prefix:{prefix}")
  objects = hook.list(bucket_name=bucket_name, prefix=prefix)
  if not objects or len(objects) <= 0:
    raise AirflowFailException()

@task
def validate_csi_checkpoint(
    project_id: str, 
    region: str, 
    cluster_name: str
) -> bool:
  """Check the csi container checkpointing files status."""
  core_api = _get_core_api_client(project_id, region, cluster_name)
  pods = _get_pods(core_api, 'gke-managed-checkpointing')
  if any(pod.status.phase in ["Pending"] for pod in pods.items):
    logging.info("Some of the pods is still pending. Waiting to start")
    return False

  cmd = ["bash", "-c", f"ls /local/tmpfs/client/"]
  for pod in pods.items:
    # Need to be imporved so it can compare steps with csid driver
    if pod.status.phase == "Running" and "multitier-driver" in pod.metadata.name:
      response = _execute_command_in_pod(core_api=core_api, pod=pod, command=cmd, container='csi')
      files = response.strip().split("\n")
      logging.info("Files ===> ", files)
      if len(files) > 0:
        return True
