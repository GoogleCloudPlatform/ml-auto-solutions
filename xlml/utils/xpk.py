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


def get_xpk_setup_cmd(tmpdir):
  return [
      "set -xue",
      f"git clone --branch v0.4.1 https://github.com/AI-Hypercomputer/xpk {tmpdir}/xpk",
      "pip install ruamel.yaml docker",
  ]


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
    ramdisk_directory: str = "",
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
        f" --restart-on-user-code-failure --ramdisk-directory={ramdisk_directory}"
    )
    cmds = get_xpk_setup_cmd(tmpdir)
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
          f"No pods found but job is complete for workload selector: {workload_id}"
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
    workload_id: str, project_id: str, zone: str, cluster_name: str
) -> bool:
  """Delete workload."""
  with tempfile.TemporaryDirectory() as tmpdir:
    workload_delete_cmd = (
        f"python {tmpdir}/xpk/xpk.py workload delete"
        f" --cluster={cluster_name} --workload={workload_id}"
        f" --project={project_id} --zone={zone}"
    )

    cmds = get_xpk_setup_cmd(tmpdir)
    cmds.append(workload_delete_cmd)
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={**os.environ, "KUBECONFIG": os.path.join(tmpdir, "xpk.conf")},
    )
    assert (
        result.exit_code == 0
    ), f"XPK clean-up failed with code {result.exit_code}"
