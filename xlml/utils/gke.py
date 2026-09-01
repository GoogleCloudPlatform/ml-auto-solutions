"""Utilities for GKE."""

import base64
import concurrent.futures
import datetime
import logging
import tempfile
import time
from typing import Any, Dict, Optional

from airflow.decorators import task, task_group
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
import google.auth
import google.auth.transport.requests
from google.cloud import container_v1
import kubernetes

from xlml.apis import gcp_config, test_config
from xlml.utils import composer


class PodsNotReadyError(Exception):
  """Exception raised when pods are not ready within the expected timeout."""

  def __init__(self, message):
    super().__init__(message)


def get_authenticated_client(
    project_name: str, region: str, cluster_name: str
) -> kubernetes.client.ApiClient:
  container_client = container_v1.ClusterManagerClient()
  cluster_path = (
      f"projects/{project_name}/locations/{region}/clusters/{cluster_name}"
  )
  response = container_client.get_cluster(name=cluster_path)
  creds, _ = google.auth.default()
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  configuration = kubernetes.client.Configuration()
  configuration.host = f"https://{response.endpoint}"

  ca_cert_content = base64.b64decode(
      response.master_auth.cluster_ca_certificate
  )
  with tempfile.NamedTemporaryFile(delete=False) as ca_cert:
    ca_cert.write(ca_cert_content)
    configuration.ssl_ca_cert = ca_cert.name
  configuration.api_key_prefix["authorization"] = "Bearer"
  configuration.api_key["authorization"] = creds.token

  return kubernetes.client.ApiClient(configuration)


def get_core_api_client(
    project_id: str, region: str, cluster_name: str
) -> kubernetes.client.CoreV1Api:
  """Create a core API client for the given cluster."""
  client = get_authenticated_client(project_id, region, cluster_name)
  core_api = kubernetes.client.CoreV1Api(client)
  logging.info(
      "Successfully initialized k8s core API client from cluster response."
  )
  return core_api


def get_batch_api_client(
    project_id: str, region: str, cluster_name: str
) -> kubernetes.client.BatchV1Api:
  """Create a batch API client for the given cluster."""
  client = get_authenticated_client(project_id, region, cluster_name)
  batch_api = kubernetes.client.BatchV1Api(client)
  logging.info(
      "Successfully initialized k8s batch API client from cluster response."
  )
  return batch_api


def get_custom_objects_api_client(
    project_id: str, region: str, cluster_name: str
) -> kubernetes.client.CustomObjectsApi:
  """Create a custom objects API client for the given cluster."""
  client = get_authenticated_client(project_id, region, cluster_name)
  return kubernetes.client.CustomObjectsApi(client)


def list_workload_pods(
    core_api: kubernetes.client.CoreV1Api,
    workload_id: str,
    namespace: str = "default",
) -> kubernetes.client.V1PodList:
  """List all pods for the given workload (Job or JobSet)."""
  logging.info(
      f"Getting pods for workload_id: {workload_id} in namespace: {namespace}"
  )
  pods = core_api.list_namespaced_pod(
      namespace=namespace, label_selector=f"job-name={workload_id}"
  )
  if not pods.items:
    pods = core_api.list_namespaced_pod(
        namespace=namespace,
        label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
    )
  return pods


def get_workload_job(
    batch_api: kubernetes.client.BatchV1Api,
    workload_id: str,
    namespace: str = "default",
) -> Optional[kubernetes.client.V1Job]:
  """Get the Kubernetes Job object for a given workload."""
  logging.info(
      f"Getting job for workload_id: {workload_id} in namespace: {namespace}"
  )
  try:
    return batch_api.read_namespaced_job(name=workload_id, namespace=namespace)
  except kubernetes.client.exceptions.ApiException as e:
    logging.info(
        f"Direct job read failed for {workload_id} ({e}); trying label"
        " selector..."
    )

  try:
    jobs = batch_api.list_namespaced_job(
        label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
        namespace=namespace,
    )
    if not jobs.items:
      return None
    if len(jobs.items) > 1:
      logging.info(f"Got more than one job for workload_id: {workload_id}")
    return jobs.items[0]
  except kubernetes.client.exceptions.ApiException as e:
    logging.info(f"Could not list Kubernetes Jobs for {workload_id}: {e}")
    return None


def get_workload_jobset(
    custom_api: kubernetes.client.CustomObjectsApi,
    workload_id: str,
    namespace: str = "default",
) -> Optional[Dict[str, Any]]:
  """Get the Kubernetes JobSet CRD object for a given workload."""
  try:
    return custom_api.get_namespaced_custom_object(
        group="jobset.sigs.k8s.io",
        version="v1alpha2",
        namespace=namespace,
        plural="jobsets",
        name=workload_id,
    )
  except kubernetes.client.exceptions.ApiException as e:
    logging.info(f"Could not read JobSet {workload_id}: {e}")
    return None


def print_pod_logs(core_api: kubernetes.client.CoreV1Api, pod: kubernetes.client.V1Pod) -> None:
  """Prints logs for all containers in a pod."""
  try:
    for container in pod.spec.containers:
      try:
        logs = core_api.read_namespaced_pod_log(
            name=pod.metadata.name,
            namespace=pod.metadata.namespace,
            container=container.name,
        )
        logging.info(f"--- Logs for pod {pod.metadata.name}, container {container.name} ---")
        for line in logs.split("\n"):
          logging.info(line)
      except Exception as e:
        logging.info(f"Failed to fetch logs for {pod.metadata.name}:{container.name}: {e}")
  except Exception as e:
    logging.info(f"Failed to process pod containers: {e}")


def log_workload_pod_statuses(
    workload_id: str, pods: kubernetes.client.V1PodList
) -> None:
  """Logs the status of each retrieved pod and its containers."""
  if not pods or not pods.items:
    return

  logging.info(f"{f' Pod Statuses for Workload {workload_id} ':-^80}")
  for pod in pods.items:
    logging.info(f"Pod: {pod.metadata.name}, Status: {pod.status.phase}")
    if not pod.status.container_statuses:
      continue
    for container_status in pod.status.container_statuses:
      match container_status.state:
        case state if state.waiting:
          w = state.waiting
          logging.warning(
              f"  Container '{container_status.name}' WAITING. "
              f"Reason: {w.reason}. Message: {w.message}"
          )
        case state if state.terminated:
          t = state.terminated
          logging.error(
              f"  Container '{container_status.name}' TERMINATED. "
              f"Reason: {t.reason}. Exit Code: {t.exit_code}"
          )
  logging.info("-" * 80)


LOGGING_URL_FORMAT = (
    "https://console.cloud.google.com/logs/viewer"
    "?project={project}&resource=k8s_container"
    "&minLogLevel=0&expandAll=false"
    "&customFacets=&limitCustomFacetWidth=true"
    "&filters=text:job-name%3D{workload_id}"
)


@task.sensor(poke_interval=60, timeout=7200, mode="reschedule")
def wait_for_workload_start(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    namespace: str = "default",
) -> bool:
  """Wait for workload to start running."""
  core_api = get_core_api_client(project_id, region, cluster_name)
  pods = list_workload_pods(core_api, workload_id, namespace=namespace)
  log_workload_pod_statuses(workload_id, pods)

  if not pods.items:
    logging.info(f"Waiting for pods of workload: {workload_id} to be created.")
    return False

  for pod in pods.items:
    if pod.status.phase in ["Pending", "Unknown"]:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
      return False
    if pod.status.phase == "Failed":
      print_pod_logs(core_api, pod)
      url = LOGGING_URL_FORMAT.format(
          project=project_id,
          region=region,
          cluster=cluster_name,
          namespace=namespace,
          workload_id=workload_id,
      )
      raise AirflowFailException(
          f"Workload {workload_id} failed during startup with pod phase:"
          f" {pod.status.phase}. Link to logs: {url}"
      )

  logging.info("All pod(s) phase are ready to run.")
  return True


@task.sensor(poke_interval=60, timeout=18000, mode="reschedule")
def wait_for_workload_completion(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    namespace: str = "default",
) -> bool:
  """Wait for workload to finish successfully."""
  core_api = get_core_api_client(project_id, region, cluster_name)
  pods = list_workload_pods(core_api, workload_id, namespace=namespace)
  log_workload_pod_statuses(workload_id, pods)

  if not pods.items:
    batch_api = get_batch_api_client(project_id, region, cluster_name)
    job = get_workload_job(batch_api, workload_id, namespace=namespace)
    if job and job.status and job.status.conditions:
      conditions = job.status.conditions
      if any(
          c.type == "Failed" and getattr(c, "status", "True") != "False"
          for c in conditions
      ):
        url = LOGGING_URL_FORMAT.format(
            project=project_id,
            region=region,
            cluster=cluster_name,
            namespace=namespace,
            workload_id=workload_id,
        )
        raise AirflowFailException(
            f"Workload {workload_id} failed. Logs: {url}"
        )
      if any(
          c.type == "Complete" and getattr(c, "status", "True") != "False"
          for c in conditions
      ):
        logging.info(f"Workload {workload_id} Job completed successfully.")
        return True

    custom_api = get_custom_objects_api_client(project_id, region, cluster_name)
    jobset = get_workload_jobset(custom_api, workload_id, namespace=namespace)
    if jobset and "status" in jobset and "conditions" in jobset["status"]:
      conditions = jobset["status"]["conditions"]
      if any(
          c.get("type") == "Failed" and c.get("status", "True") != "False"
          for c in conditions
      ):
        url = LOGGING_URL_FORMAT.format(
            project=project_id,
            region=region,
            cluster=cluster_name,
            namespace=namespace,
            workload_id=workload_id,
        )
        raise AirflowFailException(
            f"Workload {workload_id} JobSet failed. Logs: {url}"
        )
      if any(
          c.get("type") == "Completed" and c.get("status", "True") != "False"
          for c in conditions
      ):
        logging.info(f"Workload {workload_id} JobSet completed successfully.")
        return True

    logging.info(f"No pods found for workload: {workload_id}")
    return False

  for pod in pods.items:
    if pod.status.phase in ["Pending", "Running", "Unknown"]:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
      return False
    if pod.status.phase == "Failed":
      print_pod_logs(core_api, pod)
      url = LOGGING_URL_FORMAT.format(
          project=project_id,
          region=region,
          cluster=cluster_name,
          namespace=namespace,
          workload_id=workload_id,
      )
      raise AirflowFailException(
          f"Workload {workload_id} failed with pod phase: {pod.status.phase}."
          f" Link to logs: {url}"
      )

  for pod in pods.items:
    if pod.status.container_statuses:
      for container_status in pod.status.container_statuses:
        if (
            container_status.state
            and container_status.state.terminated
            and container_status.state.terminated.exit_code != 0
        ):
          try:
            container_name = (
                container_status.name or pod.spec.containers[0].name
            )
            logs = core_api.read_namespaced_pod_log(
                name=pod.metadata.name,
                namespace=namespace,
                container=container_name,
            )
            for line in logs.split("\n"):
              logging.info(line)
          except kubernetes.client.exceptions.ApiException as e:
            logging.warning(
                f"Could not retrieve pod logs for {pod.metadata.name}: {e}"
            )
          url = LOGGING_URL_FORMAT.format(
              project=project_id,
              region=region,
              cluster=cluster_name,
              namespace=namespace,
              workload_id=workload_id,
          )
          raise AirflowFailException(
              f"Workload {workload_id} failed with container exit code "
              f"{container_status.state.terminated.exit_code}. Logs: {url}"
          )

  logging.info("All pod(s) phase are succeeded.")
  return True


@task_group
def run_job(
    body: Dict[str, Any],
    gcp: gcp_config.GCPConfig,
    gke_test_config: test_config.GpuGkeTest,
    cluster_name: str,
    job_create_timeout: datetime.timedelta,
    task_owner: str,
    gcs_location: str = "",
):
  """Run a batch job directly on a GKE cluster.

  Args:
    body: Dict that defines a Kubernetes `Job`.
    gcp: GCP config with the project name and zone of the GKE cluster.
    gke_test_config: Test config with the accelerator information of the GKE
      cluster.
    cluster_name: Name of the GCP cluster.
    job_create_timeout: Amount of time to wait for all pods to become active.
    task_owner: Task owner username or link.
    gcs_location: GCS path for all artifacts of the test.
  """

  @task
  def deploy_job(gcs_location):
    # Log required info for XLML PLX Dashboard
    composer.log_metadata_for_xlml_dashboard({
        "cluster_project": gcp.project_name,
        "zone": gcp.zone,
        "dataset_name": gcp.dataset_name.value,
        "composer_project": gcp.composer_project,
        "dataset_project": gcp.dataset_project,
        "cluster_name": cluster_name,
        "accelerator_type": gke_test_config.accelerator.machine_type,
    })

    body["spec"]["template"]["spec"]["containers"][0]["env"].append(
        {"name": "GCS_OUTPUT", "value": gcs_location}
    )
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    jobs_client = kubernetes.client.BatchV1Api(client)

    resp = jobs_client.create_namespaced_job(namespace="default", body=body)

    logging.info(f"response: {resp}")

    return resp.metadata.name

  @task.sensor(
      poke_interval=60,
      timeout=job_create_timeout.total_seconds(),
      mode="reschedule",
  )
  def wait_all_pods_ready(name: str):
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    batch_api = kubernetes.client.BatchV1Api(client)
    job = batch_api.read_namespaced_job(namespace="default", name=name)

    # TODO(wcromar): Handle other conditions (e.g. unschedulablility)
    logging.info(f"Job status: {job.status}")
    if job.status.failed:
      raise RuntimeError(f"Job has {job.status.failed} failed pods.")

    core_api = kubernetes.client.CoreV1Api(client)
    pod_label_selector = f"batch.kubernetes.io/job-name={name}"
    pods = core_api.list_namespaced_pod(
        namespace="default", label_selector=pod_label_selector
    )

    if len(pods.items) != body["spec"]["parallelism"]:
      logging.info("Waiting for all pods to be created...")
      return False

    return True

  @task(retries=6)
  def stream_logs(name: str):
    def _watch_pod(name, namespace) -> Optional[int]:
      logs_watcher = kubernetes.watch.Watch()

      logging.info(f"Waiting for pod {name} to start...")
      pod_watcher = kubernetes.watch.Watch()
      for event in pod_watcher.stream(
          core_api.list_namespaced_pod,
          namespace,
          field_selector=f"metadata.name={name}",
      ):
        status = event["object"].status
        logging.info(
            f'Pod {event["object"].metadata.name} status: {status.phase}'
        )
        if status.phase != "Pending":
          break

      logging.info(f"Streaming pod logs for {name}...")
      for line in logs_watcher.stream(
          core_api.read_namespaced_pod_log,
          name,
          namespace,
          _request_timeout=3600,
      ):
        logging.info(f"{name}] {line}")

      logging.warning(f"Lost logs stream for {name}.")

      pod = core_api.read_namespaced_pod(namespace="default", name=name)
      if pod.status.container_statuses:
        container_status = pod.status.container_statuses[0]
        if pod.status.container_statuses[0].state.terminated:
          exit_code = container_status.state.terminated.exit_code
          if exit_code:
            logging.error(f"Pod {name} had non-zero exit code {exit_code}")

          return exit_code

      logging.warning(f"Unknown status for pod {name}")
      return None

    # We need to re-authenticate if the stream_logs fail. This can happen when
    # the job runs for too long and the credential expire.
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    core_api = kubernetes.client.CoreV1Api(client)
    pod_label_selector = f"batch.kubernetes.io/job-name={name}"
    pods = core_api.list_namespaced_pod(
        namespace="default", label_selector=pod_label_selector
    )
    # TODO(piz): Use time.sleep may not be a good solution here. However, I
    # expect resources are all ready in wait_all_pods_ready stage. This just in
    # case authentication takes time. Check with Will for better solutions.
    time.sleep(30)
    if len(pods.items) != body["spec"]["parallelism"]:
      logging.info("Waiting for all pods to be re-connected...")
      raise PodsNotReadyError("pods are not ready after refreshing credential.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = []
      for pod in pods.items:
        f = executor.submit(
            _watch_pod, pod.metadata.name, pod.metadata.namespace
        )
        futures.append(f)

      # Wait for pods to complete, and exit with the first non-zero exit code.
      for f in concurrent.futures.as_completed(futures):
        try:
          # TODO(piz/wcromar): it looks like there is a delay between
          # as_completed and update of f.result(). exit_code can be None even
          # task is complete.
          exit_code = f.result()
        except kubernetes.client.ApiException as e:
          logging.error("Kubernetes error. Retrying...", exc_info=e)
          exit_code = None

        # Retry if status is unknown
        if exit_code is None:
          raise RuntimeError("unknown exit code")
        if exit_code:
          raise RuntimeError("Non-zero exit code")

  name = deploy_job.override(owner=task_owner)(gcs_location)
  chain(wait_all_pods_ready(name), stream_logs(name))


def zone_to_region(zone: str) -> str:
  zone_terms = zone.split("-")
  return zone_terms[0] + "-" + zone_terms[1]
