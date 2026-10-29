"""Utilities for GKE."""

import base64
import concurrent.futures
import dataclasses
import datetime
import logging
import re
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

from airflow.decorators import task, task_group
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
import google.auth
import google.auth.transport.requests
from google.cloud import compute_v1
from google.cloud import container_v1
import kubernetes
import urllib3

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


def print_container_logs(
    core_api: kubernetes.client.CoreV1Api,
    namespace: str,
    pod_name: str,
    container_name: str,
) -> None:
  """Prints the full logs of a single container."""
  try:
    # `tail_lines` is intentionally omitted so that the whole log is returned.
    # `_preload_content=False` streams the logs instead of buffering the whole
    # response in memory. It returns the raw `urllib3.HTTPResponse`, which
    # yields one `bytes` line per iteration, hence the decoding below.
    response = core_api.read_namespaced_pod_log(
        name=pod_name,
        namespace=namespace,
        container=container_name,
        _preload_content=False,
    )
  except (
      kubernetes.client.exceptions.ApiException,
      # The Kubernetes client only converts SSL errors into `ApiException`, so
      # connection errors surface as raw urllib3 errors.
      urllib3.exceptions.HTTPError,
  ) as e:
    logging.warning(
        "Could not retrieve logs for pod %s, container %s: %s",
        pod_name,
        container_name,
        e,
    )
    return

  logging.info(
      "--- Logs for pod %s, container %s ---", pod_name, container_name
  )
  try:
    for line in response:
      logging.info(line.decode("utf-8", "replace").rstrip("\n"))
  except urllib3.exceptions.HTTPError as e:
    logging.warning(
        "Log stream of pod %s, container %s was interrupted: %s",
        pod_name,
        container_name,
        e,
    )
  finally:
    # The connection is not returned to the pool automatically when the
    # response is streamed.
    response.release_conn()


def print_pod_logs(
    core_api: kubernetes.client.CoreV1Api,
    pod: kubernetes.client.V1Pod,
) -> None:
  """Prints the full logs of all containers in a pod."""
  containers = pod.spec.containers if pod.spec else None
  if not containers:
    logging.warning("No containers found for pod %s.", pod.metadata.name)
    return

  for container in containers:
    print_container_logs(
        core_api,
        namespace=pod.metadata.namespace,
        pod_name=pod.metadata.name,
        container_name=container.name,
    )


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


@dataclasses.dataclass
class WorkloadState:
  """The pods of a workload bucketed by phase, collected in a single pass.

  `failed_containers` holds every container that terminated with a non-zero
  exit code, regardless of the phase of its pod.
  """

  pending: list[kubernetes.client.V1Pod] = dataclasses.field(
      default_factory=list
  )
  running: list[kubernetes.client.V1Pod] = dataclasses.field(
      default_factory=list
  )
  unknown: list[kubernetes.client.V1Pod] = dataclasses.field(
      default_factory=list
  )
  failed: list[kubernetes.client.V1Pod] = dataclasses.field(
      default_factory=list
  )
  succeeded: list[kubernetes.client.V1Pod] = dataclasses.field(
      default_factory=list
  )
  failed_containers: list[
      tuple[kubernetes.client.V1Pod, kubernetes.client.V1ContainerStatus]
  ] = dataclasses.field(default_factory=list)

  @property
  def active_pods(self) -> list[kubernetes.client.V1Pod]:
    """Pods that may still change phase."""
    return self.pending + self.running + self.unknown

  @property
  def not_completed_pods(self) -> list[kubernetes.client.V1Pod]:
    """Pods that are not Succeeded."""
    return self.active_pods + self.failed


def collect_pod_state(pods: kubernetes.client.V1PodList) -> WorkloadState:
  """Buckets the pods by phase and gathers their failed containers."""
  state = WorkloadState()
  buckets = {
      "Pending": state.pending,
      "Running": state.running,
      "Unknown": state.unknown,
      "Failed": state.failed,
      "Succeeded": state.succeeded,
  }
  for pod in pods.items:
    # A phase outside the five Kubernetes defines is treated as Unknown.
    buckets.get(pod.status.phase, state.unknown).append(pod)
    for container_status in pod.status.container_statuses or []:
      terminated = container_status.state and container_status.state.terminated
      if terminated and terminated.exit_code != 0:
        state.failed_containers.append((pod, container_status))
  return state


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

  states = collect_pod_state(pods)
  not_started = states.pending + states.unknown
  if not_started:
    for pod in not_started:
      logging.info(f"Pod {pod.metadata.name} is in phase {pod.status.phase}")
    return False
  if states.failed:
    for pod in states.failed:
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
        f" Failed. Link to logs: {url}"
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
  # The link is the same whatever state the workload is in.
  url = LOGGING_URL_FORMAT.format(
      project=project_id,
      region=region,
      cluster=cluster_name,
      namespace=namespace,
      workload_id=workload_id,
  )

  if not pods.items:
    batch_api = get_batch_api_client(project_id, region, cluster_name)
    job = get_workload_job(batch_api, workload_id, namespace=namespace)
    if job and job.status and job.status.conditions:
      conditions = job.status.conditions
      if any(
          c.type == "Failed" and getattr(c, "status", "True") != "False"
          for c in conditions
      ):
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

  # Judge the workload from the state of all its pods at once, so the verdict
  # does not depend on the order the API lists them in.
  #
  # A Failed pod does not necessarily mean the workload failed: pod damaged
  # (e.g. the node-interruption tests delete its node) -> k8s restarts the
  # job -> the damaged pod gets recycled. So only judge once no pod is active.
  states = collect_pod_state(pods)
  if states.active_pods:
    # Still running or being restarted; keep polling.
    logging.info(
        f"{len(states.not_completed_pods)} of {len(pods.items)} pod(s) not"
        f" completed, {len(states.failed)} of them failed. Logs: {url}"
    )
    return False
  if states.failed:
    # Nothing left to restart: the Failed pods are final.
    for pod in states.failed:
      print_pod_logs(core_api, pod)
    raise AirflowFailException(
        f"Workload {workload_id} failed with {len(states.failed)} of"
        f" {len(pods.items)} pod(s) in phase Failed. Link to logs: {url}"
    )

  # Every pod is Succeeded. Confirm no container exited non-zero; the phases
  # are final at this point, so a bad exit code is a failure rather than
  # something to poll for again.
  error_messages = []
  for pod, container_status in states.failed_containers:
    print_container_logs(
        core_api,
        namespace=namespace,
        pod_name=pod.metadata.name,
        container_name=container_status.name,
    )
    error_messages.append(
        f"{pod.metadata.name}/{container_status.name} exited with code"
        f" {container_status.state.terminated.exit_code}"
    )
  if error_messages:
    raise AirflowFailException(
        f"Workload {workload_id} failed: {', '.join(error_messages)}."
        f" Logs: {url}"
    )

  # Fetch logs for successful pods before returning
  for pod in states.succeeded:
    print_pod_logs(core_api, pod)

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


# Compute Engine zones look like `europe-west4-b`; regions like `europe-west4`.
_ZONE_PATTERN = re.compile(r"^[a-z]+-[a-z]+\d+-[a-z]$")


def is_zone(location: str) -> bool:
  """Whether `location` is a Compute Engine zone (rather than a region)."""
  return bool(location) and _ZONE_PATTERN.match(location) is not None


@task.sensor(poke_interval=3, timeout=3600, mode="reschedule")
def wait_for_workload_reach_step(
    project_id: str,
    region: str,
    cluster_name: str,
    workload_id: str,
    expect_reach_to_step: str,
    namespace: str = "default",
) -> bool:
  """
  Watch any given training pod, check the given step is already reach before
  deleting a node
  """
  core_api = get_core_api_client(project_id, region, cluster_name)
  pods = list_workload_pods(core_api, workload_id, namespace=namespace)

  if not pods.items:
    logging.info("No pods found for workload selector: %s.", workload_id)
    return False

  states = collect_pod_state(pods)

  # Report a dead pod before waiting on the others: a Pending peer must not
  # keep the sensor polling until its timeout when the workload has already
  # failed.
  if states.failed:
    # Don't keep retrying if the pod has failed
    raise AirflowFailException(
        f"Bad pod phase: Failed ({len(states.failed)} of {len(pods.items)}"
        " pod(s))"
    )
  if states.unknown:
    raise RuntimeError(f"Bad pod phase: {states.unknown[0].status.phase}")

  if states.pending:
    logging.info("Some of the pods is still pending. Waiting to start")
    return False

  if len(states.running) == len(pods.items):
    # Pick last one running pod
    pod = states.running[-1]
    current_step = _latest_completed_step(core_api, pod)
    if current_step is not None and current_step >= int(expect_reach_to_step):
      logging.info(
          "Reached to the expected step %s. Current step is %s.",
          expect_reach_to_step,
          current_step,
      )
      return True

  logging.info("Waiting for reaching expected step %s.", expect_reach_to_step)

  return False


# Trailing log lines scanned for the latest `completed step:` marker on each
# poll. MaxText logs one marker per step, so this spans the chatty stretches
# around checkpoint saves; the full log is only read if the tail has none.
STEP_LOG_TAIL_LINES = 1000
_COMPLETED_STEP_PATTERN = re.compile(r"completed step: (\d+)")


def _latest_completed_step(
    core_api: kubernetes.client.CoreV1Api, pod: kubernetes.client.V1Pod
) -> Optional[int]:
  """Returns the last step `pod` logged as completed, or None if none yet.

  The sensor polls every few seconds, so this reads only the tail of the log
  rather than the whole log each time, falling back to the full log when the
  tail holds no step marker.
  """
  for tail_lines in (STEP_LOG_TAIL_LINES, None):
    log_kwargs = {} if tail_lines is None else {"tail_lines": tail_lines}
    logs = core_api.read_namespaced_pod_log(
        name=pod.metadata.name, namespace=pod.metadata.namespace, **log_kwargs
    )
    completed_step_matches = _COMPLETED_STEP_PATTERN.findall(logs)
    if completed_step_matches:
      return int(completed_step_matches[-1])
  return None


# Workload pod names are `<workload>-slice-job-<slice>-<index>-<hash>` on XPK
# and `<workload>-main-job-<slice>-<index>-<hash>` on gcluster.
_WORKLOAD_POD_NAME_PATTERN = re.compile(r"-(?:slice|main)-job-(\d+)-(\d+)-\w+")


def extract_numbers(pod_name: str) -> Tuple[int, int]:
  """Extract slice and pod numbers from pod name."""
  match = _WORKLOAD_POD_NAME_PATTERN.search(pod_name)
  if match:
    return int(match.group(1)), int(match.group(2))
  return (0, 0)


def _find_target_pod_node(
    project_id: str,
    region: str,
    cluster_name: str,
    workload_id: str,
    last_node: bool = False,
    namespace: str = "default",
    cluster_location: Optional[str] = None,
) -> str:
  """find the node name for the workload.

  Args:
    cluster_location: The location the caller addressed the cluster by. Used
      as the node's zone only if the node carries no zone label and the
      location is itself a zone (zonal cluster); a region is never used.
  """
  core_api = get_core_api_client(project_id, region, cluster_name)
  pods = list_workload_pods(core_api, workload_id, namespace=namespace)

  pod_node_pairs = [
      (pod.metadata.name, pod.spec.node_name)
      for pod in pods.items
      if pod.status.phase == "Running"
      and _WORKLOAD_POD_NAME_PATTERN.search(pod.metadata.name)
  ]
  if not pod_node_pairs:
    raise AirflowFailException(
        f"No running pods found for workload {workload_id} matching pattern"
        f" {_WORKLOAD_POD_NAME_PATTERN.pattern}."
    )

  # Find the pod with the highest slice and pod numbers.
  # Sort by slice number, then by pod number, and get the last (highest) one
  sorted_pairs = sorted(pod_node_pairs, key=lambda x: extract_numbers(x[0]))
  target_pod, target_node = sorted_pairs[0]
  if last_node:
    target_pod, target_node = sorted_pairs[-1]

  # The node's own topology label is the authoritative source for its zone.
  # The cluster location is only a substitute when it is itself a zone: for a
  # regional cluster it is a region (gcluster clusters carry a region in their
  # zone field), which Compute Engine rejects, so in that case fail rather
  # than guess.
  node = core_api.read_node(name=target_node)
  labels = (node.metadata.labels or {}) if node.metadata else {}
  target_zone = labels.get("topology.kubernetes.io/zone") or labels.get(
      "failure-domain.beta.kubernetes.io/zone"
  )
  if not target_zone and cluster_location and is_zone(cluster_location):
    logging.warning(
        "Node %s (pod %s) has no zone label; using the zonal cluster"
        " location %s.",
        target_node,
        target_pod,
        cluster_location,
    )
    target_zone = cluster_location
  if not target_zone:
    raise AirflowFailException(
        f"Node {target_node} (pod {target_pod}) has no zone label and the"
        f" cluster location {cluster_location!r} is not a zone; refusing to"
        " delete without a verified zone."
    )

  logging.info("Identified Pod for node deletion:")
  logging.info(f"  Pod Name:   {target_pod}")
  logging.info(f"  Node Name:  {target_node}")
  logging.info(f"  Node Zone:  {target_zone}")
  logging.info("-" * 72)

  delete_info = {
      "pod": target_pod,
      "node": target_node,
      "zone": target_zone,
  }
  return delete_info


# Upper bound on how long `delete_node` waits for Compute Engine to finish
# deleting the instance. Node deletion normally completes within minutes.
NODE_DELETE_TIMEOUT_SECONDS = 600


@task
def delete_node(
    cluster_name: str,
    workload_id: str,
    zone: str,
    project: str,
    dry_run: bool = False,
    last_node: bool = False,
    namespace: str = "default",
) -> None:
  """Deletes one of the workload's nodes and waits for the deletion to finish.

  `zone` is the location the cluster is addressed by. It is a region for
  gcluster clusters, so the instance zone comes from the node's own label and
  `zone` is only used as a fallback when it is itself a zone.
  """
  delete_info = _find_target_pod_node(
      project,
      zone_to_region(zone),
      cluster_name,
      workload_id,
      last_node,
      namespace=namespace,
      cluster_location=zone,
  )
  node_name = delete_info["node"]
  node_zone = delete_info["zone"]
  # Delete the specified compute instance.
  if dry_run:
    logging.info(
        f"DRY RUN: Would delete node: {node_name}"
        f"in zone: {node_zone} (project: {project})"
    )
    return

  logging.info(f"Proceeding to delete node: {node_name}")
  try:
    # Initialize the Compute Engine client
    instances_client = compute_v1.InstancesClient()

    # Delete the instance
    operation = instances_client.delete(
        project=project, zone=node_zone, instance=node_name
    )
    logging.info(f"Deletion operation started for node: {node_name}")
    logging.info(f"Operation: {operation.name}")

    # `delete` only submits a long-running operation. Wait for it so the
    # interruption flow does not continue with the node still alive, and
    # surface a failed deletion instead of reporting success.
    operation.result(timeout=NODE_DELETE_TIMEOUT_SECONDS)
    if operation.error_code:
      raise AirflowFailException(
          f"Deleting node {node_name} failed: [{operation.error_code}]"
          f" {operation.error_message}"
      )
    for warning in getattr(operation, "warnings", None) or []:
      logging.warning(
          f"Deleting node {node_name}: {warning.code}: {warning.message}"
      )
    logging.info(f"Deleted node: {node_name}")
  except Exception:
    logging.error(f"Error deleting node {node_name}")
    raise
