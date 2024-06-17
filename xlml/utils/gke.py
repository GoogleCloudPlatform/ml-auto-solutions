import base64
import concurrent.futures
import datetime
import logging
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

from airflow.decorators import task, task_group
import google.auth
import google.auth.transport.requests
from google.cloud import container_v1
import kubernetes

from xlml.apis import gcp_config

"""Utilities for GKE."""


class PodsNotReadyError(Exception):
  """Exception raised when pods are not ready within the expected timeout."""

  def __init__(self, message):
    super().__init__(message)


def get_authenticated_client(
    project_name: str, region: str, cluster_name: str
) -> kubernetes.client.ApiClient:
  container_client = container_v1.ClusterManagerClient()
  cluster_path = (
      f'projects/{project_name}/locations/{region}/clusters/{cluster_name}'
  )
  response = container_client.get_cluster(name=cluster_path)
  creds, _ = google.auth.default()
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  configuration = kubernetes.client.Configuration()
  configuration.host = f'https://{response.endpoint}'

  ca_cert_content = base64.b64decode(response.master_auth.cluster_ca_certificate)
  with tempfile.NamedTemporaryFile(delete=False) as ca_cert:
    ca_cert.write(ca_cert_content)
    configuration.ssl_ca_cert = ca_cert.name
  configuration.api_key_prefix['authorization'] = 'Bearer'
  configuration.api_key['authorization'] = creds.token

  return kubernetes.client.ApiClient(configuration)


@task_group
def run_job(
    body: Dict[str, Any],
    gcp: gcp_config.GCPConfig,
    cluster_name: str,
    job_create_timeout: datetime.timedelta,
    gcs_location: str = '',
):
  """Run a batch job directly on a GKE cluster.

  Args:
    body: Dict that defines a Kubernetes `Job`.
    gcp: GCP config with the project name and zone of the GKE cluster.
    cluster_name: Name of the GCP cluster.
    job_create_timeout: Amount of time to wait for all pods to become active.
  """

  @task
  def deploy_job(gcs_location):
    body['spec']['template']['spec']['containers'][0]['env'].append(
        {'name': 'GCS_OUTPUT', 'value': gcs_location}
    )
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    jobs_client = kubernetes.client.BatchV1Api(client)

    resp = jobs_client.create_namespaced_job(namespace='default', body=body)

    logging.info(f'response: {resp}')

    return resp.metadata.name

  @task.sensor(
      poke_interval=60,
      timeout=job_create_timeout.total_seconds(),
      mode='reschedule',
  )
  def wait_all_pods_ready(name: str):
    client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

    batch_api = kubernetes.client.BatchV1Api(client)
    job = batch_api.read_namespaced_job(namespace='default', name=name)

    # TODO(wcromar): Handle other conditions (e.g. unschedulablility)
    logging.info(f'Job status: {job.status}')
    if job.status.failed:
      raise RuntimeError(f'Job has {job.status.failed} failed pods.')

    core_api = kubernetes.client.CoreV1Api(client)
    pod_label_selector = f'batch.kubernetes.io/job-name={name}'
    pods = core_api.list_namespaced_pod(
        namespace='default', label_selector=pod_label_selector
    )

    if len(pods.items) != body['spec']['parallelism']:
      logging.info('Waiting for all pods to be created...')
      return False

    return True

  @task(retries=5)
  def stream_logs(name: str):

    def _watch_pod(name: str, namespace: str, start_line: int) -> Tuple[Optional[int], int]:
      """ Reads the gke workload log continuously

      Args:
        name: name of the workload.
        namespace: namespace of the pod that runs the workload.
        start_line: Omit the log message before start_line. This is used to skip
          messages that have been fetched previously.
      Returns:
        Tuple of exit code and line number of the log that the process is reading.
      """
      logs_watcher = kubernetes.watch.Watch()

      logging.info(f'Waiting for pod {name} to start...')
      pod_watcher = kubernetes.watch.Watch()
      for event in pod_watcher.stream(
          core_api.list_namespaced_pod,
          namespace,
          field_selector=f'metadata.name={name}',
      ):
        status = event['object'].status
        logging.info(
            f'Pod {event["object"].metadata.name} status: {status.phase}'
        )
        if status.phase != 'Pending':
          break

      logging.info(f'Streaming pod logs for {name}...')
      line_cnt = 0
      for line_cnt, line in enumerate(logs_watcher.stream(
          core_api.read_namespaced_pod_log,
          name,
          namespace,
          # This controls client side log reader timeout. Timeout can be triggered
          # if no new update line from the watcher stream.
          _request_timeout=3600,
      )):
        if line_cnt < start_line:
          continue
        logging.info(f'{name}] {line}')

      logging.warning(f'Lost logs stream for {name}.')

      # following line will trigger 401 exception if credential expire
      try:
        pod = core_api.read_namespaced_pod(namespace='default', name=name)
      except kubernetes.client.ApiException as e:
        status_code = e.status
        logging.warning(f'Kubernetes error (status code {status_code}).', exc_info=e)
        return (None, line_cnt)

      if pod.status.container_statuses:
        container_status = pod.status.container_statuses[0]
        if pod.status.container_statuses[0].state.terminated:
          exit_code = container_status.state.terminated.exit_code
          if exit_code:
            logging.error(f'Pod {name} had non-zero exit code {exit_code}')
          return (exit_code, line_cnt)

      logging.warning(f'Unknown status for pod {name}')
      return (None, line_cnt)


    exit_code = None
    start_line = 0
    while exit_code != 0:
      # We need to re-authenticate if the stream_logs fail. This can happen when
      # the job runs for too long and the credential expire.
      client = get_authenticated_client(gcp.project_name, gcp.zone, cluster_name)

      batch_api = kubernetes.client.BatchV1Api(client)
      core_api = kubernetes.client.CoreV1Api(client)
      pod_label_selector = f'batch.kubernetes.io/job-name={name}'
      pods = core_api.list_namespaced_pod(
          namespace='default', label_selector=pod_label_selector
      )
      # TODO(piz): Use time.sleep may not be a good solution here. However, I expect
      # resources are all ready in wait_all_pods_ready stage. This just in case
      # authentication takes time. Check with Will for better solutions.
      time.sleep(30)
      if len(pods.items) != body['spec']['parallelism']:
        logging.info('Waiting for all pods to be re-connected...')
        raise PodsNotReadyError('pods are not ready after refreshing credential.')

      with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for pod in pods.items:
          f = executor.submit(
              _watch_pod,
              pod.metadata.name,
              pod.metadata.namespace,
              start_line,
          )
          futures.append(f)

        for f in concurrent.futures.as_completed(futures):
          exit_code, start_line = f.result()
          if exit_code not in [None, 0]:
            raise RuntimeError("Potential infra issue with the host. Retry the task...")

  name = deploy_job(gcs_location)
  wait_all_pods_ready(name) >> stream_logs(name)
