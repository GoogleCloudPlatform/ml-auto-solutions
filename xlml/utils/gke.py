import base64
import concurrent.futures
import logging
import tempfile
from typing import Dict, TypeAlias

from airflow.decorators import task
import google.auth
import google.auth.transport.requests
from google.cloud import container_v1
import kubernetes

from xlml.apis import gcp_config


def get_authenticated_client(gcp: gcp_config.GCPConfig, cluster_name: str) -> kubernetes.client.ApiClient:
  container_client = container_v1.ClusterManagerClient()
  cluster_path = f"projects/{gcp.project_name}/locations/{gcp.zone}/clusters/{cluster_name}"
  response = container_client.get_cluster(name=cluster_path)
  creds, _ = google.auth.default()
  auth_req = google.auth.transport.requests.Request()
  creds.refresh(auth_req)
  configuration = kubernetes.client.Configuration()
  configuration.host = f"https://{response.endpoint}"
  with tempfile.NamedTemporaryFile(delete=False) as ca_cert:
    ca_cert.write(base64.b64decode(response.master_auth.cluster_ca_certificate))
  configuration.ssl_ca_cert = ca_cert.name
  configuration.api_key_prefix["authorization"] = "Bearer"
  configuration.api_key["authorization"] = creds.token

  return  kubernetes.client.ApiClient(configuration)


@task
def deploy_job(body: Dict[str, object], gcp: gcp_config.GCPConfig):
  client = get_authenticated_client(gcp, 'wcromar-test-cluster')

  jobs_client = kubernetes.client.BatchV1Api(client)
  resp = jobs_client.create_namespaced_job(namespace='default', body=body)

  print(resp)
  print(type(resp))

  core_v1 = kubernetes.client.CoreV1Api(client)

  pod_label_selector = "controller-uid=" + resp.metadata.uid
  pods = core_v1.list_namespaced_pod(namespace='default', label_selector=pod_label_selector)
  print(pods)


  def _watch_pod(name, namespace):
    logs_watcher = kubernetes.watch.Watch()

    while True:
      logging.info('Waiting for pod %s to start...', name)
      pod_watcher = kubernetes.watch.Watch()
      for event in pod_watcher.stream(core_v1.list_namespaced_pod, namespace,
                                      field_selector=f'metadata.name={name}'):
        status = event['object'].status
        logging.info('Pod %s status: %s', event['object'].metadata.name, status.phase)
        if status.phase != 'Pending':
          break

      if status.container_statuses:
        container_status = status.container_statuses[0]
        if status.container_statuses[0].state.terminated:
          exit_code = container_status.state.terminated.exit_code
          if exit_code:
            logging.error('Pod %s had non-zero exit code %d', name, exit_code)

          return exit_code

      logging.info('Streaming pod logs for %s...', name)
      for line in logs_watcher.stream(core_v1.read_namespaced_pod_log,
                                      name, namespace, _request_timeout=3600):
        logging.info('%s] %s', name, line)

      logging.warning('Lost logs stream for %s.', name)

  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for pod in pods.items:
      f = executor.submit(_watch_pod, pod.metadata.name, pod.metadata.namespace)
      futures.append(f)

    # Wait for pods to complete, and exit with the first non-zero exit code.
    for f in concurrent.futures.as_completed(futures):
      exit_code = f.result()
      if exit_code:
        raise RuntimeError(f'Non-zero exit code: {exit_code}')


