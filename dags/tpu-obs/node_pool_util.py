"""
Manages the lifecycle of a GKE node pool and verifies its status as an Airflow DAG.
"""

import dataclasses
import enum
import logging
import random
import re
import time
from typing import List

from airflow import decorators
import subprocess
from airflow.providers.standard.operators.bash import BashOperator
from google import auth
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types
from googleapiclient import discovery


dataclass = dataclasses.dataclass
logger = logging.getLogger(__name__)


class Status(enum.Enum):
  """Enum for GKE node pool status."""
  RUNNING = enum.auto()
  PROVISIONING = enum.auto()
  STOPPING = enum.auto()
  RECONCILING = enum.auto()
  ERROR = enum.auto()
  UNKNOWN = enum.auto()

  @staticmethod
  def from_str(s: str) -> "Status":
      """Converts a string to a Status enum member."""
      status = Status.__members__.get(s)
      if status is None:
          logging.warning("Unknown status: %s", s)
          return Status.UNKNOWN
      return status


@dataclasses.dataclass
class Info():
  """Class to hold GKE node pool configuration parameters."""
  project_id: str
  cluster_name: str
  node_pool_name: str
  location: str
  node_locations: str
  machine_type: str
  num_nodes: int
  tpu_topology: str


@decorators.task
def create(info: Info, force_task_success: bool = False) -> BashOperator:
  """Creates the GKE node pool using gcloud command."""
  command_suffix = " 2>&1 || true" if force_task_success else ""

  command = f"""
                gcloud container node-pools create {info.node_pool_name} \\
                --project={info.project_id} \\
                --cluster={info.cluster_name} \\
                --location={info.location} \\
                --node-locations {info.node_locations} \\
                --num-nodes={info.num_nodes} \\
                --machine-type={info.machine_type} \\
                --tpu-topology={info.tpu_topology}{command_suffix}
        """
  process = None
  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
  logger.debug("STDOUT message: %s", process.stdout)
  logger.debug("STDERR message: %s", process.stderr)


@decorators.task
def delete(
    info: Info,
) -> BashOperator:
  """Deletes the GKE node pool using gcloud command."""
  command = f"""
                gcloud container node-pools delete {info.node_pool_name} \\
                --project {info.project_id} \\
                --cluster {info.cluster_name} \\
                --location {info.location} \\
                --quiet
        """

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
  logger.debug("STDOUT message: %s", process.stdout)
  logger.debug("STDERR message: %s", process.stderr)


def list_nodes(info: Info) -> List[str]:
  """Lists all VM instances (nodes) within the specified GKE node pool.

  This method queries the Google Cloud Container API and Compute API
  to retrieve details about the nodes belonging to the configured
  node pool. It parses instance group URLs to extract node names and zones.
  Args:
      info (Info): An instance of the Info class containing GKE node pool
                   configuration parameters.
  Returns:
      A dictionary where keys are node names (str) and values are
            the zones (str) where the nodes are located. Returns an empty
            dictionary if no nodes are found, if GCP clients are not
            initialized, or in case of a 404 HttpError (node pool not found).
  Raises:
      RuntimeError: If no instance groups or zone are found for the node pool.
  """
  credentials, _ = auth.default()
  container_client = discovery.build(
      "container", "v1", credentials=credentials, cache_discovery=False
  )
  compute_client = discovery.build(
      "compute", "v1", credentials=credentials, cache_discovery=False
  )

  nodepool_path = (
      f"projects/{info.project_id}/locations/{info.location}"
      f"/clusters/{info.cluster_name}/nodePools/{info.node_pool_name}"
  )
  nodepool = (
      container_client.projects().locations().clusters().nodePools()
      .get(name=nodepool_path)
      .execute()
  )

  instance_group = nodepool.get("instanceGroupUrls", [])
  if not instance_group:
    raise RuntimeError(
        f"No instance groups found for node pool {info.node_pool_name}."
    )

  node_names = []
  zone = None
  for url in instance_group:
    # Regex refined to be more specific to GCP instance group URLs
    match = re.search(
        r"zones/([\w-]+)/instanceGroupManagers/([\w-]+)", url
    )
    if not match:
      logging.warning("Could not parse instance group URL: %s", url)
      continue

    zone = match.group(1)
    ig_name = match.group(2)

    instances = (
        compute_client.instanceGroups()
        .listInstances(
            project=info.project_id,
            zone=zone,
            instanceGroup=ig_name,
            body={"instanceState": "ALL"},
        ).execute()
    )

    for instance_item in instances.get("items", []):
      instance_url = instance_item["instance"]
      # Regex refined to match GKE node names
      # (e.g., gke-cluster-node-xxxx)
      node_name = re.search(r"gke[\w-]+", instance_url).group()
      if node_name:
        node_names.append(node_name)
      else:
        logging.warning(
            "Could not extract node name from URL: %s", instance_url
        )
  if zone is None:
    raise RuntimeError(
        f"No zone found for node pool {info.node_pool_name}."
    )
  return node_names, zone


@decorators.task
def delete_node(info: Info):
  """Defines an Airflow task to delete a random node from the GKE node pool.

  This function uses Airflow's `@task` decorator to create a Python callable
  that will be executed as an Airflow task. The callable itself performs
  the node listing, selection, and deletion using `gcloud` commands.

  Args:
      info (Info): An instance of the Info class containing GKE node pool
                   configuration parameters.

  Returns:
      The decorated Airflow task object, ready to be included in a task flow.
  """

  nodes_list, zone = list_nodes(info)
  if not nodes_list:
    logging.warning(
        "No nodes found in node pool '%s'. No deletion will be performed.",
        info.node_pool_name,
    )
    return None  # Task succeeds, but no node deleted

  node_to_delete = random.choice(nodes_list)
  logging.info(
      "Randomly selected node for deletion: %s",
      node_to_delete,
  )

  command = f"""
      gcloud compute instances delete {node_to_delete} \\
          --project={info.project_id} \\
          --zone={zone} \\
          --quiet
      """

  process = None
  try:
    process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
    logging.info("Successfully deleted node %s", node_to_delete)
  except Exception as e:
    logger.error('Failed to run "%s": %s', " ".join(command), e)
    if process:
      # Subprocess hook stores the stdout + stderr in the `output` attribute
      logger.debug("STDOUT message: %s", process.stdout)
      logger.debug("STDERR message: %s", process.stderr)
    raise


def _query_status_metric(info: Info) -> Status:
  """Fetches the status time series data for this specific node pool."""
  project_name = f"projects/{info.project_id}"
  now = int(time.time())
  request = {
      "name": project_name,
      "filter": (
          'metric.type="kubernetes.io/node_pool/status" '
          f'resource.labels.project_id = "{info.project_id}" '
          f'resource.labels.cluster_name = "{info.cluster_name}" '
          f'resource.labels.node_pool_name = "{info.node_pool_name}"'
      ),
      "interval": types.TimeInterval({
          "end_time": {"seconds": now},
          "start_time": {"seconds": now - 300},
      }),
      "view": monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  }

  monitoring_client = monitoring_v3.MetricServiceClient()
  time_series_data = monitoring_client.list_time_series(request)
  records = []
  for series in time_series_data:
    np_status = series.metric.labels.get("status", "unknown").upper()
    for point in series.points:
      end_ts_dt = point.interval.end_time
      records.append((end_ts_dt, np_status))
  if not records:
    logging.info("No records found yet. Retrying in 60s...")
    return Status.UNKNOWN

  _, latest_status = max(records, key=lambda r: r[0])

  return Status.from_str(latest_status)


@decorators.task.sensor(poke_interval=60, timeout=600, mode="reschedule")
def wait_for_status(
    info: Info,
    status: Status,
    **context,
) -> None:
  """Waits for the node pool to enter the target status."""
  # Consistent with Airflow's default timeout for sensor tasks.
  timeout = context["task"].timeout
  logging.info(
      "Waiting for node pool '%s' status to become '%s' within %s"
      " seconds...",
      info.node_pool_name,
      status.name,
      timeout,
  )

  latest_status = _query_status_metric(info)
  return latest_status == status


