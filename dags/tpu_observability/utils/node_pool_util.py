"""Utility functions for managing GKE node pools."""

import dataclasses
import enum
import json
import logging
import random
import re
import subprocess
import time
from typing import List

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types


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
class Info:
  """Encapsulates information related to a GKE node pool and represents a specific node pool."""

  project_id: str
  cluster_name: str
  node_pool_name: str
  location: str
  node_locations: str
  machine_type: str
  num_nodes: int
  tpu_topology: str


@task
def create(node_pool: Info, ignore_failure: bool = False) -> None:
  """Creates a GKE node pool by the given node pool information."""

  command = (
      f"gcloud container node-pools create {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      f"--node-locations {node_pool.node_locations} "
      f"--num-nodes={node_pool.num_nodes} "
      f"--machine-type={node_pool.machine_type} "
      f"--tpu-topology={node_pool.tpu_topology}"
  )
  if ignore_failure:
    command += " 2>&1 || true"

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
  logger.info("STDOUT message: %s", process.stdout)
  logger.info("STDERR message: %s", process.stderr)


@task
def delete(node_pool: Info) -> None:
  """Deletes the GKE node pool using gcloud command."""

  command = (
      f"gcloud container node-pools delete {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      "--quiet"
  )

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
  logger.info("STDOUT message: %s", process.stdout)
  logger.info("STDERR message: %s", process.stderr)


def list_nodes(node_pool: Info) -> List[str]:
  """Lists all node names in the specified GKE node pool.

  It queries GKE and Compute APIs and parses instance group URLs
  to extract VM instance names.

  Args:
      node_pool: An instance of the Info class that encapsulates the
        configuration and metadata of a GKE node pool.
  Returns:
      A list of node names within the specified GKE node pool.
  Raises:
      RuntimeError: If no instance groups or zone are found for the node pool.
  """
  instance_group_urls_key = "instanceGroupUrls"
  process = subprocess.run(
      (
          f"gcloud container node-pools describe {node_pool.node_pool_name} "
          f"--project={node_pool.project_id} "
          f"--cluster={node_pool.cluster_name} "
          f"--location={node_pool.location} "
          f"--format='json({instance_group_urls_key})'"
      ),
      shell=True,
      check=True,
      capture_output=True,
      text=True,
  )

  instance_group_urls_val = json.loads(process.stdout).get(
      instance_group_urls_key, []
  )
  if not instance_group_urls_val:
    raise AirflowFailException(
        f"No instance groups found for node pool {node_pool.node_pool_name}."
    )

  node_names = []

  for url in instance_group_urls_val:
    # Extract the {instance_group_name} segments from an URL:
    # https://www.googleapis.com/compute/v1/projects/tpu-prod-env-one-vm/zones/asia-northeast1-b/instanceGroups/gke-yuna-xpk-v6e-2-yuna-xpk-v6e-2-np--b3a745c7-grp
    # in which, `gke-yuna-xpk-v6e-2-yuna-xpk-v6e-2-np--b3a745c7-grp`
    # is the of the instance group
    match = re.search(r"instanceGroupManagers/([\w-]+)", url)
    if not match:
      logging.warning("Could not parse instance group URL: %s", url)
      continue

    instance_group_name = match.group(1)

    process = subprocess.run(
        (
            "gcloud compute instance-groups list-instances"
            f" {instance_group_name} "
            f"--project={node_pool.project_id} "
            f"--zone={node_pool.node_locations} "
            "--format='json(instance)'"
        ),
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    instances = json.loads(process.stdout)

    for instance_item in instances:
      instance_url = instance_item["instance"]
      # Extract the {node_name} segments from an URL like this:
      # https://www.googleapis.com/compute/v1/projects/<project>/zones/<zone>/instances/<node_name>
      # in which, `gke-tpu-b3a745c7-08bk` is the name of the node
      node_name = re.search(r"gke[\w-]+", instance_url).group()
      if node_name:
        node_names.append(node_name)
      else:
        logging.warning(
            "Could not extract node name from URL: %s", instance_url
        )
  return node_names


@task
def delete_one_random_node(node_pool: Info) -> None:
  """Delete one random node from the specified GKE node pool.

  This function first lists all nodes under the given node pool,
  then randomly selects one node and deletes it.

  Args:
      node_pool: An instance of the Info class that encapsulates
        the configuration and metadata of a GKE node pool.

  Raises:
      ValueError: If no nodes are found in the specified node pool.
  """

  nodes_list = list_nodes(node_pool)
  if not nodes_list:
    raise AirflowFailException(
        f"No nodes found in node pool '{node_pool.node_pool_name}'. "
        "Cannot proceed with node deletion."
    )

  node_to_delete = random.choice(nodes_list)
  logging.info(
      "Randomly selected node for deletion: %s",
      node_to_delete,
  )

  command = (
      f"gcloud compute instances delete {node_to_delete} "
      f"--project={node_pool.project_id} "
      f"--zone={node_pool.node_locations} "
      "--quiet"
  )

  process = subprocess.run(
      command, shell=True, check=True, capture_output=True, text=True
  )
  logger.info("STDOUT message: %s", process.stdout)
  logger.info("STDERR message: %s", process.stderr)


def _query_status_metric(node_pool: Info) -> Status:
  """Queries the latest status of the specified GKE node pool.

  This function retrieves the status by querying the metric
  "kubernetes.io/node_pool/status" via the Google Cloud Monitoring API.

  Args:
      node_pool: An instance of the Info class that encapsulates
                   the configuration and metadata of a GKE node pool.

  Returns:
      A `Status` enum representing the latest status of the node pool.
  """
  monitoring_client = monitoring_v3.MetricServiceClient()
  project_name = f"projects/{node_pool.project_id}"
  now = int(time.time())
  request = monitoring_v3.ListTimeSeriesRequest(
      name=project_name,
      filter=(
          'metric.type="kubernetes.io/node_pool/status" '
          f'resource.labels.project_id = "{node_pool.project_id}" '
          f'resource.labels.cluster_name = "{node_pool.cluster_name}" '
          f'resource.labels.node_pool_name = "{node_pool.node_pool_name}"'
      ),
      interval=types.TimeInterval({
          "end_time": {"seconds": now},
          # Metrics are sampled every 60s and stored in the GCP backend,
          # but it may take up to 2 minutes for the data to become
          # available on the client side.
          # Therefore, a longer time interval is necessary.
          # A 5-minute window is an arbitrary but sufficient choice to
          # ensure we can retrieve the latest metric data.
          "start_time": {"seconds": now - 300},
      }),
      view="FULL",
  )

  # A single query to the Monitoring API can return multiple TimeSeries objects,
  # especially if the 'status' label changed within the time window (e.g., from
  # 'PROVISIONING' to 'RUNNING').
  #
  # To robustly find the absolute latest status, this block first aggregates all
  # data points from all series into a single flat list ('records'). It then
  # finds the record with the maximum timestamp from this list to ensure the
  # true latest status is identified.
  time_series_data = monitoring_client.list_time_series(request)
  records = []
  for series in time_series_data:
    np_status = series.metric.labels.get("status", "unknown").upper()
    for point in series.points:
      end_ts_dt = point.interval.end_time
      records.append((end_ts_dt, np_status))
  if not records:
    return Status.UNKNOWN

  _, latest_status = max(records, key=lambda r: r[0])

  return Status.from_str(latest_status)


@task.sensor(poke_interval=60, timeout=600, mode="reschedule")
def wait_for_status(
    node_pool: Info,
    status: Status,
    **context,
) -> bool:
  """Waits for the node pool to enter the target status.

  This is a task waits for the node pool to enter the target status by querying
  the status metric and comparing it with the expected status.
  defaults task poke interval to 60 seconds and timeout to 600 seconds.

  Args:
      node_pool: An instance of the Info class that encapsulates
        the configuration and metadata of a GKE node pool.
      status: The target status to wait for, represented as a `Status` enum.
      context: The Airflow context dictionary, which includes task metadata.
  Returns:
      A boolean indicating whether the node pool has reached the target status.
  """
  timeout = context["task"].timeout
  logging.info(
      "Waiting for node pool '%s' status to become '%s' within %s"
      " seconds...",
      node_pool.node_pool_name,
      status.name,
      timeout,
  )

  latest_status = _query_status_metric(node_pool)
  return latest_status == status
