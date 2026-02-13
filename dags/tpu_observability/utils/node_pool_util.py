# Copyright 2025 Google LLC
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

"""Utility functions for managing GKE node pools."""

import dataclasses
import datetime
import enum
import json
import logging
import random
import re

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import monitoring_v3

from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.gcp_util import list_time_series
from dags.tpu_observability.utils import subprocess_util as subprocess
from xlml.apis import gcs
from xlml.utils import composer


NODE_POOL_SELECTOR_KEY = "tpu-observability/workload"
"""The label key for binding JobSet workloads to specific GKE node pools.

This key is used as a Kubernetes node label to ensure pods are scheduled
on the correct node pool. It is applied to both:
- GKE node pools via `--node-labels` during creation
- JobSet YAML via `nodeSelector` to target the labeled nodes
"""


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
  """
  Encapsulates information related to a GKE node pool and represents a specific
  node pool.
  """

  project_id: str = None
  cluster_name: str = None
  node_pool_name: str = None
  region: str = None
  zone: str = None
  location: str = None
  node_locations: str = None
  machine_type: str = None
  num_nodes: int = None
  tpu_topology: str = None
  reservation: str = None
  node_pool_selector: str = None


@task
def build_node_pool_info_from_gcs_yaml(
    gcs_path: str, dag_name: str, is_prod: bool = True, **overrides
) -> Info:
  """Builds a node_pool.Info instance by merging configurations.

  This task merges values from the 'common' section of the provided dag_config,
  a DAG-specific section (given by dag_name), and any provided overrides.
  Only fields that exist in the node_pool.Info dataclass are included.

  Args:
      gcs_path: The GCS path to the DAG configuration YAML file.
      dag_name: The top-level key in the YAML representing the
        specific DAG's configuration (e.g., 'dag_gke_node_pool_label_update').
      is_prod: Boolean indicating whether to load the 'prod' or 'dev' environment
        from the YAML configuration.
      **overrides: Additional key-value pairs to override any settings
        from the YAML.

  Returns:
      An initialized node_pool.Info dataclass instance.
  """
  current_env = "prod" if is_prod else "dev"
  config = gcs.load_yaml_from_gcs(gcs_path)

  env_cfg = config.get("env", {}).get(current_env, {})
  dag_cfg = config.get("dag", {}).get(dag_name, {})

  known_fields = {f.name for f in dataclasses.fields(Info)}

  def warn_unknown(name: str, d: dict) -> None:
    unknown = [k for k in d.keys() if k not in known_fields]
    if unknown:
      logging.warning(f"Ignoring unknown fields in {name}: {unknown}")

  warn_unknown("common section in yaml", env_cfg)
  warn_unknown(f"{dag_name} section in yaml", dag_cfg)
  warn_unknown("override fields", overrides)

  # --- Configuration Merging Logic ---
  # Priority Order (lowest to highest):
  # 1. 'env' section: Base defaults for the environment (prod vs dev).
  # 2. 'dag' section: Overrides specific to a DAG name.
  # 3. 'overrides' dict: Code-level overrides passed into the task.

  # Initialize with lowest priority: Environment-level defaults
  merged = {k: v for k, v in env_cfg.items() if k in known_fields}

  # Apply medium priority: DAG-specific config (overwrites env-level values)
  for k, v in dag_cfg.items():
    if k in known_fields and v is not None:
      merged[k] = v

  # Apply highest priority: Manual task overrides (overwrites both above)
  for k, v in overrides.items():
    if k in known_fields and v is not None:
      merged[k] = v

  return Info(**merged)


@task
def copy_node_pool_info_with_override(info: Info, **overrides) -> Info:
  """Copies a node_pool.Info instance and applies overrides.

  Args:
      info: The base node_pool.Info instance to copy.
      **overrides: Key-value pairs to override fields in the copied Info.

  Returns:
      A new node_pool.Info instance with overrides applied.
  """
  known = {f.name for f in dataclasses.fields(Info)}

  unknown = [k for k in overrides if k not in known]
  if unknown:
    logging.warning(f"Ignoring unknown fields in overrides: {unknown}")

  cfg = {k: v for k, v in overrides.items() if k in known and v is not None}
  if not cfg:
    return info  # Nothing to change.

  # return dataclasses.replace(info, **cfg)
  replaced_info = dataclasses.replace(info, **cfg)
  logging.info(
      f"Created a new node_pool.Info instance with overrides: {replaced_info}"
  )
  return replaced_info


def _node_pool_exists(node_pool: Info) -> bool:
  check_cmd = (
      f"gcloud container node-pools describe {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      f"--format='value(name)'"
  )
  try:
    subprocess.run_exec(check_cmd)
    return True
  except Exception:
    return False


@task
def create(
    node_pool: Info,
    ignore_failure: bool = False,
) -> None:
  """Creates a GKE node pool by the given node pool information.

  Args:
    node_pool: The node pool configuration.
    ignore_failure: If True, command failures are ignored.
  """

  composer.log_metadata_for_xlml_dashboard({
      "cluster_project": node_pool.project_id,
      "region": node_pool.location,
      "zone": node_pool.node_locations,
      "cluster_name": node_pool.cluster_name,
      "node_pool_name": node_pool.node_pool_name,
      "accelerator_type": node_pool.machine_type,
  })

  if _node_pool_exists(node_pool):
    logging.info(
        f"Node pool {node_pool.node_pool_name} already exists. Skipping."
    )
    return

  command = (
      f"gcloud container node-pools create {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      f"--node-locations={node_pool.node_locations} "
      f"--num-nodes={node_pool.num_nodes} "
      f"--machine-type={node_pool.machine_type} "
      f"--tpu-topology={node_pool.tpu_topology} "
  )

  if node_pool.reservation:
    command += f" --reservation-affinity=specific --reservation={node_pool.reservation}"

  if node_pool.node_pool_selector:
    command += f" --node-labels={NODE_POOL_SELECTOR_KEY}={node_pool.node_pool_selector}"

  if ignore_failure:
    command += "2>&1 || true "

  try:
    subprocess.run_exec(command)
  except Exception as e:
    debug_cmd = (
        "gcloud container operations list "
        f"--project={node_pool.project_id} "
        f"--region={node_pool.location} "
        f"--format=json(name,status)"
    )
    debug_res = subprocess.run_exec(debug_cmd)

    raise AirflowFailException(
        f"Primary task failed. Current operations:\n{debug_res}"
    ) from e


@task
def delete(node_pool: Info) -> None:
  """Deletes the GKE node pool using gcloud command."""

  """Check if the node pool is valid."""
  if not _node_pool_exists(node_pool):
    logging.info(
        f"Node pool {node_pool.node_pool_name} already deleted or does not exist. Skipping."
    )
    return

  command = (
      f"gcloud container node-pools delete {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      "--quiet"
  )

  subprocess.run_exec(command)


def list_nodes(node_pool: Info) -> list[str]:
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

  command = (
      f"gcloud container node-pools describe {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      f"--format='json({instance_group_urls_key})'"
  )

  stdout = subprocess.run_exec(command)

  instance_group_urls_val = json.loads(stdout).get(instance_group_urls_key, [])
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

    command = (
        "gcloud compute instance-groups list-instances"
        f" {instance_group_name} "
        f"--project={node_pool.project_id} "
        f"--zone={node_pool.node_locations} "
        "--format='json(instance)'"
    )
    stdout = subprocess.run_exec(command)

    instances = json.loads(stdout)

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

  subprocess.run_exec(command)


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

  now = datetime.datetime.now()
  # Metrics are sampled every 60s and stored in the GCP backend,
  # but it may take up to 2 minutes for the data to become
  # available on the client side.
  # Therefore, a longer time interval is necessary.
  # A 5-minute window is an arbitrary but sufficient choice to
  # ensure we can retrieve the latest metric data.
  start_time_datetime = now - datetime.timedelta(minutes=5)
  start_time = TimeUtil.from_datetime(start_time_datetime)
  end_time = TimeUtil.from_datetime(now)

  filter_string = [
      'metric.type="kubernetes.io/node_pool/status"',
      f'resource.labels.project_id = "{node_pool.project_id}"',
      f'resource.labels.cluster_name = "{node_pool.cluster_name}"',
      f'resource.labels.node_pool_name = "{node_pool.node_pool_name}"',
  ]

  # A single query to the Monitoring API can return multiple TimeSeries objects,
  # especially if the 'status' label changed within the time window (e.g., from
  # 'PROVISIONING' to 'RUNNING').
  #
  # To robustly find the absolute latest status, this block first aggregates all
  # data points from all series into a single flat list ('records'). It then
  # finds the record with the maximum timestamp from this list to ensure the
  # true latest status is identified.
  time_series_data = list_time_series(
      project_id=node_pool.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
  )

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


@task.sensor(poke_interval=60, timeout=600, mode="poke")
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


@task
def rollback(node_pool: Info) -> None:
  """Performs a rollback on given GKE node pool using the gcloud command.

  Args:
      node_pool: An instance of the Info class that encapsulates the
        configuration and metadata of a GKE node pool.
  """
  command = (
      f"gcloud container node-pools rollback {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--region={node_pool.location} "
      f"--quiet"
  )

  subprocess.run_exec(command)


@task.sensor(poke_interval=30, timeout=1200, mode="poke")
def wait_for_availability(
    node_pool: Info,
    availability: bool,
    **context,
) -> bool:
  """Check current multi-host nodepool availability.

  This is a sensor task which retrieves the current list of the
  multi_host availability outputs for the last 600s, aggregated
  to 60s intervals. The results are then sorted, and the most recent
  result is checked to determine if it matches the desired result,
  either True or False.
  The default task runs every 30s for 1200s.

  Args:
      node_pool: An instance of the Info class that encapsulates
        the configuration and metadata of a GKE node pool.
      availability(bool): True if the function is checking for the
        nodepool to become available, False if the function is checking for
        it to become unavailble.
      context: The Airflow context dictionary, which includes task metadata.

  """
  now = datetime.datetime.now()
  # Metrics are sampled every 60s and stored in the GCP backend,
  # but it may take up to 2 minute for the metric data to become
  # available on the client side.
  # Therefore, a longer time interval is necessary.
  # A 10-minute window is an arbitrary but sufficient choice to
  # ensure we can retrieve the latest metric data.
  start_time_datetime = now - datetime.timedelta(minutes=10)
  start_time = TimeUtil.from_datetime(start_time_datetime)
  end_time = TimeUtil.from_datetime(now)

  filter_string = [
      'metric.type="kubernetes.io/node_pool/multi_host/available"',
      f'resource.labels.project_id = "{node_pool.project_id}"',
      f'resource.labels.cluster_name="{node_pool.cluster_name}"',
      f'resource.labels.node_pool_name="{node_pool.node_pool_name}"',
  ]

  page_result = list_time_series(
      project_id=node_pool.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
  )

  # We only want the most recent point, so we record all points in all
  # time series in a dictionary with their corresponding bool values to
  # ensure no overlapping time series can interfere.
  records = []
  for time_series in page_result:
    for point in time_series.points:
      end_ts_dt = point.interval.end_time
      pb = monitoring_v3.TypedValue.pb
      if pb(point.value).WhichOneof("value") == "bool_value":
        records.append((end_ts_dt, point.value.bool_value))

  if not records:
    logging.info("No records returned")
    return False

  _, state = max(records, key=lambda x: x[0])

  timeout = context["task"].timeout
  logging.info(
      "Waiting for node pool '%s' to become '%s' within %s seconds...",
      node_pool.node_pool_name,
      availability,
      timeout,
  )
  return availability == state


@task.sensor(poke_interval=30, timeout=3600, mode="poke")
def wait_for_ttr(
    node_pool: Info,
    operation_start_time: TimeUtil,
    **context,
) -> bool:
  """Waits for the node pool Times To Recover(TTR) records to occur.

  This task verifies if TTR records have been generated by querying
  the metric "kubernetes.io/node_pool/accelerator/times_to_recover"
  via the Google Cloud Monitoring API.
  Checks every 30s for up to 3600s.

  Args:
      node_pool: An instance of the Info class that encapsulates
        the configuration and metadata of a GKE node pool.
      operation_start_time: A TimeUtil object representing the start time of
        the operation. This serves as the anchor for the metric query window.
      context: The Airflow context dictionary, which includes task metadata.

  Returns:
      A boolean indicating whether TTR records have occurred for the node pool.
  """
  timeout = context["task"].timeout
  logging.info(
      "Waiting for TTR records for node pool '%s' (Timeout: %s seconds)...",
      node_pool.node_pool_name,
      timeout,
  )

  # Using a dynamic query window to handle Metric Ingestion Delay (~60 mins):
  # We anchor the start_time to the operation start time.
  # This serves two purposes:
  # 1. It ensures the window is wide enough to catch the delayed metric data
  #    (which retains its original Event Timestamp) as the sensor retries.
  # 2. It strictly filters out any stale records from previous DAG runs.
  start_time = operation_start_time
  now = datetime.datetime.now()
  end_time = TimeUtil.from_datetime(now)

  filter_string = [
      'metric.type = "kubernetes.io/node_pool/accelerator/times_to_recover"',
      f'resource.labels.project_id = "{node_pool.project_id}"',
      f'resource.labels.location = "{node_pool.location}"',
      f'resource.labels.cluster_name = "{node_pool.cluster_name}"',
      f'resource.labels.node_pool_name = "{node_pool.node_pool_name}"',
  ]

  page_result = list_time_series(
      project_id=node_pool.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
  )

  for time_series in page_result:
    if time_series.points:
      logging.info("TTR records found! Proceeding to validation.")
      return True

  logging.info("TTR records not found yet.")
  return False


def get_node_pool_disk_size(node_pool: Info) -> int:
  """Gets the disk size of a GKE node pool using gcloud command.

  Args:
    node_pool: An instance of the Info class that encapsulates the
      configuration and metadata of a GKE node pool.

  Returns:
    The disk size of the node pool in GB.
  """
  command = (
      f"gcloud container node-pools describe {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      f'--format="value(config.diskSizeGb)"'
  )

  result = subprocess.run_exec(command).strip()

  return int(result)


def get_node_pool_labels(node_pool: Info) -> dict[str, str]:
  """Gets the labels of a GKE node pool using gcloud command.

  Args:
    node_pool: An instance of the Info class that encapsulates the
      configuration and metadata of a GKE node pool.

  Returns:
    A dictionary contains the node pool labels.
  """
  command = (
      f"gcloud container node-pools describe {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      f"--format='json(config.resourceLabels)'"
  )

  result = (
      json.loads(subprocess.run_exec(command).strip())
      .get("config", {})
      .get("resourceLabels", {})
  )

  return result


class UpdateTarget(enum.Enum):
  """Defines what to update on the node pool."""

  DISK_SIZE = "disk-size"
  LABEL = "labels"


@dataclasses.dataclass
class NodePoolUpdateSpec:
  """Configuration parameters defining a mutation on a GKE node pool.

  Attributes:
    target: The specific node pool attribute to update.
    delta: The change to apply to the target's current state.
  """

  target: UpdateTarget
  delta: int | dict[str, str]

  @staticmethod
  def DiskSize(delta: int) -> "NodePoolUpdateSpec":
    if not isinstance(delta, int):
      raise TypeError(f"Disk size delta must be an integer. Got: {type(delta)}")

    if delta <= 0:
      raise ValueError(f"Disk size delta must be positive. Got: {delta}")

    return NodePoolUpdateSpec(
        target=UpdateTarget.DISK_SIZE,
        delta=delta,
    )

  @staticmethod
  def Label(delta: dict[str, str]) -> "NodePoolUpdateSpec":
    if not isinstance(delta, dict):
      raise TypeError(f"Label delta must be a dictionary. Got: {type(delta)}")

    key_pattern = re.compile(r"^[a-z][a-z0-9_-]*$")
    for k, v in delta.items():
      if not isinstance(k, str) or not isinstance(v, str):
        raise TypeError(
            f"All label keys and values must be strings. "
            f"Found incompatible item: key='{k}'({type(k)}), "
            f"value='{v}'({type(v)})"
        )

      if not key_pattern.match(k):
        raise ValueError(
            f"Invalid label key: '{k}'. "
            "Keys must start with a lowercase letter and contain only "
            "lowercase letters ([a-z]), numeric characters ([0-9]), "
            "underscores (_) and dashes (-)."
        )

    return NodePoolUpdateSpec(
        target=UpdateTarget.LABEL,
        delta=delta,
    )


@task
def update(node_pool: Info, spec: NodePoolUpdateSpec) -> TimeUtil:
  """Applies an update to a GKE node pool based on the provided specification.

  This task performs a state-aware update. It retrieves the current node pool
  state, resolves the final desired configuration based on the provided `spec`,
  and executes the update operation.

  Args:
    node_pool: An instance of the Info class that encapsulates the
      configuration and metadata of a GKE node pool.
    spec: An instance of the NodePoolUpdateSpec class defining the
      update target and parameters.

  Returns:
    A TimeUtil object representing the UTC timestamp when the operation started.

  Raises:
    ValueError: If the target is unsupported.
  """
  flags: list[str] = []

  match spec.target:
    case UpdateTarget.DISK_SIZE:
      current_disk_size = get_node_pool_disk_size(node_pool=node_pool)
      updated_disk_size = current_disk_size + spec.delta
      flags.append(f"--{spec.target.value}={updated_disk_size}")

    case UpdateTarget.LABEL:
      current_labels = get_node_pool_labels(node_pool=node_pool)
      updated_labels = []
      for key, val in spec.delta.items():
        if current_labels.get(key) == val:
          val += val
        updated_labels.append(f"{key}={val}")
      flags.append(f"--{spec.target.value}={','.join(updated_labels)}")

    case _:
      raise ValueError(f"Unsupported target: {spec.target}")

  flags_str = " ".join(flags)

  update_cmd = (
      f"gcloud container node-pools update {node_pool.node_pool_name} "
      f"--project={node_pool.project_id} "
      f"--cluster={node_pool.cluster_name} "
      f"--location={node_pool.location} "
      f"--quiet "
      f"{flags_str}"
  )

  operation_start_time = TimeUtil.from_datetime(
      datetime.datetime.now(datetime.timezone.utc)
  )

  subprocess.run_exec(update_cmd)
  return operation_start_time
