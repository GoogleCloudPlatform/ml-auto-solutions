"""Utilities to get workloads logs and some utils."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from absl import logging

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as log_explorer


@task
def generate_timestamp():
  return datetime.now(timezone.utc)


@task
def validate_log_with_step(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    vali_step_list: Optional[list] = None,
) -> bool:
  """
  Validate the workload log is training correct
  Args:
      project_id: The Google Cloud project ID
      location: GKE cluster location
      cluster_name: GKE cluster name
      namespace: Kubernetes namespace (defaults to "default")
      pod_pattern: Pattern to match pod names (defaults to "*")
      container_name: Optional container name to filter logs
      text_filter: Optional comma-separated string to
      filter log entries by textPayload content
      start_time: Optional start time for log retrieval
      (defaults to 12 hours ago)
      end_time: Optional end time for log retrieval (defaults to now)
      vali_step_list: optional to validate list of steps
  Returns:
      bool: validate success or not
  """
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter=text_filter,
      start_time=start_time,
      end_time=end_time,
  )
  if vali_step_list is None:
    return False
  new_step_list = []
  for entry in entries:
    if not entry.payload:
      continue
    payload_str = str(entry.payload)
    for line in payload_str.split("\n"):
      if vali_step_list is not None:
        for step in vali_step_list:
          vali_str = "directory=/local/" + str(step)
          if vali_str in line and step not in new_step_list:
            logging.info(f"├─ Timestamp: {entry.timestamp}")
            logging.info("└─ Payload:")
            logging.info(f"   {line}")
            new_step_list.append(step)
  if len(vali_step_list) == len(new_step_list):
    logging.info("Validate success")
    return True
  else:
    raise AirflowFailException(
        f"{len(vali_step_list)} saves are expected,"
        f"but got {len(new_step_list)}"
    )


def list_log_entries(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> list:
  """
  List log entries for the specified Google Cloud project.
  This function connects to Google Cloud Logging,
  constructs a filter for Kubernetes container logs
  within a specific project, location, cluster, namespace,
  and pod name pattern, and retrieves log
  entries from the specified time range.
  It prints the timestamp, severity, resource information,
  and payload for each log entry found.
  Args:
      project_id: The Google Cloud project ID
      location: GKE cluster location
      cluster_name: GKE cluster name
      namespace: Kubernetes namespace (defaults to "default")
      pod_pattern: Pattern to match pod names (defaults to "*")
      container_name: Optional container name to filter logs
      text_filter: Optional comma-separated string to
      filter log entries by textPayload content
      start_time: Optional start time for log retrieval
      (defaults to 12 hours ago)
      end_time: Optional end time for log retrieval (defaults to now)
  Returns:
      bool: Number of log entries found
  """

  # Create a Logging Client for the specified project
  logging_client = log_explorer.Client(project=project_id)

  # Set the time window for log retrieval:
  # default to last 12 hours if not provided
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
      "severity>=DEFAULT "
      f'timestamp>="{start_time_str}" '
      f'timestamp<="{end_time_str}"'
  )

  # Add container name filter if provided
  if container_name:
    log_filter += f' resource.labels.container_name="{container_name}"'

  # Add text content filter if provided
  if text_filter:
    log_filter += f' SEARCH("{text_filter}")'

  # Retrieve log entries matching the filter
  logging.info(f"Log filter constructed: {log_filter}")
  entries = logging_client.list_entries(filter_=log_filter)

  return entries
