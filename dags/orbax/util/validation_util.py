"""Utilities to get workloads logs and some utils."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from absl import logging
import re

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api


@task
def generate_timestamp():
  return datetime.now(timezone.utc)


@task
def validate_checkpoint_at_steps_are_saved(
    project_id: str,
    location: str,
    cluster_name: str,
    steps_to_validate: list,
    ram_disk: str = "/local",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """
  Validates that a workload is training correctly by checking for specific log steps.

  This function queries logs from a specified GKE cluster and namespace.
  It searches for a log entry containing the string '(blocking + background)'
  and then compares the number of steps found against an expected list of steps.

  A mismatch in the number of steps will cause the validation to fail. This can
  happen if, for example, a restore operation causes the step count to restart
  from zero, leading to `len(vali_step_list) != len(found_steps)`.

  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    start_time: Optional start time for log retrieval
      (defaults to 12 hours ago)
    end_time: Optional end time for log retrieval (defaults to now)
    steps_to_validate: Optional to validate list of steps
  Returns:
    None: This function does not return a value.
  """

  log_pattern = (
      r"Finished async_save \(blocking \+ background\)\. "
      rf"Time taken: \d+\.\d+s\. directory={ram_disk}/(\d+)"
  )
  complied_pattern = re.compile(log_pattern)
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      text_filter=f'jsonPayload.message=~"{log_pattern}"',
      start_time=start_time,
      end_time=end_time,
  )

  steps_are_saved: set[int] = set()  # Use a set for faster lookup.
  for entry in entries:
    if not isinstance(entry, logging_api.StructEntry):
      raise AirflowFailException(
          "Log entry must be contain a jsonPayload attribute."
      )
    message = entry.payload.get("message")
    if not message:
      raise AirflowFailException(f"Failed to parse entry {entry}")

    m = complied_pattern.search(message)
    if m:
      steps_are_saved.add(int(m.group(1)))

  for step in steps_to_validate:
    if step not in steps_are_saved:
      logging.info(f"Found entries: {entries}")
      raise AirflowFailException(
          f"Failed to validate. Expect steps are saved: {steps_to_validate}; "
          f"got: {steps_are_saved}"
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
) -> list[logging_api.LogEntry]:
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

  logging_client = logging_api.Client(project=project_id)

  # Set the time window for log retrieval:
  # default to last 12 hours if not provided
  if end_time is None:
    end_time = datetime.now(timezone.utc)
  if start_time is None:
    start_time = end_time - timedelta(hours=12)

  # Format times as RFC3339 UTC "Zulu" format required by the Logging API
  start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
  end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

  conditions = [
      f'resource.labels.project_id="{project_id}"',
      f'resource.labels.location="{location}"',
      f'resource.labels.cluster_name="{cluster_name}"',
      f'resource.labels.namespace_name="{namespace}"',
      f'resource.labels.pod_name:"{pod_pattern}"',
      "severity>=DEFAULT",
      f'timestamp>="{start_time_str}"',
      f'timestamp<="{end_time_str}"',
  ]

  if container_name:
    conditions.append(f'resource.labels.container_name="{container_name}"')
  if text_filter:
    conditions.append(f"{text_filter}")

  log_filter = " AND ".join(conditions)

  logging.info(f"Log filter constructed: {log_filter}")
  return list(logging_client.list_entries(filter_=log_filter))
