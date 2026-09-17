import re
from datetime import datetime, timezone, timedelta
from typing import Optional
from absl import logging

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api


def list_log_entries(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> list[logging_api.LogEntry]:
  """
  List log entries for the specified Google Cloud project.
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
      f'resource.labels.pod_name=~"{pod_pattern}"',
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


@task
def validate_semantic_inference_output(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """Validates the log output for semantic inference quality."""
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

  if not entries:
    raise AirflowFailException(
        "The log history is empty! Cannot validate inference."
    )

  buffer = ""
  for entry in entries:
    message = None
    if isinstance(entry, logging_api.TextEntry):
      message = entry.payload
    elif isinstance(entry, logging_api.StructEntry):
      message = entry.payload.get("message")

    if message:
      buffer += message + "\\n"

  # Isolate the post-training inference which is the last output in the script
  # The script runs pre-train inference -> train -> post-train inference
  inference_blocks = buffer.split("Generated text:")
  if len(inference_blocks) > 1:
    eval_buffer = inference_blocks[-1]
    logging.info("Isolated the final 'Generated text:' block for validation.")
  else:
    eval_buffer = buffer
    logging.warning(
        "Could not find 'Generated text:' delimiter, validating entire log."
    )

  if "\ufffd" in eval_buffer:
    raise AirflowFailException(
        "Semantic validation failed: Non-UTF-8 characters detected (\\ufffd) in post-training inference."
    )

  if not re.search(
      r"(?i)(big ben|london eye|tower of london|buckingham palace)", eval_buffer
  ):
    raise AirflowFailException(
        "Semantic validation failed: Expected London landmark keywords not found in post-training inference output."
    )

  logging.info("Semantic inference verification passed successfully.")


@task
def generate_timestamp():
  return datetime.now(timezone.utc)
