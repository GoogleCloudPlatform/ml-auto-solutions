"""Validation utilities for post training DAGs.

This module provides validation functions specific to post training
workflows, reusing generic utilities from the orbax module where applicable.
"""

import datetime

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging_v2

# Re-export commonly used validation functions from orbax
from dags.orbax.util.validation_util import (
    generate_timestamp,
    validate_log_exist,
)

__all__ = [
    "generate_posttraining_run_name",
    "generate_timestamp",
    "validate_log_exist",
    "validate_tpu_vm_log_exist",
]


@task
def generate_posttraining_run_name(
    short_id: str,
    checkpointing_type: str,
    slice_number: int,
    mode: str,
) -> str:
  """
  Generates a short run name for a post-training run.

  Args:
      short_id: A short identifier for the specific model or experiment.
      checkpointing_type: The name of the checkpointing strategy (e.g., 'grpo').
      slice_number: The number of TPU slices used.
      mode: The setup mode (e.g., 'nightly').

  Returns:
      A short string formatted as
      '{short_id}-{checkpointing_type}-{mode}-{slice_number}'.
  """
  run_name = f"{short_id}-{checkpointing_type}-{mode}-{slice_number}"
  return run_name


@task
def validate_tpu_vm_log_exist(
    project_id: str,
    zone: str,
    node_id_pattern: str,
    text_filter: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
  """Validate that a log entry exists in TPU VM logs.

  Args:
      project_id: GCP project ID.
      zone: GCP zone where the TPU VM is located.
      node_id_pattern: Pattern to match TPU VM instance names (regex).
      text_filter: Text to search for in log entries.
      start_time: Start time for log search.
      end_time: End time for log search.

  Raises:
      AirflowFailException: If no matching log entries are found.
  """
  client = logging_v2.Client(project=project_id)

  # Build filter for TPU VM logs (Compute Engine instances)
  filter_parts = [
      'resource.type="gce_instance"',
      f'resource.labels.zone="{zone}"',
      f'timestamp>="{start_time.isoformat()}Z"',
      f'timestamp<="{end_time.isoformat()}Z"',
  ]

  # Add instance name pattern if provided
  if node_id_pattern and node_id_pattern != ".*":
    # Convert simple wildcard pattern to regex if needed
    instance_filter = node_id_pattern.replace("*", ".*")
    filter_parts.append(f'resource.labels.instance_id=~"{instance_filter}"')

  # Add text filter
  if text_filter:
    filter_parts.append(f"textPayload=~{text_filter}")

  filter_str = " AND ".join(filter_parts)

  # Query logs
  entries = list(client.list_entries(filter_=filter_str, page_size=10))

  if not entries:
    raise AirflowFailException(
        f"No log entries found matching filter: {filter_str}"
    )
