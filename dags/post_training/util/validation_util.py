"""Validation utilities for post training DAGs.

This module provides validation functions specific to post training
workflows, reusing generic utilities from the orbax module where applicable.
"""

from airflow.decorators import task
from airflow.exceptions import AirflowException

# Re-export commonly used validation functions from orbax
from dags.orbax.util.validation_util import (
    generate_timestamp,
    validate_log_exist,
)

__all__ = [
    "generate_posttraining_run_name",
    "generate_timestamp",
    "validate_log_exist",
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
