"""Validation utilities for post training DAGs.

This module provides validation functions specific to post training
workflows, reusing generic utilities from the orbax module where applicable.
"""

from absl import logging
from datetime import datetime, timezone

from airflow import AirflowException
from airflow.decorators import task
from google.cloud import aiplatform
from google.cloud import storage

# Re-export commonly used validation functions from orbax
from dags.orbax.util.validation_util import (
    generate_timestamp,
    validate_log_exist,
)


def _ensure_tfevents_exist(logdir: str) -> None:
  """Ensures the provided GCS logdir contains TensorBoard event files."""
  if not logdir.startswith("gs://"):
    raise AirflowException("Vertex AI upload only supports GCS paths.")

  gcs_path = logdir[len("gs://") :]
  bucket_name, _, prefix = gcs_path.partition("/")
  prefix = prefix.lstrip("/")
  storage_client = storage.Client()
  blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
  has_event_files = any(".tfevents." in blob.name for blob in blobs)
  if not has_event_files:
    raise AirflowException(
        "No TensorBoard event files found at logdir: "
        f"{logdir}. Ensure the path contains .tfevents.* files."
    )


@task
def generate_run_name(
    prefix: str,
    mode: str,
    num_slices: int,
) -> str:
  """Generates a unique run name with a timestamp."""
  current_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
  return f"{prefix}-{mode}-{num_slices}x-{current_datetime}"


@task
def upload_to_vertex_ai(
    project_id: str,
    region: str,
    tensorboard_id: str,
    logdir: str,
    experiment_name: str,
    run_name_prefix: str,
) -> None:
  """Uploads TensorBoard logs to Vertex AI.

  This task uses the Vertex AI Python SDK to sync logs from a GCS directory
  to a specific TensorBoard experiment.
  """
  _ensure_tfevents_exist(logdir)

  aiplatform.init(project=project_id, location=region)
  aiplatform.upload_tb_log(
      tensorboard_id=tensorboard_id,
      tensorboard_experiment_name=experiment_name,
      logdir=logdir,
      run_name_prefix=run_name_prefix,
  )
  logging.info("Upload completed successfully.")


__all__ = [
    "generate_timestamp",
    "validate_log_exist",
    "upload_to_vertex_ai",
]
