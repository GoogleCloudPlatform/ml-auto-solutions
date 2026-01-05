"""Utilities to get workloads logs and some utils."""

from datetime import datetime, timezone, timedelta
from typing import Optional
from absl import logging
import re

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud import logging as logging_api
from xlml.apis import gcs


@task
def generate_timestamp():
  return datetime.now(timezone.utc)


@task
def validate_checkpoint_at_steps_are_saved(
    project_id: str,
    location: str,
    cluster_name: str,
    steps_to_validate: list[int],
    ram_disk: str = "/local",
    pod_pattern: Optional[str] = ".*",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """
  Validates that a workload is training correctly by checking for specific log
  steps.

  This function queries logs from a specified GKE cluster and namespace.
  It searches for a log entry containing the string '(blocking + background)'
  and then compares the number of steps found against an expected list of
  steps.

  A mismatch in the number of steps will cause the validation to fail. This can
  happen if, for example, a restore operation causes the step count to restart
  from zero, leading to `len(steps_to_validate) != len(found_steps)`.

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

  directory_pattern = (
      rf"{re.escape(ram_disk)}/(\d+)"
      if ram_disk != "gcs"
      else r"gs://[^/]+/[^/]+/[^/]+/checkpoints/(\d+)"
  )
  log_pattern = (
      rf"Finished async_save \(blocking \+ background\)\. "
      rf"Time taken: \d+\.\d+s\. directory={directory_pattern}"
  )

  complied_pattern = re.compile(log_pattern)
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      pod_pattern=pod_pattern,
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


@task
def generate_run_name(
    short_id: str,
    checkpointing_type: str,
    slice_number: int,
    accelerator: str,
) -> str:
  """
  Generates a unique run name for a MaxText run based on given parameters.

  The function creates a formatted string that includes a short identifier,
  the number of slices, the accelerator type, and the current timestamp. This
  run name is useful for uniquely identifying a specific training run,
  especially for checkpointing and logging purposes.

  Args:
    short_id: A short identifier for the specific model or experiment.
    checkpointing_type: The name of the checkpointing strategy (e.g., 'emc').
    slice_number: The number of TPU slices used for the training run.
    accelerator: The type of accelerator used (e.g., 'tpu-v4').

  Returns:
    A string formatted as
      '{short_id}-mtc-{slice_number}x-{accelerator}-{timestamp}'.
  """

  run_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
  run_name = (
      f"{short_id}-{checkpointing_type}-"
      f"{slice_number}x-{accelerator}-{run_time}"
  )
  return run_name


@task
def validate_log_with_gcs(
    project_id: str,
    location: str,
    cluster_name: str,
    checkpoint_dir: str,
    namespace: str = "default",
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """
  Validates workload logs against GCS bucket checkpoints.

  This function first queries logs from a specified GKE cluster to determine
  the GCS bucket path used for checkpointing. It then retrieves logs related to
  checkpoint save operations, extracts the step numbers, and verifies that the
  corresponding checkpoint files exist in the GCS bucket. The function passes
  if the latest step found in the logs matches the latest step found in the GCS
  bucket's checkpoint filenames. It raises an exception on failure.

  Args:
    project_id: The Google Cloud project ID.
    location: The GKE cluster location.
    cluster_name: The GKE cluster name.
    namespace: The Kubernetes namespace. Defaults to "default".
    pod_pattern: A glob pattern to match pod names. Defaults to "*".
    container_name: An optional container name to filter logs by.
    text_filter: An optional string to filter log entries by their
      `textPayload`.
    start_time: The start time for log retrieval.
    end_time: The end time for log retrieval.

  Returns:
    None. The function completes successfully if all validation steps are found.

  Raises:
    AirflowFailException: If the bucket path format is invalid, if checkpoint
      files are missing, if steps cannot be extracted from log lines,if step
      lists are empty, or if the latest steps do not match.
  """

  # Get the entries for the backup steps in the bucket. To later compare the
  # latest stored step in bucket with the latest recorded step in training pod.
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter=f'textPayload=~"{text_filter}"',
      start_time=start_time,
      end_time=end_time,
  )
  gcs_save_step_list = []
  gcs_save_step_list_bucket = []
  for entry in entries:
    if entry.payload is not None:
      payload_str = str(entry.payload)
      for line in payload_str.split("\n"):
        # Extract the gcs bucket path from replicator logs
        gcs_pattern = r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2,}"
        step_pattern = r"step (\d+)"
        match_gcs = re.search(gcs_pattern, line)
        match_step = re.search(step_pattern, line)
        validate_check_gcs = False

        # If could not found those values eg. gcs=2025-08-10_12-09 and step=60.
        if match_gcs and match_step and checkpoint_dir:
          gcs_checkpoint_path = match_gcs.group(0)
          step = match_step.group(1)
          logging.info(f"get gcs path from: {gcs_checkpoint_path}")
          bucket_files = gcs.obtain_file_list(
              f"{checkpoint_dir}/{gcs_checkpoint_path}/"
          )
          logging.info(f"gcs bucket files lenght: {len(bucket_files)}")
          if len(bucket_files) > 0:
            # Extract .meta file to future comparison
            for file in bucket_files:
              if ".meta" in file:
                gcs_save_step_list_bucket.append(file)
                break

            # Check for correct format .data
            for file in bucket_files:
              if ".data" in file:
                validate_check_gcs = True
                break

          if not validate_check_gcs:
            raise AirflowFailException(
                f"Checkpoint files can not found in {gcs_checkpoint_path}"
            )

          # Add it to a global list that we will use later to compare with
          # bucket
          gcs_save_step_list.append(int(step))
        else:
          raise AirflowFailException(
              f"Could not find gcs_checkpoint_path or step in line: {line}"
          )

  # Extract the step number from the LAST RECORDED file in GCS bucket ended
  # with .meta, which is prefixed with 's' and followed by digits
  # eg. <name_job>-<cluster_config>-<date>-36-s60-n26-w0.meta
  # The extracted step should be step=60 and is the last saved step.
  if len(gcs_save_step_list_bucket) > 0 and len(gcs_save_step_list) > 0:
    pattern_bucket_step = r"s(\d+)"
    raw_str_filename = gcs_save_step_list_bucket[-1]
    match = re.search(pattern_bucket_step, raw_str_filename)
    if match is None:
      raise AirflowFailException(
          f"Could not extract step from filename: {raw_str_filename}"
      )
    last_step_bucket = match.group(0)[1:]
    if int(last_step_bucket) == max(gcs_save_step_list):
      logging.info("Validate success")
  else:
    raise AirflowFailException(
        f"Steps in bucket or replicator logs are empty. "
        f"GCS bucket steps found: {len(gcs_save_step_list_bucket)}. "
        f"Replicator log steps found: {len(gcs_save_step_list)}."
    )
  return max(gcs_save_step_list), max(gcs_save_step_list_bucket)


@task
def validate_gcs_checkpoint_files(
    bucket_path: str,
    enable_multi_tier_checkpointing: bool = False,
    steps_to_validate: Optional[list] = None,
) -> None:
  """
  Validates that checkpoint files exist in GCS bucket for expected steps.
  This function uses the GCS utility to check that checkpoint files
  are properly saved in the bucket for each expected step.
  Args:
    bucket_path: The full gs:// path to the GCS bucket
    steps_to_validate: Optional list of steps to validate
  Returns:
    None: Raises AirflowFailException if checkpoint validation fails
  """
  if steps_to_validate is None:
    logging.info(
        "No validation steps provided, skipping GCS checkpoint validation"
    )
    return

  logging.info("Validate GCS checkpoint files on path: %s", bucket_path)
  try:
    checkpoint_files = gcs.obtain_file_list(bucket_path)
    logging.info("Found checkpoint files in GCS: %s", checkpoint_files)
    # Extract step from checkpoint files or directories
    found_steps = set()
    for file_path in checkpoint_files:
      step_num = extract_step_number_from_file_path(
          file_path, enable_multi_tier_checkpointing
      )
      if step_num is not None:
        found_steps.add(step_num)

    expected_steps = set(steps_to_validate)
    missing_steps = expected_steps - found_steps

    logging.info("Expected steps: %s", sorted(expected_steps))
    logging.info("Found steps: %s", sorted(found_steps))

    if missing_steps:
      raise AirflowFailException(
          "GCS checkpoint validation failed: Missing checkpoint files for "
          f"steps {sorted(missing_steps)}. "
          f"Expected steps: {sorted(expected_steps)}, "
          f"Found steps: {sorted(found_steps)}"
      )

    logging.info("GCS checkpoint validation successful!")

  except Exception as e:
    raise AirflowFailException(f"Error validating GCS checkpoints: {e}") from e


def extract_step_number_from_file_path(
    path: str,
    enable_multi_tier_checkpointing: bool,
) -> Optional[int]:
  """
  Extracts the step number from a file path.
  Args:
    path: The full path to the file or the folder.
    enable_multi_tier_checkpointing: Whether to enable multi-tier
      checkpointing.
  Returns:
    Optional[int]: The step number extracted from the file path, or None if
    not found.
  """
  logging.info("checking for mtc checkpoint file path: %s", path)
  if enable_multi_tier_checkpointing:
    # max-mtc-resume-gcs-mtc-2x-tpu-v5p-16-2025-10-22-08-36/2025-10-22_08-42/max-mtc-resume-gcs-mtc-2x-tpu-v5p-16-2025-10-22-08-36-s199-n2-w0.meta
    step_pattern = r"-s(\d+)-n\d+-w\d+\.meta$"
    complied_step_pattern = re.compile(step_pattern)
    matched = complied_step_pattern.search(path)
    logging.info("matched: %s", matched)
    if matched:
      logging.info("found step: %s", matched.group(1))
      return int(matched.group(1))
  else:
    path_parts = path.split("/")
    for part in path_parts:
      if part.isdigit():
        logging.info("found step: %s", part)
        return int(part)
  return None


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
def validate_log_exist(
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
  """Validate the workload log text filter it is found during training."""

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
    raise AirflowFailException("The log history is empty!")


@task
def validate_restored_correct_checkpoint(
    project_id: str,
    location: str,
    cluster_name: str,
    interrupt_at_step: int,
    pod_pattern: str = ".*",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    check_last_two_local_saves=True,
) -> None:
  """Validate the restored step is in the expected range."""

  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace="default",
      pod_pattern=pod_pattern,
      text_filter="jsonPayload.message:\"'event_type'\"",
      start_time=start_time,
      end_time=end_time,
  )

  if not entries:
    raise AirflowFailException("No event_type found in the log.")

  local_saved_steps_before_restore = []
  for entry in entries:
    if not isinstance(entry, logging_api.StructEntry):
      raise AirflowFailException(
          "Log entry must be contain a jsonPayload attribute."
      )

    message = entry.payload.get("message")

    if re.search(r"'event_type': 'save'", message):
      saved_step_match = re.search(r"'step': (\d+)", message)
      if not saved_step_match:
        raise AirflowFailException(
            f"Found save event with no step number, message: {message}"
        )

      local_saved_steps_before_restore.append(int(saved_step_match.group(1)))

    elif re.search(r"'event_type': '(emergency_)?restore'", message):
      logging.info("Found restore event: %s", message)
      logging.info(
          "Saved steps before restore: %s", local_saved_steps_before_restore
      )

      restored_step_match = re.search(
          r"'step':\s*(?:np\.int32\()?(\d+)", message
      )
      restored_step = (
          int(restored_step_match.group(1)) if restored_step_match else None
      )

      if not restored_step:
        raise AirflowFailException(
            f"Found restore event with no step number, message: {message}"
        )

      if restored_step < interrupt_at_step:
        raise AirflowFailException(
            f"Restored step {restored_step} should be "
            f"greater than or equal to step {interrupt_at_step}."
        )

      if (
          check_last_two_local_saves
          and restored_step not in local_saved_steps_before_restore[-2:]
      ):
        raise AirflowFailException(
            f"Restored step {restored_step} should be "
            "in the last two saved steps."
        )

      if (
          not check_last_two_local_saves
          and restored_step not in local_saved_steps_before_restore
      ):
        raise AirflowFailException(
            f"Restored step {restored_step} should be in the saved steps."
        )

      logging.info("Restoration happened at the expected step.")
      return

  raise AirflowFailException(
      "Failed to validate that restoration happened at the expected step."
  )


@task
def validate_replicator_gcs_restore_log(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    container_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    backed_up_steps: Optional[set[int]] = None,
) -> None:
  """
  Validates that the replicator successfully restored checkpoints from GCS
  backup. This function queries logs from a specified GKE cluster and namespace
  to look for log entries showing replicator restoring from GCS backup.
  It validates that restored steps match previously backed up steps.
  Expected log format:
  "Restoring from backup '2025-08-22_03-01', checkpoint 120"
  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    namespace: Kubernetes namespace (defaults to "default")
    pod_pattern: Pattern to match pod names (defaults to "*")
    start_time: Optional start time for log retrieval (defaults to 12 hours)
    end_time: Optional end time for log retrieval (defaults to now)
    backed_up_steps: Optional list of backed up steps
  Returns:
    None: Raises AirflowFailException if replicator GCS restore validation
      fails
  """
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter="Restoring from backup",
      start_time=start_time,
      end_time=end_time,
  )

  restored_steps = set()

  for entry in entries:
    if not entry.payload:
      continue
    payload_str = str(entry.payload)

    for line in payload_str.split("\n"):
      # Look for restore initiation logs
      if "Restoring from backup" in line and "checkpoint" in line:
        # Extract step from restore log
        # Example: "Restoring from backup '2025-08-22_03-01', checkpoint 120"
        checkpoint_match = re.search(
            r"Restoring from backup '[\w-]+', checkpoint (\d+)", line
        )

        if checkpoint_match:
          step = checkpoint_match.group(1)
          restored_steps.add(int(step))

          logging.info("├─ Found restore initiation at step %s", step)
          logging.info("├─ Timestamp: %s", entry.timestamp)
          logging.info("└─ Full log:")
          logging.info("   %s", line)

  if not restored_steps:
    raise AirflowFailException(
        "No replicator restore operations found. "
        "Replicator may not be working correctly."
    )

  logging.info("Backed up steps: %s", sorted(backed_up_steps))
  logging.info("Restored steps: %s", sorted(restored_steps))

  # If backup_info is provided,
  # validate that restored steps match backed up steps
  if backed_up_steps:
    # Validate that restored steps were previously backed up
    # restored should be subset of backed up
    invalid_restores = restored_steps - backed_up_steps
    if invalid_restores:
      raise AirflowFailException(
          "Restore validation failed: "
          f"Steps {sorted(invalid_restores)} were restored but were not found "
          f"in backed up steps {sorted(backed_up_steps)}."
      )

  logging.info("Replicator restore validation successful!")


@task
def validate_replicator_gcs_backup_log(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = "*",
    container_name: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> set[int]:
  """
  Validates that the replicator successfully backed up checkpoints to GCS and
  returns backup info.
  This function queries logs from a specified GKE cluster and namespace
  to look for log entries showing replicator backing up checkpoints to GCS.
  Replicator backups happen at time intervals, not specific steps.
  Expected log format:
  "Created backup '2025-08-22_03-01' for checkpoint 100"
  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    namespace: Kubernetes namespace (defaults to "default")
    pod_pattern: Pattern to match pod names (defaults to "*")
    start_time: Optional start time for log retrieval (defaults to 12 hours)
    end_time: Optional end time for log retrieval (defaults to now)
  Returns:
    dict: Dictionary mapping step numbers to backup folder names {step: folder}
  Raises:
    AirflowFailException: If replicator backup validation fails
  """
  step_regex = "backup for step (\d+) to [^\s]+"
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      namespace=namespace,
      pod_pattern=pod_pattern,
      container_name=container_name,
      text_filter=f'textPayload=~"{step_regex}"',
      start_time=start_time,
      end_time=end_time,
  )

  backed_up_steps = set()

  for entry in entries:
    if not entry.payload:
      continue
    payload_str = str(entry.payload)

    for line in payload_str.split("\n"):
      # Extract step from backup log
      # Example: "backup for step 399 to backup/gcs/2025-10-30_07-15"
      backup_match = re.search(rf"{step_regex}", line)

      if backup_match:
        step = int(backup_match.group(1))

        backed_up_steps.add(step)

        logging.info("├─ Found backup step %s", step)
        logging.info("└─ Full log: %s", line)

  if not backed_up_steps:
    raise AirflowFailException(
        "No replicator backup operations found. "
        "Replicator backup may not be working correctly."
    )

  logging.info("Backed up checkpoint steps: %s", sorted(backed_up_steps))
  logging.info("Replicator backup validation successful!")

  return backed_up_steps


@task
def validate_checkpoints_save_regular_axlearn(
    project_id: str,
    run_name: str,
    location: str,
    cluster_name: str,
    steps_to_validate: list,
    pod_pattern: Optional[str] = ".*",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  # This log pattern will be looged in a random pod of the first slice.
  log_pattern = r"^Serialization.*?step_(?P<step>\d+).*"
  complied_pattern = re.compile(log_pattern)
  logging.info(f"Run_name: {run_name.split('-')[0]}\n")
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      pod_pattern=pod_pattern,
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
  logging.info(
      f"Successful Validation.\nExpected  Steps:{steps_to_validate}"
      f"\tFound Steps:{steps_are_saved}"
  )
