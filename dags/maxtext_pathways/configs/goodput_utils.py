# Copyright 2023 Google LLC
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

"""Common funcitons and tasks for Pathways Elastic Goodput measurement DAGs."""

# TODO(cienet): circle back for the whole file

# TODO(cienet): import grouping

import ast
from absl import logging
from ml_goodput_measurement import goodput_elastic

from airflow.decorators import task
from airflow.sensors.base import PokeReturnValue
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup
from airflow.models.baseoperator import chain
from google.cloud import logging as gcp_logging
from google.cloud.monitoring_v3 import types
# TODO(cienet): promote them as shared utility
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.gcp_util import list_time_series
from google.cloud import logging as gcp_logging

GOODPUT_LOG_LIST = [
    "Cumulative goodput monitoring process started for job: {workload_id}",
    "Started Goodput upload to Tensorboard & GCM in the background!",
    "Sent Goodput metrics to GCM Monitoring.",
    "Performing final goodput query and upload for job: {workload_id}",
    "Flushed final metrics and safe exited from Goodput monitoring.",
]


@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def check_goodput_logname(
    project_id: str,
    workload_id: str,
) -> bool:
  """
  Counts occurrences of a string pattern in GCP
  Cloud Logging for a specific workload.
  """
  # TODO(cienet): reuse utility in `from dags.tpu_observability.utils.gcp_util`
  # Initialize the GCP Logging Client
  client = gcp_logging.Client(project=project_id)

  log_filter = f'logName="projects/{project_id}/logs/goodput_{workload_id}" '

  logging.info(f"Querying GCP Logging with filter: {log_filter}")

  # Fetch the entries. (Adjust page_size based on log volume to optimize speed)
  entries = client.list_entries(filter_=log_filter, page_size=500)

  # Consolidate all log payloads into a single text body
  log_lines = []
  for entry in entries:
    payload = entry.payload

    if payload:
      if isinstance(payload, str):
        log_lines.append(payload)
      elif isinstance(payload, dict):
        message = payload.get("message") or payload.get("textPayload")
        if message:
          log_lines.append(str(message))
        else:
          log_lines.append(str(payload))

  full_logs_text = "\n".join(log_lines)

  if not full_logs_text:
    logging.info("No logs found yet in Cloud Logging for filter.")
    return False
  logging.info(f"Full Logs Text:\n{full_logs_text}")

  return True


@task.sensor(poke_interval=30, timeout=3600, mode="poke")
def elastic_goodput(workload_id: str, project_id: str):
  # Ensure Cloud Logging project and credentials are available
  logging.info(f"workload_id: {workload_id}, project: {project_id}")

  calc = goodput_elastic.ElasticGoodputCalculator(
      job_name=workload_id,
      logger_name=f"goodput_{workload_id}",
      using_pathways=True,
  )

  # Fetch Goodput and Badput Breakdown
  (
      _,
      badput_breakdown,
      _,
  ) = calc.get_job_goodput(include_badput_breakdown=True)

  # ELASTIC_SCALE_UP, ELASTIC_SCALE_DOWN, ELASTIC_REINITIALIZATION
  for badput_type, percentage in badput_breakdown.items():
    type_name = getattr(badput_type, "name", str(badput_type))
    if type_name.startswith("ELASTIC_"):
      logging.info(f"Badput due to {type_name}: {percentage:.2f}%")
      return True
  logging.info("No elastic badput found.")
  return False


@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def check_slice_counts(
    project_id: str,
    workload_id: str,
    start_time: str | None = None,
) -> PokeReturnValue:
  """
  Counts occurrences of a string pattern in GCP
  Cloud Logging for a specific workload.
  """
  # Initialize the GCP Logging Client
  client = gcp_logging.Client(project=project_id)

  log_filter = f'logName="projects/{project_id}/logs/goodput_{workload_id}" '
  if start_time:
    logging.info(f"Filtering logs starting from: {start_time}")
    log_filter += f' AND timestamp >= "{start_time}"'

  logging.info(f"Querying GCP Logging with filter: {log_filter}")

  # Fetch the entries. (Adjust page_size based on log volume to optimize speed)
  entries = client.list_entries(
      filter_=log_filter,
      page_size=500,
      order_by="timestamp desc",
  )

  latest_available_slices = None
  latest_active_slices = None
  latest_total_slices = None

  for entry in entries:
    payload = entry.payload
    parsed_payload = None

    if not payload:
      continue

    if isinstance(payload, dict):
      parsed_payload = payload
    # 2. Safely parse stringified dictionaries (textPayload)
    elif isinstance(payload, str):
      try:
        parsed_payload = ast.literal_eval(payload)
      except (ValueError, SyntaxError):
        continue

    if isinstance(parsed_payload, dict):
      if "available_slices" in parsed_payload:
        latest_available_slices = parsed_payload["available_slices"]
      if "active_slices" in parsed_payload:
        latest_active_slices = parsed_payload["active_slices"]
      if "total_slices" in parsed_payload:
        latest_total_slices = parsed_payload["total_slices"]
      if (
          latest_available_slices is not None
          and latest_active_slices is not None
          and latest_total_slices is not None
      ):
        logging.info(f"Latest metrics discovered: {parsed_payload}")
        break

  if (
      latest_available_slices is None
      and latest_total_slices is None
      and latest_active_slices is None
  ):
    logging.info("No slice metrics found yet in recent logs.")
    return PokeReturnValue(is_done=False, xcom_value={})

  return PokeReturnValue(
      is_done=True,
      xcom_value={
          "available_slices": latest_available_slices,
          "active_slices": latest_active_slices,
          "total_slices": latest_total_slices,
      },
  )


@task
def validate_slice_equality(slices_data: dict, label: str) -> dict:
  """
  Validates if available, active, and total slices are equal.
  Passes data through to avoid breaking pipeline stream.
  """
  avail = slices_data.get("available_slices")
  active = slices_data.get("active_slices")
  total = slices_data.get("total_slices")

  if None not in [avail, active, total]:
    if avail == active == total:
      logging.info(f"[{label}] Success: All slices are equal to {avail}.")
    else:
      logging.warning(
          f"[{label}] Mismatch: available={avail}, active={active}, total={total}"
      )
  return slices_data


@task
def compare_slice_trend(slices_before: dict, slices_after: dict) -> str:
  """
  Compares available slices before and after an operation.
  """
  avail_before = slices_before.get("available_slices")
  avail_after = slices_after.get("available_slices")

  if avail_before is None or avail_after is None:
    return "unknown"

  if avail_after > avail_before:
    trend = "increasing"
  elif avail_after < avail_before:
    trend = "decreasing"
  else:
    trend = "stable"

  logging.info(f"Available Slices Trend: {trend.upper()}")
  return trend


def query_workload_metrics(
    project_id: str,
    workload_id: str,
    query_metric: str,
    start_time: TimeUtil,
    end_time: TimeUtil,
) -> list[types.TimeSeries]:
  """Queries the Workload's metric from Cloud Monitoring."""
  filter_string = [
      f'metric.type = "{query_metric}"',
      'resource.type = "compute.googleapis.com/Workload"',
      f'resource.labels.workload_id = "{workload_id}"',
  ]

  return list_time_series(
      project_id=project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
      view=types.ListTimeSeriesRequest.TimeSeriesView.FULL,
      log_enable=True,
  )


@task.sensor(poke_interval=30, timeout=3600, mode="poke")
def slice_effiency_metrics(
    project_id: str,
    workload_id: str,
    start_time: str,
):
  """Verify uptime data exists after jobset application."""
  end_time = TimeUtil.now()
  start_time_obj = TimeUtil.from_iso_string(start_time)

  def fetch_metric(metric_type: str):
    data = query_workload_metrics(
        project_id,
        workload_id,
        metric_type,
        start_time_obj,
        end_time,
    )
    return data

  def log_metric(data: list[types.TimeSeries]):
    for series in data:
      for point in series.points:
        val = point.value.double_value or point.value.int64_value
        logging.info(f"  {point.interval.end_time}: {val}")

  stepping_se = "compute.googleapis.com/workload/stepping_slice_efficiency"
  stepping_data = fetch_metric(stepping_se)
  log_metric(stepping_data)

  available_se = "compute.googleapis.com/workload/available_slice_efficiency"
  available_data = fetch_metric(available_se)
  log_metric(available_data)

  if stepping_data or available_data:
    logging.info("Slice efficiency metrics found.")
    return True
  return False


def phase1_validate(
    project_id: str = "",
    workload_id: str = "",
    start_time: str = "",
    times: int = 0,
) -> DAGNode:
  """Run a test job with worker pod interruption."""
  with TaskGroup(group_id=f"phase1_validate_{times}") as group:
    slices_phase1 = check_slice_counts.override(
        task_id="check_slices_phase1",
        timeout=180,
    )(
        project_id=project_id,
        workload_id=workload_id,
    )
    validated_slices_phase1 = validate_slice_equality.override(
        task_id="validate_initial_equality"
    )(slices_data=slices_phase1, label="BEFORE")

    slice_efficiency_1 = slice_effiency_metrics.override(
        task_id="slice_effiency_metrics"
    )(
        project_id=project_id,
        workload_id=workload_id,
        start_time=start_time,
    )

    chain(
        slices_phase1,
        validated_slices_phase1,
        slice_efficiency_1,
    )

    return slices_phase1


def phase2_validate(
    project_id: str = "",
    workload_id: str = "",
    slices_phase1: dict = {},
    start_time: str = "",
    times: int = 0,
) -> DAGNode:
  """Run a test job with worker pod interruption."""
  with TaskGroup(group_id=f"phase2_validate_{times}") as group:
    slices_phase2 = check_slice_counts.override(
        task_id="check_slices_phase2",
        timeout=180,
    )(
        project_id=project_id,
        workload_id=workload_id,
        start_time=start_time,
    )
    validated_slices_phase2 = compare_slice_trend.override(
        task_id="compare_slice_trend"
    )(slices_before=slices_phase1, slices_after=slices_phase2)

    slice_efficiency_drop = slice_effiency_metrics.override(
        task_id="slice_effiency_metrics"
    )(
        project_id=project_id,
        workload_id=workload_id,
        start_time=start_time,
    )
    slice_down = elastic_goodput.override(task_id="elastic_goodput_check")(
        workload_id=workload_id,
        project_id=project_id,
    )

    chain(
        slices_phase2,
        validated_slices_phase2,
        slice_efficiency_drop,
        slice_down,
    )

    return slices_phase2


def phase3_validate(
    project_id: str = "",
    workload_id: str = "",
    start_time: str = "",
    times: int = 0,
) -> DAGNode:
  """Run a test job with worker pod interruption."""
  with TaskGroup(group_id=f"phase3_validate_{times}") as group:
    slice_efficiency_resume = slice_effiency_metrics.override(
        task_id="slice_effiency_metrics"
    )(
        project_id=project_id,
        workload_id=workload_id,
        start_time=start_time,
    )

    slice_up = elastic_goodput.override(task_id="elastic_goodput_check_resume")(
        workload_id=workload_id,
        project_id=project_id,
    )

    chain(
        slice_efficiency_resume,
        slice_up,
    )

    return group
