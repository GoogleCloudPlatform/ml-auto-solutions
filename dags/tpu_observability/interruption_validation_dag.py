"""A DAG to validate the accuracy of interruption counts from metrics."""

import dataclasses
import datetime
import enum
import re

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup

from dags.common import test_owner
from dags.common.vm_resource import Project
from dags.multipod.configs.common import Platform
from dags.tpu_observability.utils import gcp_util, time_util
from google.cloud import monitoring_v3
from dags import composer_env


_UNKNOWN_RESOURCE_NAME = 'Unknown'


@dataclasses.dataclass
class TimeRange:
  """Class containing proper time range for the validation."""

  start: int
  end: int


class InterruptionReason(str, enum.Enum):
  """Enum class for interruption reasons."""

  DEFRAGMENTATION = 'Defragmentation'
  EVICTION = 'Eviction'
  HOST_ERROR = 'HostError'
  MIGRATE_ON_HWSW_MAINTENANCE = 'Migrate on HW/SW Maintenance'
  HWSW_MAINTENANCE = 'HW/SW Maintenance'
  BARE_METAL_PREEMPTION = 'Bare Metal Preemption'
  OTHER = 'Other'

  def metric_label(self) -> str:
    """Returns the corresponding metric label for the interruption reason."""

    return self.value

  def log_filter(self) -> str:
    """Returns the corresponding filter for the interruption reason.

    These filters are in accordance with the definitions in this file from Google3:
    //depot/google3/java/com/google/cloud/cluster/manager/compute/services/instancemanagerevent/InstanceEventNotificationAction.java
    """

    filters = []
    match self:
      case InterruptionReason.DEFRAGMENTATION | InterruptionReason.EVICTION:
        filters = ['compute.instances.preempted']
      case InterruptionReason.HOST_ERROR:
        filters = ['compute.instances.hostError']
      case InterruptionReason.MIGRATE_ON_HWSW_MAINTENANCE:
        filters = ['compute.instances.migrateOnHostMaintenance']
      case InterruptionReason.HWSW_MAINTENANCE:
        filters = ['compute.instances.terminateOnHostMaintenance']
      case InterruptionReason.BARE_METAL_PREEMPTION:
        filters = ['compute.instances.baremetalCaretakerPreempted']
      case InterruptionReason.OTHER:
        filters = [
            'compute.instances.guestTerminate',
            'compute.instances.instanceManagerHaltForRestart',
            'compute.instances.stoppedDueToPdDoubleServe',
            'compute.instances.kmsKeyError',
            'compute.instances.shredmillKeyError',
            'compute.instances.invalidVmImage',
            'compute.instances.scratchDiskCreationFailed',
            'compute.instances.localSsdInitializationError',
            'compute.instances.localSsdInitializationKeyError',
            'compute.instances.localSsdVerifyTarError',
            'compute.instances.localSsdRecoveryAttempting',
            'compute.instances.localSsdRecoveryTimeoutError',
            'compute.instances.localSsdRecoveryFailedError',
        ]
      case _:
        raise ValueError(f'Unmapped interruption reason: {self}')
    return ' OR '.join(
        f'protoPayload.methodName="{filter}"' for filter in filters
    )


@dataclasses.dataclass
class Configs:
  """Validation configuration.

  Attributes:
    project_id: The ID of the GCP project.
    platform: The platform (GCE or GKE) where the validation is performed.
    interruption_reason: The specific interruption reason to validate.
  """

  project_id: str
  platform: Platform
  interruption_reason: InterruptionReason


@dataclasses.dataclass
class EventRecord:
  """Represents lists of metric points or log events for a single resource.

  Attributes:
    resource_name: The name of the resource (e.g., node or instance).
    record_timestamps: A list of timestamps for metric points or log events
     related to the resource.
  """

  resource_name: str
  record_timestamps: list[int] = dataclasses.field(default_factory=list)


def fetch_interruption_metric_records(
    configs: Configs,
    time_range: TimeRange,
) -> list[EventRecord]:
  """Retrieve the metrics from Cloud Monitoring API and group them.

  This function fetches time series data for interruption events based on the
  provided configuration and time range. It is used to identify when and on
  which resources interruptions have occurred.

  Args:
    configs: The configuration contains the parameters for validation.
    time_range: The time range to query for metrics.

  Returns:
    A list of EventRecord objects. Each EventRecord must contains the metric
    points timestamps for the resource name.

  Raises:
    RuntimeError: If the resource name cannot be determined from the time series
      data or encounter an unexpected TypedValue.
  """
  match configs.platform:
    case Platform.GCE:
      metric_type = 'tpu.googleapis.com/instance/interruption_count'
      resource_type = 'tpu.googleapis.com/GceTpuWorker'
      resource_label_key = 'instance_name'
      time_series_type = 'metric'
    case Platform.GKE:
      metric_type = 'kubernetes.io/node/interruption_count'
      resource_type = 'k8s_node'
      resource_label_key = 'node_name'
      time_series_type = 'resource'
    case _:
      raise ValueError(f'Unsupported platform: {configs.platform.value}')

  metric_filter = (
      f'resource.labels.project_id = "{configs.project_id}" '
      f'metric.type = "{metric_type}" '
      f'resource.type = "{resource_type}" '
      f'metric.labels.interruption_reason = "{configs.interruption_reason.metric_label()}" '
  )

  # key: resource_name, value: EventRecord
  event_records: dict[str, EventRecord] = {}
  response = gcp_util.list_time_series(
      project_id=configs.project_id,
      filter_str=metric_filter,
      start_time=time_util.TimeUtil.from_unix_seconds(time_range.start),
      end_time=time_util.TimeUtil.from_unix_seconds(time_range.end),
  )

  for time_series in response:
    resource = getattr(time_series, time_series_type)
    resource_name = resource.labels.get(
        resource_label_key, _UNKNOWN_RESOURCE_NAME
    )
    if resource_name == _UNKNOWN_RESOURCE_NAME:
      raise RuntimeError(
          f'Failed to extract resource name from "{time_series}"'
      )

    for point in time_series.points:
      end_time = time_util.TimeUtil.from_datetime(point.interval.end_time)
      match monitoring_v3.TypedValue.pb(point.value).WhichOneof('value'):
        case 'int64_value':
          event_count = point.value.int64_value
        case 'double_value':
          event_count = int(point.value.double_value)
        case _:
          raise RuntimeError(f'Unexpected TypedValue: {point}')

      # Value 0 indicates the interruption didn't occur at this timestamp.
      if event_count == 0:
        continue

      if resource_name not in event_records:
        event_records[resource_name] = EventRecord(
            resource_name=resource_name,
        )
      # The event_count represents a count of interruption events occurring
      # at the same time.
      # We need to add each event separately to the list of metric points.
      event_records[resource_name].record_timestamps.extend(
          [end_time.to_unix_seconds()] * event_count
      )

  return list(event_records.values())


@task
def fetch_interruption_log_records(
    configs: Configs,
    time_range: TimeRange,
) -> list[EventRecord]:
  """Retrieve log entries from Cloud Logging API and update the event record.

  This function fetches log entries related to interruption events that occurred
  within a specified time range for a given set of resources.

  Args:
    configs: The configuration contains the parameters for validation.
    time_range: The time range (start and end) to query for log entries.

  Returns:
    A list of EventRecord objects, updated with the timestamps of the log
    events for each resource.

  Raises:
    RuntimeError: Raised when log entries exceed a hardcoded limit, in
      which case a manual inspection may be more appropriate.
  """
  log_entries = gcp_util.query_log_entries(
      project_id=configs.project_id,
      filter_str=configs.interruption_reason.log_filter(),
      start_time=time_util.TimeUtil.from_unix_seconds(time_range.start),
      end_time=time_util.TimeUtil.from_unix_seconds(time_range.end),
  )

  # key: resource_name, value: EventRecord
  event_records: dict[str, EventRecord] = {}
  for entry in log_entries:
    # Obtain the text segment contains information of resourceName from payload.
    resource_name = entry.payload.get('resourceName', '')

    # Extract the resource name from a text like this:
    # "projects/{project_id}/zones/{zone}/instances/{resource_name}"
    regex_pattern = r'^projects/[\w-]+/zones/[\w-]+/instances/([\w-]+)$'
    match = re.match(regex_pattern, resource_name)
    if match:
      log_node_name = match.group(1)
      log_timestamp = time_util.TimeUtil.from_datetime(entry.timestamp)

      if log_node_name not in event_records:
        event_records[log_node_name] = EventRecord(
            resource_name=log_node_name,
        )
      event_records[log_node_name].record_timestamps.append(
          log_timestamp.to_unix_seconds()
      )

  return list(event_records.values())


@task
def determine_time_range(
    configs: Configs,
    **context,
) -> TimeRange:
  """Determines an optimal time range for interruption event validation.

  This function identifies a time window that is free of metric events near its
  boundaries. This "quiet" period, defined by `allowed_gap`, ensures that all
  metric events within the window can be reliably correlated with their
  corresponding log entries without ambiguity from events outside the window.

  The function starts with a recent time window and expands it backwards in
  time until a suitable window is found.

  Args:
    configs: The configuration object containing the necessary parameters for
      fetching metrics.
    context: The Airflow context dictionary, which includes task metadata.

  Returns:
    TimeRange object representing the start and end of the optimal validation
    window.

  Raises:
    AirflowSkipException: Raised to notify Airflow to skip the task and DAG when
      no suitable time window is found.
  """
  # We assume the max shift of the log is 30 minutes. (call it max_shift)
  # The allowed_gap should be 2 * 30 minutes. Here's why we need this buffer:
  #
  # A 1x max_shift is used to capture the last relevant metric event. This
  # ensures we can correlate it with its corresponding log event, even if the
  # log event occurs up to max_shift later, allowing us to find event pairs at
  # the very edge of the query window.
  #
  # The second 1x max_shift is crucial for preventing a different issue: it
  # explicitly excludes the next metric event. This is to avoid an incorrect
  # correlation, as the log for that next metric event might fall within our
  # query window, leading to misleading associations.
  allowed_gap = int(datetime.timedelta(minutes=30).total_seconds()) * 2

  # This test is scheduled to run every day,
  # so we validate the interruption within a day (at least)
  min_time_window = int(datetime.timedelta(days=1).total_seconds())
  time_window_step = min_time_window

  task_instance = context['ti']
  task_start_time = int(task_instance.start_date.timestamp())

  right_bound = task_start_time
  left_bound = task_start_time - 2 * time_window_step

  found_right = False
  while not found_right:
    # If the search for the right boundary has to go back more than three days
    # from the task start time, it indicates that the data in the past few days
    # is too dense. In such a case, validating the interruption count (three
    # days ago) does not make sense, and manual inspection is required.
    if abs(task_start_time - right_bound) > int(
        datetime.timedelta(days=3).total_seconds()
    ):
      raise AirflowSkipException('the data has been too dense in past few days')

    metric_records = fetch_interruption_metric_records(
        configs, TimeRange(start=left_bound, end=right_bound)
    )
    if not metric_records:
      raise AirflowSkipException(
          'No metric events found in the specified time range.'
      )

    total_metric_timestamps = []
    for record in metric_records:
      total_metric_timestamps.extend(record.record_timestamps)
    total_metric_timestamps.sort(reverse=True)  # Newest to oldest.

    # Find the right-most data that has a sufficient gap from the
    # right boundary.
    for r in total_metric_timestamps:
      if abs(r - right_bound) > allowed_gap:
        found_right = True
        break
      right_bound = r

    if not found_right:
      left_bound -= time_window_step
      continue
    else:
      # Right bound is determined.
      # We need to add an additional allowed_gap / 2 to the right bound,
      # to ensure that the log of the next metric event is not included in the
      # validation.
      right_bound = int(right_bound - allowed_gap / 2 - 1)

  found_left = False
  iteration_count = 0
  while not found_left:
    iteration_count += 1
    # At this point, the right bound has been determined.
    # However, since the left bound keeps shifting and the time window keeps
    # expanding, validating the interruption count over such a long duration
    # might not make sense.
    if right_bound - left_bound > int(
        datetime.timedelta(days=5).total_seconds()
    ):
      raise AirflowSkipException('the time window has been too long')

    metric_records = fetch_interruption_metric_records(
        configs, TimeRange(start=left_bound, end=right_bound)
    )

    total_metric_timestamps = []
    for record in metric_records:
      total_metric_timestamps.extend(record.record_timestamps)
    total_metric_timestamps.sort()  # Oldest to newest.

    # Find the left-most data that has a sufficient gap from the
    # left boundary.
    for r in total_metric_timestamps:
      if abs(r - left_bound) > allowed_gap:
        found_left = True
        break
      left_bound = r

    if not found_left or (right_bound - left_bound) < min_time_window:
      # The initial left bound is 2 * time_window_step before task_start_time.
      # Extend the time range by additional time_window_step.
      left_bound = task_start_time - (2 + iteration_count) * time_window_step
      continue
    else:
      # Left bound is determined.
      # We need to add an additional allowed_gap / 2 to the left bound,
      # to ensure that the log of the previous metric event is not included in
      # the validation.
      left_bound = int(left_bound + allowed_gap / 2 + 1)

  return TimeRange(start=left_bound, end=right_bound)


@task
def validate_interruption_count(
    metric_records: list[EventRecord],
    log_records: list[EventRecord],
):
  """Validates that the metric and log event counts match for each resource.

  This function compares the number of interruption events found in the metrics
  with the number of events found in the logs for each resource.

  Args:
    metric_records: A list of EventRecord objects containing metric timestamps
      for a specific resource.
    log_records: A list of EventRecord objects containing log timestamps for a
      specific resource.

  Raises:
    RuntimeError: If there is a mismatch between the metric and log event
      counts for any resource.
  """

  log_map = {log.resource_name: log.record_timestamps for log in log_records}

  mismatch_nodes = []

  for metric in metric_records:
    resource_name = metric.resource_name
    log_timestamps = log_map.get(resource_name, [])

    if len(log_timestamps) != len(metric.record_timestamps):
      mismatch_nodes.append(
          f'mismatch resource name: {resource_name}, metric_count: {len(metric.record_timestamps)}, log_count: {len(log_timestamps)}'
      )

  if mismatch_nodes:
    error_detail = '\n'.join(mismatch_nodes)
    raise RuntimeError(
        'Event count mismatch detected for the following nodes:\n'
        f'{error_detail}'
    )


def get_staggered_schedule(base_schedule: str, minutes_offset: int) -> str:
  """Generates a new schedule string by applying a minute offset."""
  # Split the base Cron string into its components.
  # "0 2 * * 2,3,4,6" -> ["0", "2", "*", "*", "2,3,4,6"]
  parts = base_schedule.split(' ')
  if len(parts) != 5:
    raise ValueError(
        f"Base cron '{base_schedule}' is not in the expected 5-field format."
    )

  # Extract components using descriptive variable names.
  current_minute = int(parts[0])
  current_hour = int(parts[1])

  # Calculate the new schedule time.
  total_minutes = (current_hour * 60) + current_minute + minutes_offset
  new_minute = total_minutes % 60
  new_hour = (total_minutes // 60) % 24

  # Reconstruct the new schedule string.
  # Format: [New Minute] [New Hour] [Fixed Date Parts]
  staggered_schedule = ' '.join([str(new_minute), str(new_hour)] + parts[2:])

  return staggered_schedule


def create_interruption_dag(
    dag_id: str,
    platform: Platform,
    interruption_reason: InterruptionReason,
    schedule_offset_minutes: int,
) -> models.DAG:
  """Creates an Airflow DAG for interruption event validation.

  This function generates a DAG that validates the accuracy of interruption
  events between metrics and logs for a specific platform and interruption
  reason.

  Args:
    dag_id: The unique identifier for the DAG.
    platform: The platform (GCE or GKE) to validate.
    interruption_reason: The specific interruption reason to validate.
    schedule_offset_minutes: The offset in minutes to apply to the schedule.

  Returns:
    An Airflow DAG object."""
  if composer_env.is_prod_env():
    dag_schedule = get_staggered_schedule('0 18 * * *', schedule_offset_minutes)
  else:
    dag_schedule = None
  with models.DAG(
      dag_id=dag_id,
      start_date=datetime.datetime(2025, 7, 20),
      schedule=dag_schedule,
      catchup=False,
      tags=[
          platform.value,
          'tpu-observability',
          'interruption-count',
          'TPU',
          'v6e-16',
      ],
      description=(
          'This DAG validates the accuracy of the interruption count metric by '
          'comparing it against logs.'
      ),
      doc_md="""
        # Interruption Event Validation DAG

        ### Description
        This DAG automates the validation of the interruption count metric.
        It compares the number of interruption events from the logs with the
        number of events from the metrics to ensure accuracy.

        ### Procedures
        This DAG first determines a time range for validation, then fetches the
        interruption events from both the Cloud Monitoring API (metrics) and the
        Cloud Logging API (logs). Finally, it compares the number of events from
        both sources to ensure they match.
      """,
  ) as dag:
    for project in Project:
      match project:
        case Project.TPU_PROD_ENV_AUTOMATED | Project.CLOUD_TPU_INFERENCE_TEST:
          # Production composer lacks permission for these projects; ignore them.
          continue
        case _:
          with TaskGroup(
              group_id=f'validation_for_{project.value}',
              tooltip=f'Validation pipeline for Project ID: {project.value}',
          ) as group:
            configs = Configs(
                project_id=project.value,
                platform=platform,
                interruption_reason=interruption_reason,
            )

            @task
            def fetch_interruption_metric_records_task(
                configs: Configs,
                proper_time_range: TimeRange,
            ) -> list[EventRecord]:
              return fetch_interruption_metric_records(
                  configs,
                  proper_time_range,
              )

            proper_time_range = determine_time_range(configs)
            metric_records = fetch_interruption_metric_records_task(
                configs,
                proper_time_range,
            )
            log_records = fetch_interruption_log_records.override(
                owner=test_owner.QUINN_M
            )(
                configs,
                proper_time_range,
            )
            check_event_count = validate_interruption_count.override(
                owner=test_owner.QUINN_M
            )(
                metric_records,
                log_records,
            )

            chain(
                proper_time_range,
                [metric_records, log_records],
                check_event_count,
            )

    return dag


dag_id_prefix = 'validate_interruption_count'
stagger_interval_minutes = 5
offset_minutes = 0  # Initial minute offset

for platform in [Platform.GCE, Platform.GKE]:
  for reason in InterruptionReason:
    reason_value = reason.name.lower()
    dag_id = f'{dag_id_prefix}_{platform.value}_{reason_value}'
    _ = create_interruption_dag(dag_id, platform, reason, offset_minutes)
    offset_minutes += stagger_interval_minutes
