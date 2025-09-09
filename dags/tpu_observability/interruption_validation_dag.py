"""A DAG to validate the accuracy of interruption counts from metrics."""

import dataclasses
import datetime
import enum
import re
from typing import List

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from dags.common.vm_resource import Project
from dags.map_reproducibility.utils.constants import Schedule
from dags.multipod.configs.common import Platform
from google.cloud import logging
from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2
from proto import datetime_helpers


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
  record_timestamps: List[int] = dataclasses.field(default_factory=list)


def fetch_interruption_metric_records(
    configs: Configs,
    time_range: TimeRange,
) -> List[EventRecord]:
  """Retrieve the metrics from Cloud Monitoring API and group them.

  This function fetches time series data for interruption events based on the
  provided configuration and time range. It is used to identify when and on
  which resources interruptions have occurred.

  Args:
    configs: The configuration contains the parameters for validation.
    time_range: The time range to query for metrics.

  Returns:
    A List of EventRecord objects. Each EventRecord must contains the metric
    points timestamps for the resource name.

  Raises:
    RuntimeError: If the resource name cannot be determined from the time series
      data or encounter an unexpected TypedValue.
  """
  project_id = configs.project_id
  interruption_reason = configs.interruption_reason.metric_label()

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
      f'resource.labels.project_id = "{project_id}" '
      f'metric.type = "{metric_type}" '
      f'resource.type = "{resource_type}" '
      f'metric.labels.interruption_reason = "{interruption_reason}" '
  )

  project_name = f'projects/{project_id}'
  # key: resource_name, value: EventRecord
  event_records: dict[str, EventRecord] = {}

  start_timestamp = timestamp_pb2.Timestamp()
  start_timestamp.FromSeconds(time_range.start)
  end_timestamp = timestamp_pb2.Timestamp()
  end_timestamp.FromSeconds(time_range.end)

  interval = monitoring_v3.TimeInterval(
      start_time=start_timestamp, end_time=end_timestamp
  )

  request = monitoring_v3.ListTimeSeriesRequest(
      name=project_name,
      filter=metric_filter,
      interval=interval,
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  monitoring_api_client = monitoring_v3.MetricServiceClient()
  response = monitoring_api_client.list_time_series(request=request)

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
      end_time_obj: datetime_helpers.DatetimeWithNanoseconds = (
          point.interval.end_time
      )
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
          [int(end_time_obj.timestamp())] * event_count
      )

  return list(event_records.values())


@task
def fetch_interruption_log_records(
    configs: Configs,
    time_range: TimeRange,
) -> List[EventRecord]:
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
  start_time = datetime.datetime.fromtimestamp(
      time_range.start, tz=datetime.timezone.utc
  )
  end_time = datetime.datetime.fromtimestamp(
      time_range.end, tz=datetime.timezone.utc
  )

  project_id = configs.project_id
  log_filter_query = configs.interruption_reason.log_filter()

  logging_api_client = logging.Client(project=project_id)

  start_time_str = start_time.isoformat().replace('+00:00', 'Z')
  end_time_str = end_time.isoformat().replace('+00:00', 'Z')
  time_range_str = (
      f'timestamp>="{start_time_str}" AND timestamp<="{end_time_str}"'
  )

  max_results = 1000  # Avoid fetching too many logs.
  log_entries = logging_api_client.list_entries(
      filter_=f'({time_range_str}) AND ({log_filter_query})',
      order_by=logging.DESCENDING,
      max_results=max_results,
  )

  # key: resource_name, value: EventRecord
  event_records: dict[str, EventRecord] = {}
  entry_count = 0
  for entry in log_entries:
    entry_count += 1
    # Obtain the text segment contains information of resourceName from payload.
    resource_name = entry.payload.get('resourceName', '')

    # Extract the resource name from a text like this:
    # "projects/{project_id}/zones/{zone}/instances/{resource_name}"
    regex_pattern = r'^projects/[\w-]+/zones/[\w-]+/instances/([\w-]+)$'
    match = re.match(regex_pattern, resource_name)
    if match:
      log_node_name = match.group(1)
      aware_timestamp = entry.timestamp.replace(tzinfo=datetime.timezone.utc)

      if log_node_name not in event_records:
        event_records[log_node_name] = EventRecord(
            resource_name=log_node_name,
        )
      event_records[log_node_name].record_timestamps.append(
          int(aware_timestamp.timestamp())
      )

  if entry_count >= max_results:
    raise RuntimeError(f'Log entries limit reached ({max_results} entries).')

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
    metric_records: List[EventRecord],
    log_records: List[EventRecord],
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

  for metric in metric_records:
    resource_name = metric.resource_name
    log_timestamps = log_map.get(resource_name, [])

    if len(log_timestamps) != len(metric.record_timestamps):
      raise RuntimeError(
          'Event count mismatch. \n'
          f'Metric records count for {resource_name}: {len(metric.record_timestamps)} \n'
          f'Log records count for {resource_name}: {len(log_timestamps)}'
      )


def create_interruption_dag(
    dag_id: str,
    platform: Platform,
    interruption_reason: InterruptionReason,
) -> models.DAG:
  """Creates an Airflow DAG for interruption event validation.

  This function generates a DAG that validates the accuracy of interruption
  events between metrics and logs for a specific platform and interruption
  reason.

  Args:
    dag_id: The unique identifier for the DAG.
    platform: The platform (GCE or GKE) to validate.
    interruption_reason: The specific interruption reason to validate.

  Returns:
    An Airflow DAG object."""
  with models.DAG(
      dag_id=dag_id,
      start_date=datetime.datetime(2025, 7, 20),
      schedule=Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
      catchup=False,
      tags=[platform.value, 'tpu-observability', 'interruption-count'],
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
    configs = Configs(
        project_id=models.Variable.get(
            'INTERRUPTION_PROJECT_ID',
            default_var=Project.TPU_PROD_ENV_ONE_VM.value,
        ),
        platform=platform,
        interruption_reason=interruption_reason,
    )

    @task
    def fetch_interruption_metric_records_task(
        configs: Configs,
        proper_time_range: TimeRange,
    ) -> List[EventRecord]:
      return fetch_interruption_metric_records(configs, proper_time_range)

    proper_time_range = determine_time_range(configs)
    metric_records = fetch_interruption_metric_records_task(
        configs,
        proper_time_range,
    )
    log_records = fetch_interruption_log_records(
        configs,
        proper_time_range,
    )
    check_event_count = validate_interruption_count(metric_records, log_records)

    proper_time_range >> [metric_records, log_records] >> check_event_count

    return dag


dag_id_prefix = 'validate_interruption_count'
for platform in [Platform.GCE, Platform.GKE]:
  for reason in InterruptionReason:
    reason_value = reason.name.lower()
    dag_id = f'{dag_id_prefix}_{platform.value}_{reason_value}'
    _ = create_interruption_dag(dag_id, platform, reason)
