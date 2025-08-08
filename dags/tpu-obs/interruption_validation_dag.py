"""This script validates the consistency of interruption events between metrics and logs."""

import dataclasses
import datetime
import enum
import re
from typing import ClassVar, Dict, List

from airflow import models
from airflow.decorators import task

from dags.map_reproducibility.utils.constants import Schedule
from dags.multipod.configs.common import Platform
from dags.common.vm_resource import Project


from google.api_core import exceptions
from google.cloud import logging
from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2

from proto import datetime_helpers
import pytz

_UNKNOWN_RESOURCE_NAME = 'Unknown'


class ConfigKey(enum.Enum):
  PROJECT_ID = 'project_id'
  MAX_TIME_DIFF_SEC = 'max_time_diff_sec'
  MAX_LOG_RESULTS = 'max_log_results'
  METRIC_AGGREGATION = 'metric_aggregation'
  DESCRIPTION = 'description'
  OUTPUT_FILENAME = 'output_filename'
  METRIC_QUERY_FILTER = 'metric_query_filter'
  LOG_QUERY_FILTER = 'log_query_filter'
  RESOURCE_TYPE_HINT = 'resource_type_hint'
  INTERRUPTION_REASON = 'interruption_reason'


# --- BASE CONFIGURATION ---
# Common default settings that can be overridden by specific scenarios.
BASE_CONFIG = {
    ConfigKey.PROJECT_ID.value: Project.TPU_PROD_ENV_ONE_VM,  # Default project ID
    ConfigKey.MAX_TIME_DIFF_SEC.value: 150,
    ConfigKey.MAX_LOG_RESULTS.value: 1000,
    ConfigKey.METRIC_AGGREGATION.value: None,  # Default to no aggregation
}


class InterruptionReason(enum.Enum):
  DEFRAGMENTATION = 'Defragmentation'
  EVICTION = 'Eviction'
  HOST_ERROR = 'HostError'
  MIGRATE_ON_HWSW_MAINTENANCE = 'MigrateOnHWSWMaintenance'
  HWSW_MAINTENANCE = 'HWSWMaintenance'
  OTHER = 'Other'
  BARE_METAL_PREEMPTION = 'BareMetalPreemption'
  AUTO_REPAIR = 'AutoRepair'
  AUTO_UPGRADE = 'AutoUpgrade'
  AUTO_RESIZE = 'AutoResize'


# --- RESOURCE TYPE DEFINITIONS ---
# Defines common metric and log properties for different resource types.
RESOURCE_TYPES = {
    Platform.GKE: {
        'metric_type': 'kubernetes.io/node/interruption_count',
        'resource_type': 'k8s_node',
    },
    Platform.GCE: {
        'metric_type': 'tpu.googleapis.com/instance/interruption_count',
        'resource_type': 'tpu.googleapis.com/GceTpuWorker',
    },
}

# --- INTERRUPTION REASON DEFINITIONS ---
# Defines specific metric and log filter parts for different interruption
# reasons.
INTERRUPTION_REASONS = {
    InterruptionReason.DEFRAGMENTATION: {
        'metric_label': 'Defragmentation',
        'log_filter_fragment': (
            'protoPayload.methodName="compute.instances.preempted" '
        ),
    },
    InterruptionReason.EVICTION: {
        'metric_label': 'Eviction',
        'log_filter_fragment': (
            'protoPayload.methodName="compute.instances.preempted" '
        ),
    },
    InterruptionReason.HOST_ERROR: {
        'metric_label': 'HostError',
        'log_filter_fragment': (
            'protoPayload.methodName="compute.instances.hostError" '
        ),
    },
    InterruptionReason.MIGRATE_ON_HWSW_MAINTENANCE: {
        'metric_label': 'Migrate on HW/SW Maintenance',
        'log_filter_fragment': (
            'protoPayload.methodName="compute.instances.migrateOnHostMaintenance" '
        ),
    },
    InterruptionReason.HWSW_MAINTENANCE: {
        'metric_label': 'HW/SW Maintenance',
        'log_filter_fragment': (
            'protoPayload.methodName="compute.instances.terminateOnHostMaintenance" '
        ),
    },
    InterruptionReason.OTHER: {
        'metric_label': 'Other',
        'log_filter_fragment': (
            'protoPayload.methodName="compute.instances.guestTerminate" OR'
            ' protoPayload.methodName="compute.instances.instanceManagerHaltForRestart" OR'
            ' protoPayload.methodName="compute.instances.stoppedDueToPdDoubleServe" OR'
            ' protoPayload.methodName="compute.instances.kmsKeyError" OR'
            ' protoPayload.methodName="compute.instances.shredmillKeyError" OR'
            ' protoPayload.methodName="compute.instances.invalidVmImage" OR'
            ' protoPayload.methodName="compute.instances.scratchDiskCreationFailed" OR'
            ' protoPayload.methodName="compute.instances.localSsdInitializationError" OR'
            ' protoPayload.methodName="compute.instances.localSsdInitializationKeyError" OR'
            ' protoPayload.methodName="compute.instances.localSsdVerifyTarError" OR'
            ' protoPayload.methodName="compute.instances.localSsdRecoveryAttempting" OR'
            ' protoPayload.methodName="compute.instances.localSsdRecoveryTimeoutError" OR'
            ' protoPayload.methodName="compute.instances.localSsdRecoveryFailedError" '
        ),
    },
    InterruptionReason.BARE_METAL_PREEMPTION: {
        'metric_label': 'Bare Metal Preemption',
        'log_filter_fragment': (
            'protoPayload.methodName="compute.instances.baremetalCaretakerPreempted" '
        ),
    },
    InterruptionReason.AUTO_REPAIR: {
        'metric_label': '',
        'log_filter_fragment': '',
    },
    InterruptionReason.AUTO_UPGRADE: {
        'metric_label': '',
        'log_filter_fragment': '',
    },
    InterruptionReason.AUTO_RESIZE: {
        'metric_label': '',
        'log_filter_fragment': '',
    },
}


# Combines BASE_CONFIG, RESOURCE_TYPES, and INTERRUPTION_REASONS to build
# the seleted scenario.
def get_scenario_config(platform: common.Platform, reason: InterruptionReason):
  return _generate_scenario_config(
      platform, reason, RESOURCE_TYPES[platform], INTERRUPTION_REASONS[reason]
  )


def _generate_scenario_config(
    platform_key: common.Platform,
    reason_key: InterruptionReason,
    resource_type_config: dict[str, str],
    reason_config: dict[str, str],
) -> dict[ConfigKey, str]:
  """Generates a single scenario configuration dictionary."""
  scenario_description = (
      f"Validation of {platform_key} "
      f"{'nodes ' if platform_key == 'GKE' else 'instances'}"
      f"{reason_key} interruption's metrics and logs."
  )
  output_filename = f'{platform_key}_{reason_key}_validation_report.json'

  metric_query = (
      f'resource.labels.project_id = "{BASE_CONFIG[ConfigKey.PROJECT_ID.value]}" '
      f'metric.type = "{resource_type_config["metric_type"]}" '
      f'resource.type = "{resource_type_config["resource_type"]}" '
      f'metric.labels.interruption_reason = "{reason_config["metric_label"]}" '
  )

  return {
      ConfigKey.PROJECT_ID: BASE_CONFIG[ConfigKey.PROJECT_ID.value],
      ConfigKey.MAX_TIME_DIFF_SEC: BASE_CONFIG[ConfigKey.MAX_TIME_DIFF_SEC.value],
      ConfigKey.MAX_LOG_RESULTS: BASE_CONFIG[ConfigKey.MAX_LOG_RESULTS.value],
      ConfigKey.METRIC_AGGREGATION: BASE_CONFIG[ConfigKey.METRIC_AGGREGATION.value],
      ConfigKey.DESCRIPTION: scenario_description,
      ConfigKey.OUTPUT_FILENAME: output_filename,
      ConfigKey.METRIC_QUERY_FILTER: metric_query,
      ConfigKey.LOG_QUERY_FILTER: reason_config['log_filter_fragment'],
      ConfigKey.RESOURCE_TYPE_HINT: platform_key,
      ConfigKey.INTERRUPTION_REASON: reason_config['metric_label'],
  }


@dataclasses.dataclass
class EventRecord:
  """Represents lists of metric points and log events for a single resource.

  Attributes:
      overall_status: A class variable indicating whether all validations
        passed.
      failed_resource_reason: A class variable storing reasons for failed
        resources, if any..
      validation_conf: A class variable storing the validation configuration.
      proper_start_time: A class variable storing the proper start time for
        querying metric data and logs.
      proper_end_time: A class variable storing the proper end time for
        querying metric data and logs.
      resource_name: The name of the resource (e.g., node or instance).
      interruption_reason: The reason for the interruption event.
      log_filter: The log query filter used to fetch log events.
      metric_points_timestamps: A list of timestamps for metric points related
        to the resource.
      log_events_timestamps: A list of timestamps for log events related to
        the resource.
  """

  overall_status: ClassVar[bool] = True
  failed_resource_reason: ClassVar[List[str]] = []
  validation_conf: ClassVar[Dict[ConfigKey, str]] = None
  proper_start_time: ClassVar[datetime.datetime] = None
  proper_end_time: ClassVar[datetime.datetime] = None

  resource_name: str
  interruption_reason: str = ''
  log_filter: str = ''
  metric_points_timestamps: List[str] = dataclasses.field(default_factory=list)
  log_events_timestamps: List[str] = dataclasses.field(default_factory=list)


def query_metric_data_by_api(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> List[EventRecord]:
  """Queries the monitoring API for a given validation_conf and time range to retrieve the timeseries data for each resource.

  Args:
      start_time: The start of the time interval.
      end_time: The end of the time interval.

  Returns:
      A List of EventRecord objects. Each eventRecord must contain the metric
      points timestamps for the resource name.
  """
  project_id = EventRecord.validation_conf[ConfigKey.PROJECT_ID]
  metric_filter = EventRecord.validation_conf[ConfigKey.METRIC_QUERY_FILTER]
  resource_type_hint = EventRecord.validation_conf[ConfigKey.RESOURCE_TYPE_HINT]
  aggregation = EventRecord.validation_conf[ConfigKey.METRIC_AGGREGATION]
  interruption_reason = EventRecord.validation_conf[ConfigKey.INTERRUPTION_REASON]

  project_name = f'projects/{project_id}'
  events_records: dict[str, EventRecord] = {}

  start_timestamp = timestamp_pb2.Timestamp()
  start_timestamp.FromDatetime(start_time)
  end_timestamp = timestamp_pb2.Timestamp()
  end_timestamp.FromDatetime(end_time)

  interval = monitoring_v3.TimeInterval(
      start_time=start_timestamp,
      end_time=end_timestamp
  )

  request = monitoring_v3.ListTimeSeriesRequest(
      name=project_name,
      filter=metric_filter,
      interval=interval,
      view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  # If aggregation is provided, add it to the request arguments
  if aggregation:
    request.aggregation = aggregation

  monitoring_api_client = monitoring_v3.MetricServiceClient()
  # Here should raise the exception from the API. We don't catch it, just let
  # it raise to Airflow.
  response = monitoring_api_client.list_time_series(request=request)

  for time_series in response:
    match resource_type_hint:
      case Platform.GKE:
        resource_key = 'node_name'
      case Platform.GCE:
        resource_key = 'instance_name'
      case _:
        print(f"Warning: Unknown resource_type_hint '{resource_type_hint}'.")
        raise RuntimeError(
            f"Unsupported resource_type_hint '{resource_type_hint}'. "
            'Please check the scenario configuration.'
        )

    resource_name = time_series.resource.labels.get(
        resource_key, _UNKNOWN_RESOURCE_NAME
    )

    # Ensure we actually got a name before proceeding
    if resource_name == _UNKNOWN_RESOURCE_NAME:
      raise RuntimeError(
          'Could not determine node/instance name for time series. '
          f'Failed to extract name for resource type "{resource_type_hint}". '
          f'Time series data: {time_series}'
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
          raise RuntimeError(
              'Unexpected TypedValue:'
              f" {monitoring_v3.TypedValue.pb(point.value).WhichOneof('value')}."
              f' Full point data: {point}'
          )

      if event_count > 0:
        aware_timestamp = end_time_obj.replace(tzinfo=datetime.timezone.utc)
        for _ in range(event_count):
          if resource_name in events_records:
            found_record = events_records[resource_name]
            found_record.metric_points_timestamps.append(
                aware_timestamp.astimezone(datetime.timezone.utc).isoformat()
            )
          else:
            new_record = EventRecord(
                resource_name=resource_name,
                interruption_reason=interruption_reason,
            )
            new_record.metric_points_timestamps.append(
                aware_timestamp.astimezone(datetime.timezone.utc).isoformat()
            )
            events_records[resource_name] = new_record

  if not events_records:
    print(
        'No metric events found in the specified time range. Validation cannot'
        ' proceed.'
    )
    raise RuntimeError('No metric events found in the specified time range.')

  return list(events_records.values())


@task
def fetch_interruption_metrics_timestamps(
    max_start_time_rewind_seconds: int = 3600,
) -> List[EventRecord]:
  """Adjusts the proper time range for the validation and fetches the metric data with the proper time range.

  It will adjust the start_time and end_time to ensure there is idle_time_buffer
  before the earliest record and after the latest record by querying the metric
  data with the monitoring API.
  If the difference between the initial start_time and the adjusted start_time
  is more than max_start_time_rewind_seconds, it will raise a RuntimeError.

  Args:
      max_start_time_rewind_seconds: The maximum time in seconds the start_time
        can be rewound.

  Returns:
      A List of EventRecord objects.
  """
  max_time_diff_sec = EventRecord.validation_conf[ConfigKey.MAX_TIME_DIFF_SEC]
  initial_start_time = EventRecord.proper_start_time
  initial_end_time = EventRecord.proper_end_time

  current_start_time = initial_start_time
  current_end_time = initial_end_time
  idle_time_buffer = datetime.timedelta(seconds=max_time_diff_sec * 2)
  max_rewind_delta = datetime.timedelta(seconds=max_start_time_rewind_seconds)

  # Continue to adjust the time range until the time range has stabilized or
  # the start time rewind has reached the maximum limit.
  while True:
    print(
        '\n current range:'
        f' {current_start_time.isoformat()} to {current_end_time.isoformat()}'
    )

    # Query metric data for the current time range by API.
    metric_records = query_metric_data_by_api(
        # validation_conf,
        current_start_time,
        current_end_time,
    )

    total_metric_timestamps = []
    for record in metric_records:
      total_metric_timestamps.extend(record.metric_points_timestamps)
    total_metric_timestamps.sort(
        key=lambda timestamp: (datetime.datetime.fromisoformat(timestamp))
    )

    first_record_time = datetime.datetime.fromisoformat(
        total_metric_timestamps[0]
    )
    last_record_time = datetime.datetime.fromisoformat(
        total_metric_timestamps[-1]
    )

    new_start_time = current_start_time
    new_end_time = current_end_time

    # Adjust start time (ensure there is idle_time_buffer before the earliest
    # record). The new start_time should be first_record_time -
    # idle_time_buffer.
    calculated_new_start_time = first_record_time - idle_time_buffer

    # Allow current_start_time to be pushed forward as long as it does not
    # exceed max_rewind_delta. Update only when calculated_new_start_time is
    # earlier than current_start_time.
    if calculated_new_start_time < current_start_time:
      new_start_time = calculated_new_start_time.replace(microsecond=0)

    temp_records = total_metric_timestamps.copy()

    # If the previous_last_record_time is max, then potential_new_end >
    # previous_last_record_time will always be false. (for the first iteration)
    previous_last_record_time = datetime.datetime.max.replace(
        tzinfo=datetime.timezone.utc
    )

    while temp_records:
      current_last_record = datetime.datetime.fromisoformat(temp_records[-1])
      potential_new_end = current_last_record + idle_time_buffer

      # If potential_new_end is still later than current_end_time or the
      # previous last record time (except for the first iteration), this means
      # that even if the current last record is taken as the basis, the buffer
      # exceeds current_end_time or the previous last record time.
      # Since end_time cannot be extended, this current_last_record cannot be
      # our final last record.
      # We need to remove the current_last_record and try again.
      if (
          potential_new_end > current_end_time
          or potential_new_end > previous_last_record_time
      ):
        print(
            f'The last record time {current_last_record.isoformat()} + buffer'
            f' {idle_time_buffer} exceeds the current end_time'
            f' {current_end_time.isoformat()} or the previous last record time'
            ' {previous_last_record_time.isoformat()}. Removing this record'
            ' and recalculating.'
        )
        previous_last_record_time = datetime.datetime.fromisoformat(
            temp_records.pop()
        )

        if not temp_records:
          raise RuntimeError(
              'All records have been removed, and a suitable end_time cannot'
              ' be found.'
          )
      else:
        # If potential_new_end is less than or equal to current_end_time
        # This means that based on the current last record, the buffer does not
        # exceed current_end_time.
        # This is the new end_time we want.
        new_end_time = potential_new_end.replace(microsecond=0)
        break

    print(f'Earliest record time: {first_record_time.isoformat()}')
    print(f'Latest record time: {last_record_time.isoformat()}')
    print(f'Calculated new start time: {new_start_time.isoformat()}')
    print(f'Calculated new end time: {new_end_time.isoformat()}')

    # confirm whether the time range has stabilized
    if (
        new_start_time == current_start_time
        and new_end_time == current_end_time
    ):
      print('Time range has stabilized, no need to adjust again.')

      # Update the proper_time_range with the new start_time and end_time.
      EventRecord.proper_start_time = current_start_time + idle_time_buffer / 2
      EventRecord.proper_end_time = current_end_time - idle_time_buffer / 2
      return metric_records
    else:
      current_start_time = new_start_time
      current_end_time = new_end_time
      print('Time range has been adjusted, need to check again.')

    if (initial_start_time - current_start_time) > max_rewind_delta:
      print(
          'Start time rewind has reached the maximum limit'
          f' ({max_start_time_rewind_seconds} seconds), terminating adjustment.'
      )
      raise RuntimeError('Start time rewind has reached the maximum limit.')


@task
def fetch_interruption_logs_timestamps(
    event_records: List[EventRecord],
) -> List[EventRecord]:
  """This function queries the Logging API for a given validation_conf and time range to fetch the log entries.

  Args:
      event_records: A list of EventRecord objects, containing metric events
      grouped by resource name.

  Returns:
      A list of EventRecord objects. Each EventRecord must contain the log
      events timestamps for the resource name.
  """
  project_id = EventRecord.validation_conf[ConfigKey.PROJECT_ID]
  interruption_reason = EventRecord.validation_conf[ConfigKey.INTERRUPTION_REASON]
  log_filter_query = EventRecord.validation_conf[ConfigKey.LOG_QUERY_FILTER]
  max_results = EventRecord.validation_conf[ConfigKey.MAX_LOG_RESULTS]

  logging_api_client = logging.Client(project=project_id)

  # start_time_str = EventRecord.proper_time_range.start_time.replace('+00:00', 'Z')
  # end_time_str = EventRecord.proper_time_range.end_time.replace('+00:00', 'Z')
  start_time_str = (
      EventRecord.proper_start_time.astimezone(datetime.timezone.utc)
      .isoformat().replace('+00:00', 'Z')
  )
  end_time_str = (
      EventRecord.proper_end_time.astimezone(datetime.timezone.utc)
      .isoformat().replace('+00:00', 'Z')
  )
  time_range_str = (
      f'timestamp>="{start_time_str}" AND timestamp<="{end_time_str}"'
  )

  resource_filter_query = None
  event_records = {record.resource_name: record for record in event_records}
  for resource_name in event_records:
    if resource_filter_query is None:
      resource_filter_query = f'protoPayload.resourceName=~"^projects/[\\w-]+/zones/[\\w-]+/instances/{resource_name}$"'
    else:
      resource_filter_query += (
          f' OR protoPayload.resourceName=~"^projects/[\\w-]+/zones/[\\w-]+/instances/{resource_name}$"'
      )
  if resource_filter_query:
    log_filter_query = f'{log_filter_query} AND ({resource_filter_query})'

  try:
    log_entries = logging_api_client.list_entries(
        filter_=f'({time_range_str}) AND ({log_filter_query})',
        order_by=logging.DESCENDING,
        max_results=max_results,
    )
  except exceptions.GoogleAPIError as e:
    print(f'Error querying log data from Google Cloud Logging API: {e}')
    raise e

  entry_count = 0
  for entry in log_entries:
    entry_count += 1
    # The 'resourceName' in the log entry payload typically looks like:
    # "projects/{project_id}/zones/{zone}/instances/{node_name}"
    regex_pattern = r'^projects/[\w-]+/zones/[\w-]+/instances/([\w-]+)$'
    resource_name = entry.payload.get('resourceName', '')
    match = re.match(regex_pattern, resource_name)
    if match:
      log_node_name = match.group(1)
      if log_node_name:
        aware_timestamp = entry.timestamp.replace(tzinfo=datetime.timezone.utc)

        if log_node_name in event_records:
          found_record = event_records[log_node_name]
          found_record.log_filter = log_filter_query
          found_record.log_events_timestamps.append(
              aware_timestamp.astimezone(datetime.timezone.utc).isoformat()
          )
        else:
          new_record = EventRecord(
              resource_name=log_node_name,
              interruption_reason=interruption_reason,
              log_filter=log_filter_query,
          )
          new_record.log_events_timestamps.append(
              aware_timestamp.astimezone(datetime.timezone.utc).isoformat()
          )
          event_records[log_node_name] = new_record

  # Check if we hit the max_results limit
  if max_results is not None and entry_count == max_results:
    raise RuntimeError(
        f'Log entries limit reached ({max_results} entries). '
        'This might indicate we are missing data. '
        'Consider increasing max_results or narrowing the time range.'
    )
  if not event_records:
    raise RuntimeError('No log entries found in the specified time range.')

  return list(event_records.values())


@task
def check_event_count_match(
    event_records: List[EventRecord],
) -> List[EventRecord]:
  """Checks if the number of metric events matches the number of log events for each resource.

  Args:
      event_records: A list of EventRecord objects, containing metric and log
        events grouped by resource name.

  Returns:
      A list of EventRecord objects, containing the updated validation results.
  """
  # We are primarily concerned with validating that the number of metric events
  # matches the number of log events.
  for event_record in event_records:
    # Check event count match first
    num_metric_events = len(event_record.metric_points_timestamps)
    num_log_events = len(event_record.log_events_timestamps)

    if num_metric_events != num_log_events:
      EventRecord.overall_status = False
      EventRecord.failed_resource_reason.append(
          f'Event count mismatch. Expected {num_metric_events} metric'
          f' events but found {num_log_events} log events for node'
          f' "{event_record.resource_name}". One-to-one correspondence not'
          ' possible.'
      )

      continue  # Move to the next node_name in event_records

  if not EventRecord.overall_status:
    print(
        'EventRecord.failed_resource_reason:'
        f' {EventRecord.failed_resource_reason}'
    )
    raise RuntimeError('Overall status is failed.')

  return event_records


with models.DAG(
    dag_id='interruption_event_validation_dag',
    start_date=datetime.datetime(2025, 7, 20),
    schedule=Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=['gke', 'gce', 'tpu-observability', 'interruption_validation'],
    description=(
        'This DAG validates the interruption event metrics and logs for GKE and GCE'
    ),
    doc_md="""
    ### Interruption Event Validation DAG
    This DAG validates the consistency of interruption events between metrics and logs for both GKE and GCE environments.
    It performs the following steps:
    1.  **Fetch Interruption Metrics Timestamps**: Queries Cloud Monitoring API to fetch interruption metric timestamps, adjusting the time range to ensure proper buffer before the earliest and after the latest events.
    2.  **Fetch Interruption Logs Timestamps**: Queries Cloud Logging API to fetch interruption log timestamps, filtering by the resources identified in the metrics query.
    3.  **Check Event Count Match**: Validates if the number of metric events matches the number of log events for each resource.
    """,
) as dag:
  # Define time range for data fetching
  now = datetime.datetime.now(pytz.utc)
  start_time_interval = now - datetime.timedelta(hours=12)
  end_time_interval = now

  EventRecord.proper_start_time = start_time_interval
  EventRecord.proper_end_time = end_time_interval

  # Get scenario configuration
  my_resource_type = Platform.GKE
  my_interruption_reason = InterruptionReason.MIGRATE_ON_HWSW_MAINTENANCE
  EventRecord.validation_conf = get_scenario_config(
      my_resource_type, my_interruption_reason
  )

  event_records_after_get_metrics = fetch_interruption_metrics_timestamps(
      max_start_time_rewind_seconds=3600,  # 1 hour, customize as needed
  )

  event_records_after_get_logs_and_metrics = fetch_interruption_logs_timestamps(
      event_records=event_records_after_get_metrics,
  )
  event_records_after_check_count_match = check_event_count_match(
      event_records_after_get_logs_and_metrics
  )

  # --- Task Workflow ---
  (
      event_records_after_get_metrics
      >> event_records_after_get_logs_and_metrics
      >> event_records_after_check_count_match
  )


