"""This script validates the consistency of interruption events between metrics and logs."""

import dataclasses
import datetime
import re
from typing import Dict, List, Optional, TypedDict
from airflow import models
from airflow.decorators import task
from google.api_core import exceptions
from google.cloud import logging
from google.cloud import monitoring_v3
from google.protobuf import timestamp_pb2
from interruption_config import get_scenario_config
from interruption_config import InterruptionReason
from interruption_config import ResourceType
from proto import datetime_helpers
import pytz

_UNKNOWN_RESOURCE_NAME = 'Unknown'
SCHEDULED_TIME = None


class ResourceDetail(TypedDict):
  """Details for a single resource record.

  Attributes:
      resource_name: The name of the resource.
      status: The validation status of the resource ("pass" or "fail").
      reason: A string explaining the reason for a failure.
  """

  resource_name: str
  status: str
  reason: Optional[str]


class ValidationResult(TypedDict):
  """Information about the validation status of all the resources.

  Attributes:
      status: The overall validation status of all the resources ("pass" or
        "fail").
      details: A list of `ResourceDetail` objects, each providing details about
        a status related to every resource.
  """

  status: str
  details: List[ResourceDetail]


@dataclasses.dataclass
class EventRecord:
  """Represents Lists of metric points and log events for a single resource."""

  resource_name: str
  interruption_reason: str = ''
  log_filter: str = ''
  metric_points_timestamps: List[str] = dataclasses.field(default_factory=list)
  log_events_timestamps: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ProperTimeRange:
  """They are the proper time range for querying metric data and logs."""

  start_time: str
  end_time: str


@task
def query_metric_data_by_api_task(
    selected_scenario: Dict[str, str],
    proper_time_range: ProperTimeRange,
) -> List[EventRecord]:
  """Airflow task to query metric data by API.

  It's just a wrapper function for query_metric_data_by_api.

  Args:
      selected_scenario: The selected scenario configuration.
      proper_time_range: The proper time range for the query.

  Returns:
      A List of EventRecord objects. Each eventRecord must contain the metric
      points timestamps for the resource name.

  Raises:
      RuntimeError: If any validation fails.
      exceptions.GoogleAPIError: If there's an error communicating with the
      Google Cloud API.
  """
  start_time = datetime.datetime.fromisoformat(proper_time_range.start_time)
  end_time = datetime.datetime.fromisoformat(proper_time_range.end_time)
  return query_metric_data_by_api(
      selected_scenario,
      start_time,
      end_time,
  )


def query_metric_data_by_api(
    selected_scenario: Dict[str, str],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> List[EventRecord]:
  """Fetches node interruption count metrics.

  This function queries the monitoring API for a given selected_scenario and
  time range.

  Args:
      selected_scenario: The selected scenario configuration.
      start_time: The start of the time interval.
      end_time: The end of the time interval.

  Returns:
      A List of EventRecord objects. Each eventRecord must contain the metric
      points timestamps for the resource name.

  Raises:
      RuntimeError: If any validation fails.
      exceptions.GoogleAPIError: If there's an error communicating with the
      Google Cloud API.
  """
  project_id = selected_scenario['project_id']
  metric_filter = selected_scenario['metric_query_filter']
  resource_type_hint = selected_scenario['resource_type_hint']
  aggregation = selected_scenario['metric_aggregation']
  interruption_reason = selected_scenario['interruption_reason']

  project_name = f'projects/{project_id}'
  events_records: dict[str, EventRecord] = {}

  start_timestamp = timestamp_pb2.Timestamp()
  start_timestamp.FromDatetime(start_time)
  end_timestamp = timestamp_pb2.Timestamp()
  end_timestamp.FromDatetime(end_time)

  interval = monitoring_v3.TimeInterval(
      start_time=start_timestamp, end_time=end_timestamp
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
      case ResourceType.GKE:
        resource_key = 'node_name'
      case ResourceType.GCE:
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
              f' {monitoring_v3.TypedValue.pb(point.value).WhichOneof("value")}.'
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
def decide_proper_time_range(
    initial_start_time: datetime.datetime,
    initial_end_time: datetime.datetime,
    selected_scenario: Dict[str, str],
    max_start_time_rewind_seconds: int = 3600,
) -> ProperTimeRange:
  """Decides the proper time range for the validation.

  It will adjust the start_time and end_time to ensure there is idle_time_buffer
  before the earliest record and after the latest record by querying the metric
  data with the monitoring API.
  If the difference between initial_start_time and the adjusted start_time
  is more than max_start_time_rewind_seconds, it will raise a RuntimeError.

  Args:
      initial_start_time: The initial start of the time interval.
      initial_end_time: The initial end of the time interval.
      selected_scenario: The selected scenario configuration.
      max_start_time_rewind_seconds: The maximum time in seconds the start_time
        can be rewound.

  Returns:
      A ProperTimeRange object containing the adjusted start_time,
      end_time which is the proper time range for the validation.

  Raises:
      RuntimeError: If any validation fails.
      exceptions.GoogleAPIError: If there's an error communicating with the
      Google Cloud API.
  """
  max_time_diff_sec = selected_scenario['max_time_diff_sec']

  current_start_time = initial_start_time
  current_end_time = initial_end_time
  idle_time_buffer = datetime.timedelta(seconds=max_time_diff_sec * 2)
  max_rewind_delta = datetime.timedelta(seconds=max_start_time_rewind_seconds)

  while True:
    print(
        '\n current range:'
        f' {current_start_time.isoformat()} to {current_end_time.isoformat()}'
    )

    # Query metric data for the current time range by API.
    metric_records = query_metric_data_by_api(
        selected_scenario,
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
      break
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

  return ProperTimeRange(
      start_time=(current_start_time + idle_time_buffer / 2)
      .astimezone(datetime.timezone.utc)
      .isoformat(),
      end_time=(initial_end_time - idle_time_buffer / 2)
      .astimezone(datetime.timezone.utc)
      .isoformat(),
  )


@task
def query_log_data_by_api(
    selected_scenario: Dict[str, str],
    proper_time_range: ProperTimeRange,
    event_records: List[EventRecord],
) -> List[EventRecord]:
  """Fetches specific interruption logs for instances from gke or gce within a given project and time range.

  It will query the log data from the Google Cloud Logging API using the
  provided log_query_filter and time range. It will then group the log entries
  by resource name and populate the log_events_timestamps field in the
  corresponding EventRecord object.

  Args:
      selected_scenario: The selected scenario configuration.
      proper_time_range: The proper time range for the query.
      event_records: A list of EventRecord objects, containing metric events
      grouped by resource name.

  Returns:
      A list of EventRecord objects. Each EventRecord must contain the log
      events timestamps for the resource name.

  Raises:
      RuntimeError: If any validation fails.
      exceptions.GoogleAPIError: If there's an error communicating with the
      Google Cloud API.
  """
  project_id = selected_scenario['project_id']
  interruption_reason = selected_scenario['interruption_reason']
  log_filter_query = selected_scenario['log_query_filter']
  max_results = selected_scenario['max_log_results']

  logging_api_client = logging.Client(project=project_id)

  start_time_str = proper_time_range.start_time.replace('+00:00', 'Z')
  end_time_str = proper_time_range.end_time.replace('+00:00', 'Z')
  time_range_str = (
      f'timestamp>="{start_time_str}" AND timestamp<="{end_time_str}"'
  )

  resource_filter_query = None
  event_records = {record.resource_name: record for record in event_records}
  for resource_name in event_records:
    if resource_filter_query is None:
      resource_filter_query = f'protoPayload.resourceName=~"^projects\\/[^\\/]+\\/zones\\/[^\\/]+\\/instances\\/{resource_name}$"'
    else:
      resource_filter_query += (
          f' OR protoPayload.resourceName=~"^projects\\/[^\\/]+\\/zones\\/[^\\/]+\\/instances\\/{resource_name}$"'
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
    regex_pattern = r'^projects\/[^\/]+\/zones\/[^\/]+\/instances\/([^\/]+)$'
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
) -> ValidationResult:
  """Checks if the number of metric events matches the number of log events for each resource.

  Args:
      event_records: A list of EventRecord objects, containing metric and log
        events grouped by resource name.

  Returns:
      A dictionary containing the updated validation results.

  Raises:
      RuntimeError: If any validation fails.
  """
  task_failed = False
  current_results: ValidationResult = {
      'status': 'pass',
      'details': [],
  }

  # We are primarily concerned with validating that the number of metric events
  # matches the number of log events.
  for event_record in event_records:
    # Initialize result for the current node. Assume pass until a failure
    # condition is met.
    current_results['details'].append({
        'resource_name': event_record.resource_name,
        'status': 'pass',
        'reason': None,
    })
    details: ResourceDetail = current_results['details'][-1]

    # Check event count match first
    num_metric_events = len(event_record.metric_points_timestamps)
    num_log_events = len(event_record.log_events_timestamps)

    if num_metric_events != num_log_events:
      task_failed = True
      details['status'] = 'fail'
      details['reason'] = (
          f'Event count mismatch. Expected {num_metric_events} metric'
          f' events but found {num_log_events} log events for node'
          f' "{event_record.resource_name}". One-to-one correspondence not'
          ' possible.'
      )
      continue  # Move to the next node_name in event_records

  if task_failed:
    print(current_results)
    raise RuntimeError('Event count mismatch.')

  current_results['status'] = 'pass' if not task_failed else 'fail'
  return current_results


with models.DAG(
    dag_id='interruption_event_validation_dag',
    schedule=SCHEDULED_TIME,
    tags=['interruption', 'validation', 'gcp'],
    start_date=datetime.datetime(2025, 7, 20),
    catchup=False,
) as dag:
  # Define time range for data fetching
  now = datetime.datetime.now(pytz.utc)
  start_time_interval = now - datetime.timedelta(hours=12)
  end_time_interval = now

  # Get scenario configuration
  my_resource_type = ResourceType.GKE
  my_interruption_reason = InterruptionReason.MIGRATE_ON_HWSW_MAINTENANCE
  selected_scenario = get_scenario_config(
      my_resource_type, my_interruption_reason
  )

  if not selected_scenario:
    raise RuntimeError(
        f"Scenario '{my_resource_type.name}_{my_interruption_reason.name}' not"
        ' found. DAG cannot be initialized.'
    )

  proper_time_range = decide_proper_time_range(
      initial_start_time=start_time_interval,
      initial_end_time=end_time_interval,
      selected_scenario=selected_scenario,
      max_start_time_rewind_seconds=3600,  # 1 hour, customize as needed
  )

  event_records_after_get_metrics = query_metric_data_by_api_task(
      selected_scenario=selected_scenario,
      proper_time_range=proper_time_range,
  )

  event_records_after_get_logs_and_metrics = query_log_data_by_api(
      selected_scenario=selected_scenario,
      proper_time_range=proper_time_range,
      event_records=event_records_after_get_metrics,
  )

  check_event_count_match_results = check_event_count_match(
      event_records_after_get_logs_and_metrics
  )

  # --- Task Workflow ---
  (
      proper_time_range
      >> event_records_after_get_metrics
      >> event_records_after_get_logs_and_metrics
      >> check_event_count_match_results
  )
