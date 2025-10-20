"""
Utility functions for querying Google Cloud data.

This module uses the **`tenacity`** package to implement a retry mechanism
for API calls. This is specifically designed to handle **Quota issues**
(e.g., hitting the limit for requests per minute).
When a specific error type and message indicating a quota limit is encountered,
the function will wait for a set period and then automatically retry the data query,
improving resilience.
"""
import logging as logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    RetryCallState,
)

from google.cloud import monitoring_v3, logging_v2
from google.cloud.logging_v2 import types as logging_types
from google.cloud.monitoring_v3 import types as monitoring_types
from google.api_core.exceptions import ResourceExhausted
from google.api.error_reason_pb2 import ErrorReason

from dags.tpu_observability.utils.time_util import TimeUtil


READ_QUOTA_EXCEED_ERROR = ErrorReason.Name(ErrorReason.RATE_LIMIT_EXCEEDED)


def api_quota_exceeded(exception: BaseException) -> bool:
  check_instance: bool = isinstance(exception, ResourceExhausted)
  check_message: bool = READ_QUOTA_EXCEED_ERROR in str(exception)
  return check_instance and check_message


def retry_log_before_sleep(rs: RetryCallState) -> None:
  e = rs.outcome.exception() if rs.outcome else None
  logger.info(
      f"QUOTA HIT!!! Attempt {rs.attempt_number} failed with {e}. "
      f"Retrying in {rs.idle_for:.2f} seconds..."
  )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=60, min=60, max=600),
    retry=retry_if_exception(api_quota_exceeded),
    before_sleep=retry_log_before_sleep,
)
def query_time_series(
    project_id: str,
    filter_str: str,
    start_time: TimeUtil,
    end_time: TimeUtil,
    aggregation: monitoring_types.Aggregation | None = None,
    view: monitoring_types.ListTimeSeriesRequest.TimeSeriesView = monitoring_types.ListTimeSeriesRequest.TimeSeriesView.FULL,
    page_size: int | None = 500,
    log_enable: bool = False,
) -> list[monitoring_types.TimeSeries]:
  """A utility that queries metrics (time series data) from Google Cloud Monitoring API.

  This function provides a flexible interface to the list_time_series API,
  with robust error handling and convenient parameter types.

  Args:
    project_id: The Google Cloud project ID.
    filter_str: A Cloud Monitoring filter string that specifies which time
      series should be returned.
    start_time: The start of the time interval. Can be a datetime object, an
      ISO 8601 string, or a Unix timestamp (int/float).
    end_time: The end of the time interval. Can be a datetime object, an ISO
      8601 string, or a Unix timestamp (int/float).
    aggregation: An Aggregation object that specifies how to align and combine
      time series. Defaults to None (raw data).
    view: The level of detail to return. Can be the TimeSeriesView enum (e.g.,
      TimeSeriesView.FULL) or a string ("FULL", "HEADERS"). Defaults to FULL.
    page_size: The maximum number of results to return per page.
      The API's default is 50, we use 500 to decrease the total number of requests
      to avoid the quota issue.
    log_enable: Whether to enable logging. Defaults to False.

  Returns:
    A list of TimeSeries objects matching the query.

  Raises:
    ValueError: If the time format or view string is invalid.
    google.api_core.exceptions.GoogleAPICallError: If the API call fails.
  """
  if log_enable:
    logger.info("Querying monitoring data for project '%s'", project_id)
    logger.info("Filter: %s", filter_str)

  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{project_id}",
      filter=filter_str,
      interval=monitoring_types.TimeInterval(
          start_time=start_time.to_timestamp_pb2(),
          end_time=end_time.to_timestamp_pb2(),
      ),
      page_size=page_size,
      view=view,
  )

  if aggregation:
    request.aggregation = aggregation

  client = monitoring_v3.MetricServiceClient()
  results = client.list_time_series(request)

  return list(results)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=60, min=60, max=600),
    retry=retry_if_exception(api_quota_exceeded),
    before_sleep=retry_log_before_sleep,
)
def query_log_entries(
    project_id: str,
    filter_str: str,
    start_time: TimeUtil,
    end_time: TimeUtil,
    order_by: str | None = logging_v2.DESCENDING,
    max_results: int | None = None,
    page_size: int | None = 500,
    log_enable: bool = False,
) -> list[logging_types.LogEntry]:
  """Queries log entries from Google Cloud Logging API.

  Args:
    project_id: The Google Cloud project ID.
    filter_str: A Cloud logging filter string that specifies which log entries
      should be returned.
    start_time: The start of the time interval. Can be a datetime object, an
      ISO 8601 string, or a Unix timestamp (int/float).
    end_time: The end of the time interval. Can be a datetime object, an ISO
      8601 string, or a Unix timestamp (int/float).
    order_by: Optional. How to order the results (e.g., "timestamp desc").
      Defaults to descending timestamp.
    max_results: Optional. The maximum number of results to return overall.
    page_size: The maximum number of results to return per page.
      The API's default is 50, we use 500 to decrease the total number of requests
      to avoid the quota issue.
    log_enable: Whether to enable logging. Defaults to False.

  Returns:
    A list of LogEntry objects matching the query.

  Raises:
    ValueError: If the time format is invalid.
    google.api_core.exceptions.GoogleAPICallError: If the API call fails.
  """
  if log_enable:
    logger.info("Querying logging data for project '%s'", project_id)
    logger.info("Filter: %s", filter_str)

  logging_api_client = logging_v2.Client(project=project_id)

  filter_list = [
      f'timestamp>="{start_time.to_iso_string()}"',
      f'timestamp<="{end_time.to_iso_string()}"',
      f"({filter_str})",
  ]

  log_entries = logging_api_client.list_entries(
      filter_=" AND ".join(filter_list),
      order_by=order_by,
      max_results=max_results,
      page_size=page_size,
  )

  return list(log_entries)
