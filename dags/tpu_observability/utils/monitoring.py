"""Utility functions for querying Google Cloud Monitoring data."""
import logging
from typing import List, Optional

from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types

from dags.tpu_observability.utils.time_util import TimeUtil


def query_time_series(
    project_id: str,
    filter_str: str,
    start_time: TimeUtil,
    end_time: TimeUtil,
    aggregation: Optional[types.Aggregation] = None,
    view: types.ListTimeSeriesRequest.TimeSeriesView = types.ListTimeSeriesRequest.TimeSeriesView.FULL,
    page_size: Optional[int] = None,
    log_enable: bool = False,
) -> List[types.TimeSeries]:
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
    log_enable: Whether to enable logging. Defaults to False.

  Returns:
    A list of TimeSeries objects matching the query.

  Raises:
    ValueError: If the time format or view string is invalid.
    google.api_core.exceptions.GoogleAPICallError: If the API call fails.
  """
  if log_enable:
    logging.info("Querying monitoring data for project '%s'", project_id)
    logging.info("Filter: %s", filter_str)

  request = monitoring_v3.ListTimeSeriesRequest(
      name=f"projects/{project_id}",
      filter=filter_str,
      interval=types.TimeInterval(
          start_time=start_time.to_timestamp_pb2(),
          end_time=end_time.to_timestamp_pb2(),
      ),
      view=view,
  )

  if aggregation:
    request.aggregation = aggregation
  if page_size:
    request.page_size = page_size

  client = monitoring_v3.MetricServiceClient()
  results = client.list_time_series(request)

  return list(results)
