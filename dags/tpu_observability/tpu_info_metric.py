# Copyright 2025 Google LLC
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
"""Metric verification strategies for the tpu-info CLI tool."""

from abc import ABC, abstractmethod
import enum
import logging
import re
import textwrap
from typing import Any

from google.cloud import monitoring_v3
from google.cloud.monitoring_v3 import types as monitoring_types
from airflow.exceptions import AirflowException

from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.gcp_util import list_time_series, query_time_series


class _Percentiles(enum.Enum):
  P50 = 50
  P90 = 90
  P95 = 95
  P999 = 99.9


class BaseMetricStrategy(ABC):
  """Abstract Base Class (Interface) for a metric verification strategy.

  It defines the contract that all concrete metric strategies must follow.
  """

  metric_name: str
  tpu_info_metric_name: str
  dag_id_suffix: str
  tolerance_percent: float = 3.0

  @abstractmethod
  def list_or_query_metric(
      self,
      project_id: str,
      cluster_name: str,
      pod_name: str,
      start_time: TimeUtil,
      end_time: TimeUtil,
  ) -> list[monitoring_types.TimeSeries]:
    """Fetches metric data from Cloud Monitoring via ListTimeSeries or MQL."""
    pass

  @abstractmethod
  def parse_from_monitoring(
      self, time_series_data: list[monitoring_types.TimeSeries], **kwargs
  ) -> list[float]:
    """Parses the desired value from a list of TimeSeries objects."""
    pass

  @abstractmethod
  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[tpu_info.Table]
  ) -> list[float]:
    """Parses percentile values from the output table of the tpu-info tool.

    Args:
      tpu_info_metric_output: A list of tables from the tpu-info command output.

    Returns:
      A list of float values representing the parsed percentiles, ordered by
      buffer size and then by percentile.
    """
    pass


class _BaseSimplePointStrategy(BaseMetricStrategy):
  """Base strategy for parsing single numeric data points from Cloud Monitoring.

  This abstract base class encapsulates the common logic for processing metrics
  that consist of scalar values (int64 or double). It handles iterating through
  time series, extracting the value from the first point, type checking,
  and formatting the final output.
  """

  def _process_value(self, raw_value: int | float) -> float:
    """(Optional) Processes the extracted raw value.

    Default behavior: Simply converts it to float.

    Args:
      raw_value: The raw value extracted from the monitoring data.

    Returns:
      The processed value as a float.
    """
    return float(raw_value)

  def list_or_query_metric(
      self,
      project_id: str,
      cluster_name: str,
      pod_name: str,
      start_time: TimeUtil,
      end_time: TimeUtil,
  ) -> list[monitoring_types.TimeSeries]:
    filter_string = [
        f'metric.type = "{self.metric_name}"',
        f'resource.labels.cluster_name = "{cluster_name}"',
        f'resource.labels.pod_name = "{pod_name}"',
    ]

    return list_time_series(
        project_id=project_id,
        filter_str=" AND ".join(filter_string),
        start_time=start_time,
        end_time=end_time,
        view=monitoring_types.ListTimeSeriesRequest.TimeSeriesView.FULL,
    )

  def parse_from_monitoring(
      self, time_series_data: list[monitoring_types.TimeSeries]
  ) -> list[float]:
    """Parses raw time series data into a sorted list of float values.

    This method iterates through the provided time series, extracts the most
    recent data point (assuming index 0), and handles type conversion for
    both `int64` and `double` value types.

    The results are collected into a map keyed by 'accelerator_id' and then
    returned as a list, sorted by the numeric suffix of the accelerator ID
    (e.g., ensuring 'chip-10' comes after 'chip-2').

    Args:
      time_series_data: A list of TimeSeries objects returned from the
        Cloud Monitoring API.

    Returns:
      A list of processed metric values (floats rounded to 2 decimal places),
      ordered by the accelerator ID.

    Raises:
      AirflowException: If the metric value type is neither 'int64_value' nor
        'double_value'.
    """
    metric_values = {}
    raw_value: int | float

    for ts in time_series_data:
      if not ts.points:
        continue

      accelerator_id = ts.metric.labels["accelerator_id"]
      point = ts.points[0]

      match monitoring_v3.TypedValue.pb(point.value).WhichOneof("value"):
        case "int64_value":
          raw_value = point.value.int64_value
        case "double_value":
          raw_value = point.value.double_value
        case _:
          raise AirflowException(
              f"Unexpected metric value type of {self.metric_name}"
          )
      processed_value = self._process_value(raw_value)
      metric_values[accelerator_id] = round(processed_value, 2)

    return [
        metric_values[key]
        for key in sorted(
            metric_values.keys(), key=lambda x: int(x.split("-")[-1])
        )
    ]


class _BaseDistributionStrategy(BaseMetricStrategy):
  """Base strategy for parsing distribution (histogram) data and calculating percentiles.

  This abstract base class handles the common logic for processing metrics
  represented as distributions in Cloud Monitoring. It calculates specific
  percentiles (e.g., P50, P99) from the raw histogram data and parses
  corresponding percentile values from `tpu-info` command output tables.
  """

  _monitoring_group_by_label: str
  _tpu_info_table_name: str
  _tpu_info_group_by_key: str
  percentiles_to_check = list(_Percentiles)
  uses_mql = True

  def list_or_query_metric(
      self,
      project_id: str,
      cluster_name: str,
      pod_name: str,
      start_time: TimeUtil,
      end_time: TimeUtil,
  ) -> list[monitoring_types.TimeSeries]:
    aggregators = []
    for p in self.percentiles_to_check:
      aggregators.append(f"{p.name}: percentile(val(), {p.value})")

    aggregator_str = ",\n        ".join(aggregators)
    target_group_label = f"metric.{self._monitoring_group_by_label}"

    query = textwrap.dedent(
        f"""
        fetch k8s_container
        | metric '{self.metric_name}'
        | filter (
            resource.cluster_name == '{cluster_name}'
            && resource.pod_name == '{pod_name}'
        )
        | within {start_time.to_mql_string()}, {end_time.to_mql_string()}
        | align delta(1m)
        | every 1m
        | group_by [resource.pod_name, {target_group_label}],
            [{aggregator_str}]
        """
    ).strip()

    logging.info("Executing MQL Query:\n%s", query)
    return query_time_series(project_id, query)

  def parse_from_monitoring(
      self, time_series_data: list[monitoring_types.TimeSeries]
  ) -> list[float]:
    """Parses distribution data from Monitoring and calculates requested percentiles.

    This method iterates through time series data, extracting distribution
    values (count, bucket bounds, bucket counts). It groups these distributions
    by a specific label (defined in subclasses) and calculates the target
    percentiles for each group using the histogram data.

    Args:
      time_series_data: A list of TimeSeries objects containing distribution
      data.

    Returns:
      A flattened list of calculated percentile values (floats). The list is
      ordered first by the sorted group key, and then by the sorted percentiles
      for each group.
    """

    if not time_series_data:
      return []

    parsed_data = {}

    for ts_data in time_series_data:
      if len(ts_data.label_values) < 2:
        continue

      group_key_value = ts_data.label_values[1].string_value

      if not ts_data.point_data:
        continue

      latest_point = ts_data.point_data[0]

      parsed_data[group_key_value] = {}

      for idx, p in enumerate(self.percentiles_to_check):
        val = latest_point.values[idx].double_value
        parsed_data[group_key_value][p.name] = val

    monitoring_values = []
    for group_key in sorted(parsed_data.keys()):
      for p in self.percentiles_to_check:
        monitoring_values.append(parsed_data[group_key][p.name])

    return monitoring_values

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[tpu_info.Table]
  ) -> list[float]:
    """Parses pre-calculated percentile values from `tpu-info` output tables.

    This method locates the specific table (defined in subclasses) in the
    `tpu-info` output and extracts values for the requested percentiles.
    It handles parsing numeric values from string cells (e.g., "123.45 us")
    and mapping specific column names (like "P999" for P99.9).

    Args:
      tpu_info_metric_output: A list of parsed Table objects from the `tpu-info`
      command.

    Returns:
      A flattened list of percentile values (floats) extracted from the table.
      The list is ordered first by the sorted group key, and then by the sorted
      percentiles for each group.
    """
    parsed_values_by_group: dict[str, dict[float, float]] = {}
    table_name = self._tpu_info_table_name
    group_key = self._tpu_info_group_by_key

    for metric_table in tpu_info_metric_output:
      if metric_table.name == table_name:
        for row_dict in metric_table.body:
          group_value = row_dict.get(group_key)
          if not group_value:
            continue

          parsed_values_by_group[group_value] = {}
          for p in self.percentiles_to_check:
            # Use Enum name as key (e.g. "P999")
            value_str = row_dict.get(p.name, "")

            match = re.search(r"([\d\.]+)", value_str)
            if match:
              parsed_values_by_group[group_value][p.name] = float(
                  match.group(1)
              )

    tpu_info_data_values = []
    for group_value in sorted(parsed_values_by_group.keys()):
      for p in self.percentiles_to_check:
        tpu_info_data_values.append(parsed_values_by_group[group_value][p.name])

    return tpu_info_data_values


class MemoryUsedStrategy(_BaseSimplePointStrategy):
  """Strategy for verifying Used HBM Memory."""

  metric_name = "kubernetes.io/container/accelerator/memory_used"
  tpu_info_metric_name = "hbm_usage"
  dag_id_suffix = "memory_used"
  tolerance_percent = 1.0
  _monitoring_value_type_key = "int64_value"

  def _process_value(self, raw_value: Any) -> float:
    return raw_value / (1024**3)

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[tpu_info.Table]
  ) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU HBM Usage":
        for row_dict in metric_table.body:
          hbm_value = row_dict["HBM Usage (GiB)"]
          # Parses HBM usage from "USED.XX GiB / TOTAL.XX GiB".
          # Group 1 captures the USED memory value.
          # Group 2 captures the TOTAL memory value.
          match = re.search(
              r"(\d+\.\d+)\s*GiB\s*\/\s*(\d+\.\d+)\s*GiB", hbm_value
          )
          if match:
            tpu_info_data_values.append(float(match.group(1)))
    return tpu_info_data_values


class MemoryTotalStrategy(_BaseSimplePointStrategy):
  """Strategy for verifying Total HBM Memory."""

  metric_name = "kubernetes.io/container/accelerator/memory_total"
  tpu_info_metric_name = "hbm_usage"
  dag_id_suffix = "memory_total"
  tolerance_percent = 0.0
  _monitoring_value_type_key = "int64_value"

  def _process_value(self, raw_value: Any) -> float:
    return raw_value / (1024**3)

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[tpu_info.Table]
  ) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU HBM Usage":
        for row_dict in metric_table.body:
          hbm_value = row_dict["HBM Usage (GiB)"]
          # Regex to parse the HBM usage string format:
          # "USED.XX GiB / TOTAL.XX GiB".
          # Group 1 captures the USED memory value.
          # Group 2 captures the TOTAL memory value.
          match = re.search(
              r"(\d+\.\d+)\s*GiB\s*\/\s*(\d+\.\d+)\s*GiB", hbm_value
          )
          if match:
            tpu_info_data_values.append(float(match.group(2)))
    return tpu_info_data_values


class DutyCycleStrategy(_BaseSimplePointStrategy):
  """Strategy for verifying Duty Cycle."""

  metric_name = "kubernetes.io/container/accelerator/duty_cycle"
  tpu_info_metric_name = "duty_cycle_percent"
  dag_id_suffix = "duty_cycle"
  tolerance_percent = 1.0
  _monitoring_value_type_key = "int64_value"

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[tpu_info.Table]
  ) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TPU Duty Cycle":
        for row_dict in metric_table.body:
          dutycycle_value = row_dict["Duty Cycle (%)"]
          match = re.search(r"(\d+\.\d+)%", dutycycle_value)
          if match:
            tpu_info_data_values.append(float(match.group(1)))
    return tpu_info_data_values


class TensorcoreUtilizationStrategy(_BaseSimplePointStrategy):
  """Strategy for verifying TensorCore Utilization."""

  metric_name = "kubernetes.io/container/accelerator/tensorcore_utilization"
  tpu_info_metric_name = "tensorcore_utilization"
  dag_id_suffix = "tensorcore_utilization"
  tolerance_percent = 15.0
  _monitoring_value_type_key = "double_value"

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[tpu_info.Table]
  ) -> list[float]:
    tpu_info_data_values = []
    for metric_table in tpu_info_metric_output:
      if metric_table.name == "TensorCore Utilization":
        for row_dict in metric_table.body:
          tcu_value = row_dict["TensorCore Utilization"].replace("%", "")
          tpu_info_data_values.append(float(tcu_value))
    return tpu_info_data_values


class BufferTransferLatencyStrategy(_BaseDistributionStrategy):
  """Strategy for verifying Buffer Transfer Latency from distribution data."""

  metric_name = (
      "kubernetes.io/container/multislice/network/dcn_transfer_latencies"
  )
  tpu_info_metric_name = "buffer_transfer_latency"
  dag_id_suffix = "buffer_transfer_latency"
  tolerance_percent = 3.0
  _monitoring_group_by_label = "buffer_size"
  _tpu_info_table_name = "TPU Buffer Transfer Latency"
  _tpu_info_group_by_key = "Buffer Size"


class HostToDeviceTransferLatenciesStrategy(_BaseDistributionStrategy):
  """Strategy for verifying Host to Device Transfer Latency from distribution data."""

  metric_name = "kubernetes.io/container/multislice/accelerator/host_to_device_transfer_latencies"
  tpu_info_metric_name = "host_to_device_transfer_latency"
  dag_id_suffix = "host_to_device_transfer_latency"
  tolerance_percent = 3.0
  _monitoring_group_by_label = "buffer_size"
  _tpu_info_table_name = "TPU Host to Device Transfer Latency"
  _tpu_info_group_by_key = "Buffer Size"


class DeviceToHostTransferLatenciesStrategy(_BaseDistributionStrategy):
  """
  Strategy for verifying Device to Host Transfer Latency from distribution data.
  """

  metric_name = "kubernetes.io/container/multislice/accelerator/device_to_host_transfer_latencies"
  tpu_info_metric_name = "device_to_host_transfer_latency"
  dag_id_suffix = "device_to_host_transfer_latency"
  tolerance_percent = 3.0
  _monitoring_group_by_label = "buffer_size"
  _tpu_info_table_name = "TPU Device to Host Transfer Latency"
  _tpu_info_group_by_key = "Buffer Size"


class CollectiveEndToEndLatencyLatenciesStrategy(_BaseDistributionStrategy):
  """
  Strategy for verifying Collective End to End Latency from distribution data.
  """

  metric_name = "kubernetes.io/container/multislice/network/collective_end_to_end_latencies"
  tpu_info_metric_name = "collective_e2e_latency"
  dag_id_suffix = "collective_e2e_latency"
  tolerance_percent = 3.0
  _monitoring_group_by_label = "collective_type"
  _tpu_info_table_name = "TPU Collective End to End Latency"
  _tpu_info_group_by_key = "Buffer Size"

  def parse_from_tpu_info(
      self, tpu_info_metric_output: list[tpu_info.Table]
  ) -> list[float]:
    parsed_values_by_buffer: dict[str, dict[float, float]] = {}

    for metric_table in tpu_info_metric_output:
      if metric_table.name == self._tpu_info_table_name:
        for i, row_dict in enumerate(metric_table.body):
          # The 'Collective Type' column (e.g., ALL_GATHER / ALL_REDUCE) is
          # missing from the tpu-info output table. This prevents us from
          # explicitly filtering and distinguishing the two different
          # operations.
          #
          # As a temporary solution, we are fetching the values based on the
          # 'Buffer Size' label (e.g., '16MB+').
          # TODO: b/454457878 - This logic must be updated to filter on
          # 'Collective Type' as soon as the tpu-info column is added to the
          # table to ensure correctness.
          buffer_size = row_dict.get(self._tpu_info_group_by_key)
          if not buffer_size:
            continue

          buffer_size_name = buffer_size + "(" + str(i) + ")"
          parsed_values_by_buffer[buffer_size_name] = {}
          for p in self.percentiles_to_check:
            value_str = row_dict.get(p.name, "")

            match = re.search(r"([\d\.]+)", value_str)
            if match:
              parsed_values_by_buffer[buffer_size_name][p.name] = float(
                  match.group(1)
              )

    tpu_info_data_values = []
    for buffer_size_name in sorted(parsed_values_by_buffer.keys()):
      for p in self.percentiles_to_check:
        tpu_info_data_values.append(
            parsed_values_by_buffer[buffer_size_name][p.name]
        )

    return tpu_info_data_values


ALL_METRIC_STRATEGIES = [
    MemoryUsedStrategy(),
    MemoryTotalStrategy(),
    DutyCycleStrategy(),
    TensorcoreUtilizationStrategy(),
    # TODO(b/481177412): Re-enable and validate latency metrics.
    # Current Monitoring API aggregation differs from tpu-info, making it
    # unsuitable as a Source of Truth. Investigation for a valid verification
    # method is ongoing.
    # BufferTransferLatencyStrategy(),
    # HostToDeviceTransferLatenciesStrategy(),
    # DeviceToHostTransferLatenciesStrategy(),
    # CollectiveEndToEndLatencyLatenciesStrategy(),
]
