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

"""Tests for benchmark bigquery.py."""

import datetime
import math
from unittest import mock
from absl.testing import absltest
from absl.testing import parameterized
from google.cloud import bigquery
from implementations.utils.benchmark import bigquery as benchmark_bigquery


class BenchmarkBigQueryMetricTest(parameterized.TestCase, absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.job_history_row = benchmark_bigquery.JobHistoryRow(
        uuid="job1",
        timestamp=datetime.datetime.now(),
        owner="owner1",
        job_name="test_job1",
        job_status="success",
    )
    self.metric_history_row = benchmark_bigquery.MetricHistoryRow(
        job_uuid="job1", metric_key="metric1", metric_value=0
    )
    self.metadata_history_row = benchmark_bigquery.MetadataHistoryRow(
        job_uuid="job1", metadata_key="metadata1", metadata_value="value1"
    )
    self.test_runs = [
        benchmark_bigquery.BenchmarkTestRun(
            self.job_history_row,
            [self.metric_history_row],
            [self.metadata_history_row],
        )
    ]

  @parameterized.named_parameters(
      ("-math.inf", -math.inf, False),
      ("null", float("nan"), False),
      ("math.nan", math.nan, False),
      ("5.0", 5.0, True),
  )
  def test_is_valid_metric(self, x: float, expected_value: bool):
    bq_metric = benchmark_bigquery.BigQueryMetricClient()
    actual_value = bq_metric.is_valid_metric(x)
    self.assertEqual(actual_value, expected_value)

  @mock.patch.object(bigquery.Client, "get_table", return_value="mock_table")
  @mock.patch.object(
      bigquery.Client, "insert_rows", return_value=["there is an error"]
  )
  def test_insert_failure(self, get_table, insert_rows):
    bq_metric = benchmark_bigquery.BigQueryMetricClient()
    self.assertRaises(RuntimeError, bq_metric.insert, self.test_runs)

  @mock.patch.object(bigquery.Client, "get_table", return_value="mock_table")
  @mock.patch.object(bigquery.Client, "insert_rows", return_value=[])
  def test_insert_success(self, get_table, insert_rows):
    bq_metric = benchmark_bigquery.BigQueryMetricClient()
    bq_metric.insert(self.test_runs)


if __name__ == "__main__":
  absltest.main()
