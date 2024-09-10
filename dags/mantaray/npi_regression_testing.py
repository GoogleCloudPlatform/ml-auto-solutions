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

"""DAG to run NPI nightly regression testing."""

from airflow import models
from xlml.utils import mantaray
import datetime

benchmark_script_path = "jobs/maxstar_v6e_benchmarks.py"
# Create a DAG that runs the benchmark script
with models.DAG(
    dag_id="npi_regression_testing",
    schedule="0 2 * * *", #everyday at 2am (UTC)
    tags=["mantaray"],
    start_date=datetime.datetime(2024, 9, 4),
    catchup=False,

) as dag:
  # TODO: build docker image
  run_workload = mantaray.run_workload_temp(
      workload_file_name=benchmark_script_path,
  )
  run_workload
