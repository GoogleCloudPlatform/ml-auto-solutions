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

"""DAGs to run Mantaray benchmarks."""


import datetime
from airflow import models
from xlml.utils import mantaray
import yaml

# Download xlml_jobs.yaml from the ml-auto-solutions GCS bucket. Any update
# to this file in the bucket will automatically trigger the execution of
# this script, which recreates the Mantaray DAGs to reflect the changes.
xlml_jobs_yaml = mantaray.load_file_from_gcs(
    "gs://us-central1-ml-automation-s-bc954647-bucket/mantaray/xlml_jobs/xlml_jobs.yaml"
)
xlml_jobs = yaml.safe_load(xlml_jobs_yaml)
# Create a DAG for each job
for job in xlml_jobs:
  with models.DAG(
      dag_id=job["task_name"],
      schedule=job["schedule"],
      tags=["ray"],
      start_date=datetime.datetime(2024, 4, 22),
      catchup=False,
  ) as dag:
    run_workload = mantaray.run_workload(
        workload_file_name=job["file_name"],
    )

  run_workload
