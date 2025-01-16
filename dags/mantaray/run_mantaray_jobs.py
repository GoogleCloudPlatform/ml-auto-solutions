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
from dags import composer_env
import re

# Skip running this script in unit test because gcs loading will fail.
if composer_env.is_prod_env() or composer_env.is_dev_env():
  # Download xlml_jobs.yaml from the borgcron GCS bucket, which
  # is pulled nightly from google3.
  xlml_jobs_yaml = mantaray.load_file_from_gcs(
      f"{mantaray.MANTARAY_G3_GS_BUCKET}/xlml_jobs/xlml_jobs.yaml"
  )
  xlml_jobs = yaml.safe_load(xlml_jobs_yaml)

  # Create a DAG for PyTorch/XLA tests
  pattern = r"^(ptxla|pytorchxla).*"
  workload_file_name_list = []
  for job in xlml_jobs:
    if re.match(pattern, job["task_name"]):
      workload_file_name_list.append(job["file_name"])

  # merge all PyTorch/XLA tests ino one Dag
  with models.DAG(
      dag_id="pytorch_xla_model_regression_test_on_trillium",
      schedule="0 0 * * *",  # everyday at midnight # job["schedule"],
      tags=["mantaray", "pytorchxla", "xlml"],
      start_date=datetime.datetime(2024, 4, 22),
      catchup=False,
  ) as dag:
    for workload_file_name in workload_file_name_list:
      run_workload = mantaray.run_workload.override(
          task_id=workload_file_name[:-3]
      )(
          workload_file_name=workload_file_name,
      )
      run_workload

  # Create a DAG for each job from maxtext
  for job in xlml_jobs:
    if not re.match(pattern, job["task_name"]):
      with models.DAG(
          dag_id=job["task_name"],
          schedule=job["schedule"],
          tags=["mantaray"],
          start_date=datetime.datetime(2024, 4, 22),
          catchup=False,
      ) as dag:
        run_workload = mantaray.run_workload(
            workload_file_name=job["file_name"],
        )
    run_workload
else:
  print(
      "Skipping creating Mantaray DAGs since not running in Prod or Dev composer environment."
  )
