# Copyright 2024 Google LLC
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

"""Utility to generate names and locations."""

import datetime
import os
from airflow.decorators import task
from dags import gcs_bucket


@task
def generate_run_name(benchmark_id: str) -> str:
  """Generates a unique run name by appending the current
  datetime to benchmark_id.

  Args:
    benchmark_id: Benchmark id of the test
  """
  current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  return f"{benchmark_id}-{current_datetime}"


@task
def generate_tb_file_location(
    run_name: str, base_output_directory: str, nested_run_name_dir: bool = True
) -> str:
  """Generates a path to the tensorboard file to be used as a regex. Assumes
  the file is located in:
  <base_output_directory>/<run_name>/tensorboard/events.out.tfevents.*
  And if nested_run_name_dir then:
  <base_output_directory>/<run_name>/tensorboard/<run_name>/events.out.tfevents.*

  Args:
    run_name: run name for the tensorboard file location
    base_output_directory: GCS bucket path
    nested_run_name_dir: if True, use <run_name> twice in the path
  """
  if nested_run_name_dir:
    return os.path.join(
        base_output_directory,
        run_name,
        "tensorboard",
        run_name,
        "events.out.tfevents.*",
    )
  else:
    return os.path.join(
        base_output_directory,
        run_name,
        "tensorboard",
        "events.out.tfevents.*",
    )


@task
def generate_profile_file_location(
    run_name: str, base_output_directory: str
) -> str:
  """
  Generates a path to the directory containing profile:
  <base_output_directory>/<run_name>/tensorboard/plugins/profile
  To be used for subsequent regex file search. Assumes profile is located in:
  <base_output_directory>/<run_name>/tensorboard/plugins/profile/.*/*xplane.pb

  Args:
    run_name: run name for the profile file location
    base_output_directory: GCS bucket path
  """
  return os.path.join(
      base_output_directory,
      run_name,
      "tensorboard",
      "plugins",
      "profile",
  )


@task
def generate_gcs_folder_location(subfolder: str, benchmark_id: str) -> str:
  """Generates folder location in GCS.

  Args:
    subfolder: Folder name/path for artifacts, such as 'solutions_team/flax'
    benchmark_id: Benchmark id of the test

  Returns: GCS folder name with location
  """
  current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
  return os.path.join(
      gcs_bucket.BASE_OUTPUT_DIR,
      subfolder,
      f"{benchmark_id}-{current_datetime}/",
  )
