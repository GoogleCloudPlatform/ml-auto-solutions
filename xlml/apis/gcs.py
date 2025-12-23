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

"""Functions for GCS Bucket"""

import os
import re
import tempfile
from typing import List

from absl import logging
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.providers.google.cloud.operators.gcs import GCSHook
import yaml


def obtain_file_list(gcs_path: str) -> List[str]:
  """
  Lists files in a GCS bucket at a specified path.

  This function uses the GCSHook to connect to Google Cloud Storage.
  It parses the provided `output_path` to extract the bucket name and prefix,
  and then lists all objects within that path.

  Args:
    output_path (str): The full gs:// path to the GCS bucket and prefix
      (e.g., "gs://my-bucket/my-folder/").

  Returns:
    List[str]: A list of file names (keys) found in the specified GCS path.
  """
  hook = GCSHook()
  pattern = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<prefix>.+)$")
  m = pattern.match(gcs_path)

  if not m:
    logging.error(f"Invalid GCS path format: {gcs_path}")
    return []

  bucket_name = m.group("bucket")
  prefix = m.group("prefix")

  logging.info(f"output_path:{gcs_path}")
  logging.info(f"bucket:{bucket_name}")
  logging.info(f"prefix:{prefix}")

  files = hook.list(bucket_name=bucket_name, prefix=prefix)
  logging.info(f"Files ===> {files}")
  return files


@task.sensor(poke_interval=3, timeout=300, mode="reschedule")
def wait_for_file_to_exist(file_path: str) -> bool:
  """
  Check the target file is existing in the gsc

  Args:
    file_path (str): The full path of the target
      file (e.g gs://mybucket/commit_message.txt)

  Returns:
    bool: return True if target file is found
  """
  directory_path = os.path.dirname(file_path)
  target_file = os.path.basename(file_path)
  logging.info(f"Directory to check: {directory_path}")
  logging.info(f"Target file name: {target_file}")

  if not directory_path.startswith("gs://"):
    raise ValueError(
        f"Invalid GCS path provided: {file_path}. "
        "Path must start with 'gs://'."
    )
  checkpoint_files = obtain_file_list(directory_path)
  for file in checkpoint_files:
    if target_file in file:
      logging.info(f"Found target file in the GCS path: {file}")
      return True

  logging.info("Target file not found in the specified GCS path.")
  return False


def load_yaml_from_gcs(gcs_path: str) -> dict:
  """Loads and parses the DAG configuration YAML file from GCS."""
  logging.info(f"Attempting to load config from: {gcs_path}")

  if not gcs_path.startswith("gs://"):
    raise ValueError(
        f"Invalid GCS path: '{gcs_path}'. Path must start with 'gs://'."
    )

  if not (
      gcs_path.lower().endswith(".yaml") or gcs_path.lower().endswith(".yml")
  ):
    logging.warning(
        f"GCS path '{gcs_path}' does not have a typical YAML extension (.yaml or .yml). "
        "Proceeding, but be aware this might not be a YAML file."
    )

  with tempfile.TemporaryDirectory() as tmpdir:
    temp_file_path = os.path.join(tmpdir, "downloaded_config")

    hook = SubprocessHook()
    command = ["gsutil", "-m", "cp", gcs_path, temp_file_path]
    logging.info(f"Running command: {' '.join(command)}")
    result = hook.run_command(command)
    assert (
        result.exit_code == 0
    ), f"gsutil command failed with exit code {result.exit_code}"

    if not os.path.exists(temp_file_path):
      logging.error(
          f"gsutil cp command completed, but '{temp_file_path}' was not created. "
          "This often means the copy failed. Check gsutil stdout/stderr in logs."
      )
      raise FileNotFoundError(
          f"[Errno 2] Failed to download file from GCS path: {gcs_path}"
      )

    with open(temp_file_path, "r", encoding="utf-8") as f:
      dag_yaml = f.read()

  return yaml.safe_load(dag_yaml)
