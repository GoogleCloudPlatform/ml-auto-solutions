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
from typing import List
from absl import logging

from airflow.decorators import task
from airflow.providers.google.cloud.operators.gcs import GCSHook


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
