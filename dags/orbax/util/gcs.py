from airflow.providers.google.cloud.operators.gcs import GCSHook
import re
from absl import logging
from typing import List


def get_gcs_checkpoint(output_path: str) -> List[str]:
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
  m = pattern.match(output_path)

  if not m:
    logging.error(f"Invalid GCS path format: {output_path}")
    return []

  bucket_name = m.group("bucket")
  prefix = m.group("prefix")

  logging.info(f"output_path:{output_path}")
  logging.info(f"bucket:{bucket_name}")
  logging.info(f"prefix:{prefix}")

  files = hook.list(bucket_name=bucket_name, prefix=prefix)
  logging.info(f"Files ===> {files}")
  return files
