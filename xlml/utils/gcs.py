from airflow.providers.google.cloud.operators.gcs import GCSHook
import re
from absl import logging


def get_gcs_checkpoint(output_path):
  hook = GCSHook()
  pattern = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<prefix>.+)$")
  m = pattern.match(output_path)
  bucket_name = m.group("bucket")
  prefix = m.group("prefix")
  logging.info(f"output_path:{output_path}")
  logging.info(f"bucket:{bucket_name}")
  logging.info(f"prefix:{prefix}")
  files = hook.list(bucket_name=bucket_name, prefix=prefix)
  logging.info(f"Files ===> {files}")
  return files
