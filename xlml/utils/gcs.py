from airflow.providers.google.cloud.operators.gcs import GCSHook
import re
from absl import logging


def validate_gcs_checkpoint_p2(output_path):
  hook = GCSHook()
  pattern = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<prefix>.+)$")
  m = pattern.match(output_path)
  bucket_name = m.group("bucket")
  prefix = m.group("prefix")
  logging.info(f"output_path:{output_path}")
  logging.info(f"bucket:{bucket_name}")
  logging.info(f"prefix:{prefix}")
  files = hook.list(bucket_name=bucket_name, prefix=prefix)
  logging.info("Files ===> ", files)
  if len(files) > 0:
    for file in files:
      if ".data" in file:
        return True
  return False
