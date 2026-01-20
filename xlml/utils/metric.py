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

"""Utilities to process Benchmark metrics."""

import dataclasses
import datetime
import enum
import hashlib
import os
import re
from typing import Dict, Iterable, List, Optional
import uuid
from absl import logging
import airflow
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models import TaskInstance
from airflow.operators.python import get_current_context
from xlml.apis import gcp_config, test_config
from xlml.apis import metric_config
from xlml.utils import bigquery, composer
from dags import composer_env
from google.cloud import storage
import jsonlines
import numpy as np
import tensorflow as tf
from tensorflow.core.util import event_pb2
from urllib.parse import urlparse


@dataclasses.dataclass
class TensorBoardScalar:
  metric_value: float
  step: int


class TaskState(enum.Enum):
  FAILED = "failed"
  SKIPPED = "upstream_failed"
  SUCCESS = "success"


def is_valid_tag(
    tag: str,
    include_tag_patterns: Optional[Iterable[str]],
    exclude_tag_patterns: Optional[Iterable[str]],
) -> bool:
  """Check if it is a valid tag.

  Args:
    tag: The tag to check.
    include_tag_patterns: A list of patterns should be included.
    exclude_tag_patterns: A list of patterns should be excluded. This pattern
      has higher priority to include_tag_pattern, if any conflict.

  Returns:
    A bool to indicate if this tag should be included.
  """
  if exclude_tag_patterns and any(
      re.match(x, tag) for x in exclude_tag_patterns
  ):
    # check if tag in exclude_tag_patterns
    return False
  if include_tag_patterns:
    # check if tag in include_tag_patterns
    return any(re.match(x, tag) for x in include_tag_patterns)
  return True


def read_from_tb(
    file_location: str,
    include_tag_patterns: Optional[Iterable[str]],
    exclude_tag_patterns: Optional[Iterable[str]],
) -> (Dict[str, List[TensorBoardScalar]], Dict[str, str]):
  """Read metrics and dimensions from TensorBoard file.

  Args:
    file_location: The full path of a file in GCS.
    include_tag_patterns: The matching pattern of tags that wil be included.
    exclude_tag_patterns: The matching pattern of tags that will be excluded.
      This pattern has higher priority to include_tag_pattern, if any conflict.

  Returns:
    A dict that maps metric name to a list of TensorBoardScalar, and
    a dict that maps dimension name to dimenstion value.
  """
  metrics = {}
  metadata = {}

  serialized_examples = tf.data.TFRecordDataset(file_location)
  logging.info(f"TensorBoard metric_location is: {file_location}")
  for ex in serialized_examples:
    event = event_pb2.Event.FromString(ex.numpy())
    for value in event.summary.value:
      if not is_valid_tag(
          value.tag, include_tag_patterns, exclude_tag_patterns
      ):
        continue
      value_type = value.metadata.plugin_data.plugin_name
      if value_type == "scalars":
        if value.tag not in metrics:
          metrics[value.tag] = []
        t = tf.make_ndarray(value.tensor)
        metrics[value.tag].append(TensorBoardScalar(float(t), event.step))
      elif value_type == "text":
        metadata[value.tag] = bytes(value.tensor.string_val[0]).decode("utf-8")
      elif value.HasField("simple_value"):
        # simple_value indicates the value is a float:
        # https://github.com/tensorflow/tensorflow/blob/4dacf3f/tensorflow/core/framework/summary.proto#L122
        scalar = TensorBoardScalar(value.simple_value, event.step)
        metrics.setdefault(value.tag, []).append(scalar)
      else:
        logging.info(
            f"Discarding data point {value.tag} with type {value_type}."
        )

  return metrics, metadata


def aggregate_metrics(
    metrics: Iterable[TensorBoardScalar],
    strategy: metric_config.AggregationStrategy,
) -> float:
  """Get the aggregated value based on stragety.

  Args:
    metrics: The TensorBoardScalar from TensorBoard file.
    strategy: The strategy for aggregate values.

  Returns:
    A value after aggregation.
  """
  if strategy == metric_config.AggregationStrategy.LAST:
    last_value = max(metrics, key=lambda p: p.step)
    return last_value.metric_value
  elif strategy == metric_config.AggregationStrategy.AVERAGE:
    return np.mean([m.metric_value for m in metrics])
  elif strategy == metric_config.AggregationStrategy.MEDIAN:
    return np.median([m.metric_value for m in metrics])
  else:
    raise NotImplementedError(f"Unknown aggregation strategy: {strategy}")


def download_object_from_gcs(
    source_location: str, destination_location: str
) -> None:
  """Download object from GCS bucket.

  Args:
    source_location: The full path of a file in GCS.
    destination_location: The local path of the file.
  """

  storage_client = storage.Client()
  bucket_name = source_location.split("/")[2]
  object_name = "/".join(source_location.split("/")[3:])

  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(object_name)
  blob.download_to_filename(destination_location)
  logging.info(
      ("Download file from" f" {source_location} to {destination_location}")
  )


def process_json_lines(
    base_id: str,
    file_location: str,
) -> (
    List[List[bigquery.MetricHistoryRow]],
    List[List[bigquery.MetadataHistoryRow]],
):
  """Process metrics and dimensions from JSON Lines file.

  Args:
    base_id: The unique ID for this test job.
    file_location: The full path of a file in GCS.

  Returns:
    A list of MetricHistoryRow for all test runs, and
    a list of MetadataHistoryRow ofr all test runs in a test job.
  """

  tmp_location = "/tmp/ml-auto-solutions-metrics.jsonl"
  download_object_from_gcs(file_location, tmp_location)
  metric_list = []
  metadata_list = []

  with jsonlines.open(tmp_location) as reader:
    index = 0
    for object in reader:
      uuid = generate_row_uuid(base_id, index)
      index += 1
      raw_metrics = object["metrics"]
      metadata = object["dimensions"]
      metric_history_rows = []
      metadata_history_rows = []

      for key, value in raw_metrics.items():
        metric_history_rows.append(
            bigquery.MetricHistoryRow(
                job_uuid=uuid, metric_key=key, metric_value=value
            )
        )

      for key, value in metadata.items():
        metadata_history_rows.append(
            bigquery.MetadataHistoryRow(
                job_uuid=uuid, metadata_key=key, metadata_value=value
            )
        )

      metric_list.append(metric_history_rows)
      metadata_list.append(metadata_history_rows)

    return metric_list, metadata_list


def process_tensorboard_summary(
    base_id: str,
    summary_config: metric_config.SummaryConfig,
    use_generated_gcs_folder: bool,
    generated_gcs_folder: Optional[str],
) -> (
    List[List[bigquery.MetricHistoryRow]],
    List[List[bigquery.MetadataHistoryRow]],
):
  """Process metrics and dimensions from TensorBoard file.

  Args:
    base_id: The unique ID for this test job.
    summary_config: The configs for TensorBoard summary.
    use_generated_gcs_folder: The indicator to use default gcs folder.
    generated_gcs_folder: The GCS path of default folder.

  Returns:
    A list of MetricHistoryRow for a test run, and
    a list of MetadataHistoryRow ofr a test run in a test job.
  """
  uuid = generate_row_uuid(base_id, 0)

  if isinstance(summary_config.file_location, airflow.XComArg):
    file_location = summary_config.file_location.resolve(get_current_context())
  else:
    if use_generated_gcs_folder:
      file_location = os.path.join(
          generated_gcs_folder, summary_config.file_location
      )
    else:
      file_location = summary_config.file_location

  if summary_config.use_regex_file_location:
    file_location = get_gcs_file_location_with_regex(file_location)
    if file_location == "":
      return [[]], [[]]

  aggregation_strategy = summary_config.aggregation_strategy
  include_tag_patterns = summary_config.include_tag_patterns
  exclude_tag_patterns = summary_config.exclude_tag_patterns

  metrics, metadata = read_from_tb(
      file_location, include_tag_patterns, exclude_tag_patterns
  )
  aggregated_metrics = {}
  for key, value in metrics.items():
    aggregated_metrics[key] = aggregate_metrics(value, aggregation_strategy)
  print("aggregated_metrics", aggregated_metrics)

  metric_history_rows = []
  metadata_history_rows = []

  for key, value in aggregated_metrics.items():
    metric_history_rows.append(
        bigquery.MetricHistoryRow(
            job_uuid=uuid, metric_key=key, metric_value=value
        )
    )

  for key, value in metadata.items():
    metadata_history_rows.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid, metadata_key=key, metadata_value=value
        )
    )

  return [metric_history_rows], [metadata_history_rows]


def get_gcs_file_location_with_regex(
    file_location: str, return_largest: bool = True
) -> str:
  """
  Get a file from GCS given a regex in the form of
  `gs://<your_bucket>/<your_file_path_regex>`. Does not support
   bucket name or path regex. Only supports file name regex.

  Args:
    file_location: File location regex in the form of
        `gs://<your_bucket>/<path>/<your_file_name_regex>`.
    return_largest: If true, returns the largest matching file.

  Returns:
    The file location of the largest (unless return_largest is False)
    file that fits the given regex.
  """
  storage_client = storage.Client()

  url = urlparse(file_location)
  bucket_name = url.netloc
  file_path = url.path.strip("/")
  file_path_regex = re.compile(file_path)
  prefix = "/".join(file_path.split("/")[:-1])

  matched_blobs = [
      b
      for b in storage_client.list_blobs(bucket_name, prefix=prefix)
      if file_path_regex.match(b.name)
  ]

  if not matched_blobs:
    logging.warning(f"No objects matched supplied regex: {file_location}")
    return ""

  if return_largest:
    sizeable_blobs = [b for b in matched_blobs if b.size is not None]
    if not sizeable_blobs:
      logging.warning(
          f"No sizeable objects matched supplied regex: {file_location}"
      )
      return ""
    selected_blob = max(sizeable_blobs, key=lambda b: b.size)
  else:
    selected_blob = matched_blobs[0]

  return f"gs://{bucket_name}/{selected_blob.name}"


@task.virtualenv(
    task_id="process_profile_metrics",
    requirements=["tensorboard_plugin_profile==2.19.4"],
    system_site_packages=True,
)
def xplane_to_metrics(dir_location: str | airflow.XComArg) -> dict:
  """
  Find the first profile matching regex `{dir_location}/.*/*xplane.pb`.
    Download profile to temporary directory.
    Extract metrics from xplane file.

  Returns:
    A dictionary of profile metrics, or None if no match
  """
  # pylint: disable=redefined-outer-name, import-outside-toplevel, reimported
  import json
  import logging
  import re
  import tempfile
  from urllib.parse import urlparse
  import airflow
  from google.cloud import storage
  from tensorboard_plugin_profile.convert import raw_to_tool_data
  # pylint: enable=redefined-outer-name, import-outside-toplevel, reimported

  # --- Find and Download Profile ---
  # pylint: disable-next=redefined-outer-name
  def download_object_from_gcs(
      source_location: str, destination_location: str
  ) -> None:
    """Download object from GCS bucket.

    Args:
      source_location: The full path of a file in GCS.
      destination_location: The local path of the file.
    """
    storage_client = storage.Client()
    bucket_name = source_location.split("/")[2]
    object_name = "/".join(source_location.split("/")[3:])

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.download_to_filename(destination_location)
    logging.info(
        ("Download file from" f" {source_location} to {destination_location}")
    )

  def get_gcs_profile_location(dir_location: str) -> str:
    """
    Find first gcs location matching regex `{dir_location}/.*/*xplane.pb`.

    Returns:
      The matched gcs location, or "" if no match
    """
    storage_client = storage.Client()
    url = urlparse(dir_location)
    bucket_name = url.netloc
    file_path = url.path.lstrip("/")
    file_path_regex = re.compile(file_path + "/.*/*xplane.pb")
    for b in storage_client.list_blobs(bucket_name, prefix=file_path):
      if file_path_regex.match(b.name):
        return f"gs://{bucket_name}/{b.name}"
    logging.warning(
        f"No objects matched supplied regex: {dir_location}/.*/*xplane.pb"
    )
    return ""

  # --- Extract Metrics from Profile ---
  def round_number(number: float | str, decimal: int) -> float:
    if isinstance(number, str):
      number = float(number)
    return float(f"{number:.{decimal}f}")

  def get_tool_data(input_path: str, tool: str, params: dict) -> dict:
    data, content_type = raw_to_tool_data.xspace_to_tool_data(
        xspace_paths=[input_path],
        tool=tool,
        params=params,
    )
    if content_type != "application/json" or data is None:
      raise ValueError(f"{tool}: content is not a valid json string")
    parsed_data = json.loads(data)
    return parsed_data

  def get_tool_metrics(input_path: str) -> dict:
    out = {}
    # check available tools
    available_tools = set(
        raw_to_tool_data.xspace_to_tool_names(xspace_paths=[input_path])
    )
    # 1 overview_page
    # generate ALL_HOSTS.op_stats.pb
    if "overview_page" not in available_tools:
      logging.warning("overview_page: unavailable tool")
    else:
      try:
        overview_page = get_tool_data(
            input_path, tool="overview_page", params={}
        )
        out.update({
            "Device Type": overview_page[2]["p"]["device_type"],
            "Device Core Count": int(
                overview_page[2]["p"]["device_core_count"]
            ),
            "Average Tensor Core Step Time (ms)": float(
                overview_page[1]["p"]["steptime_ms_average"]
            ),
        })
      except ValueError as e:
        logging.warning(e)

    # 2 op_profile
    if "op_profile" not in available_tools:
      logging.warning("op_profile: unavailable tool")
      return out
    try:
      op_profile = get_tool_data(input_path, tool="op_profile", params={})
      out.update({
          "TPU FLOPS Utilization (%): exclude_idle": round_number(
              op_profile["byProgramExcludeIdle"]["metrics"]["flops"] * 100,
              2,
          ),
          "HBM Bandwidth Utilization (%): exclude_idle": round_number(
              op_profile["byProgramExcludeIdle"]["metrics"]["bandwidthUtils"][0]
              * 100,
              2,
          ),
      })
    except ValueError as e:
      logging.warning(e)
      return out

    # 3 memory_viewer: jit_train_step
    # generate jit*.hlo_proto.pb
    if "memory_viewer" not in available_tools:
      logging.warning("memory_viewer: unavailable tool")
      return out
    # find jit_train_step from 2
    # example: "jit_train_step(4869159985936022652)"
    jit_train_step = None
    for program in op_profile["byProgramExcludeIdle"]["children"]:
      if program["name"].startswith("jit_train_step"):
        jit_train_step = program["name"]
        break
    if jit_train_step is None:
      logging.warning("memory_viewer: cannot find jit_train_step in op_profile")
      return out
    # 3.1 hbm
    MEMORY_SPACE_HBM = "0"
    try:
      memory_viewer_hbm = get_tool_data(
          input_path,
          "memory_viewer",
          params={"host": jit_train_step, "memory_space": MEMORY_SPACE_HBM},
      )
      out["Peak memory allocation (MiB): jit_train_step, HBM"] = round_number(
          memory_viewer_hbm["totalBufferAllocationMib"], 2
      )
    except ValueError as e:
      logging.warning(e)
    # 3.2 host
    MEMORY_SPACE_HOST = "5"
    try:
      memory_viewer_host = get_tool_data(
          input_path,
          "memory_viewer",
          params={"host": jit_train_step, "memory_space": MEMORY_SPACE_HOST},
      )
      out["Peak memory allocation (MiB): jit_train_step, host"] = round_number(
          memory_viewer_host["totalBufferAllocationMib"], 2
      )
    except ValueError as e:
      logging.warning(e)
    return out

  # --- Main Logic ---
  # Find the first file_location matching regex `{dir_location}/.*/*xplane.pb`
  if isinstance(dir_location, airflow.XComArg):
    dir_location = dir_location.resolve(get_current_context())
  file_location = get_gcs_profile_location(dir_location)
  if file_location:
    print(f"For local download, run: gcloud storage cp {file_location} .")
    # Temporary directory for file download and extraction cache
    with tempfile.TemporaryDirectory() as tmp_dir:
      input_path = f"{tmp_dir}/gke.xplane.pb"
      download_object_from_gcs(file_location, input_path)
      # Extract profile
      return get_tool_metrics(input_path)
  # No match profile
  return None


def process_profile(
    base_id: str,
    raw_metric: dict,
) -> List[List[bigquery.MetricHistoryRow]]:
  row_uuid = generate_row_uuid(base_id, 0)
  profile_history_rows = []
  for key, value in raw_metric.items():
    profile_history_rows.append(
        bigquery.MetricHistoryRow(
            job_uuid=row_uuid, metric_key="profile/" + key, metric_value=value
        )
    )
  return [profile_history_rows]


def encode_url(url: str) -> str:
  """Replace characters with % followed by two hexadecimal digits.

  Args:
    url: The url to be encoded.

  Returns:
    An encoded url.
  """
  return str(url).replace(":", "%3A").replace("+", "%2B")


def add_airflow_metadata(
    base_id: str,
    project_name: str,
    metadata: List[List[bigquery.MetricHistoryRow]],
) -> List[List[bigquery.MetricHistoryRow]]:
  """Add airflow metadata: run_id, prev_start_date_success,
  and airflow_dag_run_link.

  Args:
    base_id: The base id to generate uuid.
    metadata: The data to append airflow metadata.
    configs: The GCP configs to get composer metadata.

  Returns:
    The data with airflow metadata.
  """
  context = get_current_context()
  run_id = context["run_id"]
  prev_start_date_success = str(context["prev_start_date_success"])
  dag_run = context["dag_run"]
  dag_id = dag_run.dag_id
  task_id = context["task"].task_id
  dag_run_id = encode_url(run_id)
  airflow_link = composer.get_airflow_url(
      project_name,
      os.environ.get(composer_env.COMPOSER_LOCATION),
      os.environ.get(composer_env.COMPOSER_ENVIRONMENT),
  )
  airflow_dag_run_link = (
      f"{airflow_link}/dags/{dag_id}/"
      f"grid?dag_run_id={dag_run_id}&task_id={task_id}"
  )
  logging.info(f"airflow_dag_run_link is {airflow_dag_run_link}")

  # append airflow metadata for each test run.
  for index in range(len(metadata)):
    uuid = generate_row_uuid(base_id, index)
    airflow_meta = []

    airflow_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid, metadata_key="run_id", metadata_value=run_id
        )
    )
    if context["prev_start_date_success"]:
      airflow_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="prev_start_date_success",
              metadata_value=prev_start_date_success,
          )
      )
    airflow_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="airflow_dag_run_link",
            metadata_value=airflow_dag_run_link,
        )
    )
    airflow_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid, metadata_key="dag_id", metadata_value=dag_id
        )
    )
    for key, value in context.get("params", {}).items():
      airflow_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key=f"param:{key}",
              metadata_value=str(value),
          )
      )

    metadata[index].extend(airflow_meta)
  return metadata


def add_test_config_metadata(
    base_id: str,
    task_test_config: test_config.TestConfig[test_config.Accelerator],
    task_gcp_config: gcp_config.GCPConfig,
    task_metric_config: metric_config.MetricConfig,
    metadata: List[List[bigquery.MetricHistoryRow]],
) -> List[List[bigquery.MetricHistoryRow]]:
  for index in range(len(metadata)):
    uuid = generate_row_uuid(base_id, index)
    test_config_meta = []

    test_config_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="accelerator",
            metadata_value=task_test_config.accelerator.name,
        )
    )
    test_config_meta.append(
        bigquery.MetadataHistoryRow(
            job_uuid=uuid,
            metadata_key="project",
            metadata_value=task_gcp_config.project_name,
        )
    )
    if hasattr(task_test_config, "num_slices"):
      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="num_slices",
              metadata_value=task_test_config.num_slices,
          )
      )
      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="multislice_topology",
              metadata_value=(
                  f"{task_test_config.num_slices}"
                  f"x{task_test_config.accelerator.name}"
              ),
          )
      )
    if (
        task_metric_config is not None
        and task_metric_config.tensorboard_summary
    ):
      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="metric_aggregation_strategy",
              metadata_value=task_metric_config.tensorboard_summary.aggregation_strategy.name,
          )
      )
    metadata[index].extend(test_config_meta)

  return metadata


def generate_row_uuid(base_id: str, index: int) -> str:
  """Generate uuid for entry.

  Args:
    base_id: The process id generated once per post process task group.
    index: The index of test runs.

  Returns:
    A uuid for table entry.
  """
  return hashlib.sha256(str(base_id + str(index)).encode("utf-8")).hexdigest()


@task(trigger_rule="all_done")
def generate_process_id() -> str:
  """Generate a process id that will be a base id for uuid of test runs.

  Returns:
    A random uuid.
  """
  return str(uuid.uuid4())


def update_dataset_name_if_needed(
    prod_dataset_name: metric_config.DatasetOption,
) -> str:
  """Update the dataset name based on stage (if needed).

  All data from prod env will be sent to benchmark_dataset or xlml_dataset;
  the rest will be sent to dev_benchmark_dataset or dev_xlml_dataset.
  """

  if not composer_env.is_prod_env():
    logging.info("This is a non-prod run, and send all data to dev dataset.")
    return f"dev_{prod_dataset_name.value}"
  return prod_dataset_name.value


def find_full_task_id_from_upstream(
    start_task: airflow.models.baseoperator.BaseOperator,
    target_task_name: str,
) -> str:
  """Finds a task in the upstream hierarchy that contains the given substring.

  For example if the current task has taskd_id as below
  `chained_tests_llama2-70b_nightly.maxtext-nightly-llama2-70b-m1-megamem-96-1.post_process.process_metrics`
  we want to find a task_id with the substring `wait_for_workload_completion`,
  then we expect to find the task_id from the upstream like below.
  `chained_tests_llama2-70b_nightly.maxtext-nightly-llama2-70b-m1-megamem-96-1.run_model.wait_for_workload_completion`
  """
  dag = start_task.dag
  queue = [start_task]
  visited_task_ids = {start_task.task_id}
  while queue:
    current = queue.pop(0)
    if target_task_name in current.task_id:
      logging.info("found task_id from upstream: %s", current.task_id)
      return current.task_id
    for upstream_task_id in current.upstream_task_ids:
      upstream_task = dag.get_task(upstream_task_id)
      if upstream_task and upstream_task.task_id not in visited_task_ids:
        visited_task_ids.add(upstream_task.task_id)
        queue.append(upstream_task)
  raise AirflowFailException(
      f"Could not find task with substring '{target_task_name}' in upstream"
      " tasks."
  )


def get_xpk_job_status() -> bigquery.JobStatus:
  """Get job status for the GKE run.

  FAILED - if any failure occurs in run_model
  SUCCESS - end-to-end model tests are successful in run_model
  """
  context = get_current_context()
  execution_date = context["dag_run"].logical_date
  current_dag = context["dag"]

  workload_completion_task_id = find_full_task_id_from_upstream(
      context["task"],
      "wait_for_workload_completion",
  )

  workload_completion = current_dag.get_task(
      task_id=workload_completion_task_id
  )
  workload_completion_ti = TaskInstance(workload_completion, execution_date)
  workload_completion_state = workload_completion_ti.current_state()

  if workload_completion_state == TaskState.SUCCESS.value:
    logging.info(
        "The wait_for_workload_completion state is success, and the job status"
        " is success."
    )
    return bigquery.JobStatus.SUCCESS

  logging.info(
      "The wait_for_workload_completion state is not success, and the job"
      " status is failed."
  )
  return bigquery.JobStatus.FAILED


def get_gke_job_status(
    task_test_config: test_config.TestConfig[test_config.Accelerator],
) -> bigquery.JobStatus:
  """Get job status for the GCE run.

  FAILED - if any failure occurs in setup & run_model (including timeout of
  run_model).
  SUCCESS - end-to-end model tests are successful from provision to run_model
  """
  context = get_current_context()
  execution_date = context["dag_run"].logical_date
  current_dag = context["dag"]
  benchmark_id = task_test_config.benchmark_id

  # check setup status to see if setup step is successful
  setup_task = current_dag.get_task(
      task_id=f"{benchmark_id}.generate_gcs_folder_location"
  )
  setup_ti = TaskInstance(setup_task, execution_date)
  setup_state = setup_ti.current_state()

  if setup_state == TaskState.FAILED.value:
    logging.info("The setup state is failed, and the job status is failed.")
    return bigquery.JobStatus.FAILED

  # check run_model status to see if run_model step is successful
  run_model_task = current_dag.get_task(
      task_id=f"{benchmark_id}.run_model.stream_logs"
  )
  run_model_ti = TaskInstance(run_model_task, execution_date)
  run_model_state = run_model_ti.current_state()

  if run_model_state == TaskState.SUCCESS.value:
    logging.info(
        "The run_model state is success, and the job status is success."
    )
    return bigquery.JobStatus.SUCCESS

  logging.info("The run_model state is failed, and the job status is failed.")
  return bigquery.JobStatus.FAILED


def get_gce_job_status(
    task_test_config: test_config.TestConfig[test_config.Accelerator],
    use_startup_script: bool,
) -> bigquery.JobStatus:
  """Get job status for the GCE run.

  MISSED - if any failure occurs in initialize & create_queued_resource
  FAILED - if any failure occurs in setup & run_model (including timeout of
  run_model) for SSH method.
  FAILED - if any failure occurs in check_if_startup_script_end
  (including timeout of check_if_startup_script_end) for startup script method.
  SUCCESS - end-to-end model tests are successful from provision to run_model
  """
  context = get_current_context()
  execution_date = context["dag_run"].logical_date
  current_dag = context["dag"]
  benchmark_id = task_test_config.benchmark_id

  # GCE SSH method
  if not use_startup_script:
    if isinstance(task_test_config.accelerator, test_config.Tpu):
      # check wait status to see if wait_for_ready_queued_resource is successful
      wait_task = current_dag.get_task(
          task_id=f"{benchmark_id}.provision.create_queued_resource.wait_for_ready_queued_resource"
      )
    elif isinstance(task_test_config, test_config.GpuVmTest):
      if task_test_config.use_existing_instance:
        wait_task = current_dag.get_task(
            task_id=f"{benchmark_id}.provision.get_existing_resource"
        )
      else:
        wait_task = current_dag.get_task(
            task_id=f"{benchmark_id}.provision.create_resource.get_ip_address"
        )
    else:
      raise NotImplementedError(
          f"Unable to get task for {type(task_test_config.accelerator)}."
      )
    wait_ti = TaskInstance(wait_task, execution_date)
    wait_state = wait_ti.current_state()

    if wait_state == TaskState.SKIPPED.value:
      logging.info(
          "The wait_for_ready_queued_resource state is skipped, and the job status is missed."
      )
      return bigquery.JobStatus.MISSED

    # check setup status to see if setup step is successful
    if (
        hasattr(task_test_config, "use_existing_instance")
        and task_test_config.use_existing_instance
    ):
      get_instance_task = current_dag.get_task(
          task_id=f"{benchmark_id}.provision.get_existing_resource"
      )
      get_instance_ti = TaskInstance(get_instance_task, execution_date)
      get_instance_state = get_instance_ti.current_state()
      if get_instance_state == TaskState.FAILED.value:
        logging.info(
            "The getting existing instance state is failed, and the job status is failed."
        )
        return bigquery.JobStatus.FAILED
    else:
      setup_task = current_dag.get_task(
          task_id=f"{benchmark_id}.provision.setup"
      )
      setup_ti = TaskInstance(setup_task, execution_date)
      setup_state = setup_ti.current_state()
      if setup_state == TaskState.FAILED.value:
        logging.info("The setup state is failed, and the job status is failed.")
        return bigquery.JobStatus.FAILED

    # check run_model status to see if run_model step is successful
    run_model_task = current_dag.get_task(task_id=f"{benchmark_id}.run_model")
    run_model_ti = TaskInstance(run_model_task, execution_date)
    run_model_state = run_model_ti.current_state()

    if run_model_state == TaskState.SUCCESS.value:
      logging.info(
          "The run_model state is success, and the job status is success."
      )
      return bigquery.JobStatus.SUCCESS

    logging.info("The run_model state is failed, and the job status is failed.")
    return bigquery.JobStatus.FAILED
  # GCE startup script method
  else:
    # check wait status to see if provision step is successful
    wait_task = current_dag.get_task(
        task_id=f"{benchmark_id}.provision_with_startup_script.create_queued_resource.wait_for_ready_queued_resource"
    )
    wait_ti = TaskInstance(wait_task, execution_date)
    wait_state = wait_ti.current_state()

    if wait_state == TaskState.SKIPPED.value:
      logging.info(
          "The wait_for_ready_queued_resource state is skipped, and the job status is missed."
      )
      return bigquery.JobStatus.MISSED

    # check startup_script status to see if startup_script step is successful
    startup_script_task = current_dag.get_task(
        task_id=f"{benchmark_id}.provision_with_startup_script.create_queued_resource.check_if_startup_script_end"
    )
    startup_script_ti = TaskInstance(startup_script_task, execution_date)
    startup_script_state = startup_script_ti.current_state()
    if startup_script_state == TaskState.FAILED.value:
      logging.info(
          "The startup_script state is failed, and the job status is failed."
      )
      return bigquery.JobStatus.FAILED
    else:
      logging.info(
          "The startup_script state is success, and the job status is success."
      )
      return bigquery.JobStatus.SUCCESS


# TODO(ranran): handle Airflow retry to avoid duplicate records in tables
@task
def process_metrics(
    base_id: str,
    task_test_config: test_config.TestConfig[test_config.Accelerator],
    task_metric_config: Optional[metric_config.MetricConfig],
    task_gcp_config: gcp_config.GCPConfig,
    use_startup_script: bool = False,
    folder_location: Optional[str] = None,
) -> None:
  benchmark_id = task_test_config.benchmark_id
  current_time = datetime.datetime.now()
  has_profile = False
  metric_history_rows_list = [[]]
  metadata_history_rows_list = [[]]
  profile_history_rows_list = []

  # process metrics, metadata, and profile
  if task_metric_config:
    if task_metric_config.json_lines:
      absolute_path = (
          os.path.join(
              folder_location, task_metric_config.json_lines.file_location
          )
          if task_metric_config.use_runtime_generated_gcs_folder
          else task_metric_config.json_lines.file_location
      )
      metric_history_rows_list, metadata_history_rows_list = process_json_lines(
          base_id, absolute_path
      )
    if task_metric_config.tensorboard_summary:
      (
          metric_history_rows_list,
          metadata_history_rows_list,
      ) = process_tensorboard_summary(
          base_id,
          task_metric_config.tensorboard_summary,
          task_metric_config.use_runtime_generated_gcs_folder,
          folder_location,
      )

    if task_metric_config.profile:
      profile_metrics = task_metric_config.profile.metrics
      if isinstance(profile_metrics, airflow.XComArg):
        profile_metrics = profile_metrics.resolve(get_current_context())
      if profile_metrics:
        profile_history_rows_list = process_profile(base_id, profile_metrics)
        has_profile = True

  # add default airflow metadata
  metadata_history_rows_list = add_airflow_metadata(
      base_id, task_gcp_config.composer_project, metadata_history_rows_list
  )

  metadata_history_rows_list = add_test_config_metadata(
      base_id,
      task_test_config,
      task_gcp_config,
      task_metric_config,
      metadata_history_rows_list,
  )

  # append profile metrics to metric_history_rows_list if any
  if has_profile:
    if len(metric_history_rows_list) != len(profile_history_rows_list):
      logging.error(
          f"The num of profile is {len(profile_history_rows_list)}, but it is"
          " different to the number of test runs"
          f" {len(metric_history_rows_list)}. Ignoring profiles."
      )
    else:
      for index in range(len(metric_history_rows_list)):
        metric_history_rows_list[index].extend(profile_history_rows_list[index])

  test_run_rows = []

  dataset_name = update_dataset_name_if_needed(task_gcp_config.dataset_name)
  bigquery_metric = bigquery.BigQueryMetricClient(
      task_gcp_config.dataset_project, dataset_name
  )

  if hasattr(task_test_config, "cluster_name"):
    test_job_status = get_xpk_job_status()
  elif isinstance(task_test_config, test_config.GpuGkeTest):
    test_job_status = get_gke_job_status(task_test_config)
  else:
    test_job_status = get_gce_job_status(task_test_config, use_startup_script)

  for index in range(len(metadata_history_rows_list)):
    job_history_row = bigquery.JobHistoryRow(
        uuid=generate_row_uuid(base_id, index),
        timestamp=current_time,
        owner=task_test_config.task_owner,
        job_name=benchmark_id,
        job_status=test_job_status.value,
    )
    test_run_row = bigquery.TestRun(
        job_history_row,
        metric_history_rows_list[index],
        metadata_history_rows_list[index],
    )
    test_run_rows.append(test_run_row)

  print("Test run rows:", test_run_rows)
  bigquery_metric.insert(test_run_rows)
