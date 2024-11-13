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

"""Utilities to construct configs for maxtext offline benchmarks DAG."""

import datetime
import json
import jsonlines
import re

from airflow import models
from dags import test_owner
from dags.vm_resource import TpuVersion, Zone, Project, RuntimeVersion, V6E_GCE_NETWORK, V6E_GCE_SUBNETWORK
from dags.multipod.configs import common
from dags.multipod.configs.common import SetupMode
from xlml.apis import gcp_config, metric_config, task, test_config

PROJECT_NAME = Project.CLOUD_TPU_INFERENCE_TEST.value
RUNTIME_IMAGE = RuntimeVersion.V2_ALPHA_TPUV6.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value

create_mlperf_log_converter_script = r"""cat > convert_logs.py << 'EOL'
import json
import re
import jsonlines

def convert_mlperf_log_to_jsonlines(log_file_path: str, output_path: str):
  dimension_keys = {
    "loadgen_version", "test_datetime", "requested_scenario",
    "requested_test_mode", "effective_scenario", "effective_test_mode",
    "power_begin", "power_end", "result_validity",
    "early_stopping_ttft_result", "early_stopping_tpot_result"
  }

  metrics = {}
  dimensions = {}

  with open(log_file_path, "r") as f:
    log_content = f.read()

  log_pattern = r":::MLLOG ({.*})"
  for line in log_content.split("\n"):
    match = re.search(log_pattern, line)
    if match:
      try:
        entry = json.loads(match.group(1))
        key = entry.get("key", "")
        value = entry.get("value")

        if isinstance(value, (int, float)):
          metrics[key] = value
        elif key in dimension_keys:
          dimensions[key] = value
      except json.JSONDecodeError:
        continue

  result = {"metrics": metrics, "dimensions": dimensions}

  with jsonlines.open(output_path, mode="w") as writer:
    writer.write(result)

  return result

if __name__ == "__main__":
    convert_mlperf_log_to_jsonlines("./mlperf_log_detail.txt", "metric_report.jsonl")
EOL"""


def maxtext_inference_offline_benchmark_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_name: str,
    test_mode: common.SetupMode,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = True,
    num_slices: int = 1,
    maxtext_branch: str = "",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )
  git_clone_maxtext = "git clone https://github.com/google/maxtext.git"
  if maxtext_branch:
    git_clone_maxtext += f" -b {maxtext_branch}"

  set_up_cmds = (
      "pip install --upgrade pip",
      "sudo apt-get -y update",
      "sudo apt-get -y install python3.10-venv",
      "sudo apt-get -y install jq",
      "python -m venv .env",
      "source .env/bin/activate",
      # Setup MaxText
      git_clone_maxtext,
      f"cd maxtext && bash setup.sh MODE={test_mode.value} && cd ..",
      "pip install torch --index-url https://download.pytorch.org/whl/cpu",
      # Setup Loadgen
      "git clone https://github.com/mlcommons/inference.git",
      "cd inference/loadgen && pip install . && cd ../..",
  )

  run_model_cmds = (
      "source .env/bin/activate",
      "cd maxtext/MaxText/inference_mlperf/trillium",
      "gsutil cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl /tmp/processed-data.pkl",
      "export DATA_DISK_DIR=/tmp",
      "export CHECKPOINT=gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_",
      "export TOKENIZER_PATH=/home/ml-auto-solutions/maxtext/assets/tokenizer.llama2",
      "echo $TOKENIZER_PATH",
      "bash benchmarks_llama2-70b-trillium_2x4.sh -x -t -s",
      "cp /tmp/logs/*/mlperf_log_detail.txt ./",
      create_mlperf_log_converter_script,
      "python3 convert_logs.py",
      "cat metric_report.jsonl",
      f"gsutil cp metric_report.jsonl {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.PATE_M,
      num_slices=num_slices,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/maxtext",
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


USER_PREFIX = ""

gcs_subfolder_prefix = test_owner.Team.INFERENCE.value

tags = ["inference_team", "maxtext", "offline", "benchmark"]

if USER_PREFIX:
  dag_id = f"{USER_PREFIX}_maxtext_inference_offline_benchmark"
  tags.append(USER_PREFIX)
else:
  dag_id = "maxtext_inference_offline_benchmark"

with models.DAG(
    dag_id=dag_id,
    tags=tags,
    start_date=datetime.datetime(2024, 1, 19),
    schedule=None,
    catchup=False,
) as dag:
  test_name_prefix = dag_id
  maxtext_offline_benchmark = maxtext_inference_offline_benchmark_config(
      tpu_version=TpuVersion.TRILLIUM,
      tpu_cores=8,
      tpu_zone=Zone.EUROPE_WEST4_A.value,
      time_out_in_min=60,
      test_name="maxtext_inference_offline_benchmark",
      test_mode=SetupMode.STABLE,
      project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
      network=V6E_GCE_NETWORK,
      subnetwork=V6E_GCE_SUBNETWORK,
      is_tpu_reserved=True,
      maxtext_branch="",
  )
