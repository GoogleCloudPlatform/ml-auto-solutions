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

from airflow import models
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, Project, RuntimeVersion, V6E_GCE_NETWORK, V6E_GCE_SUBNETWORK
from dags.multipod.configs import common
from dags.multipod.configs.common import SetupMode
from xlml.apis import gcp_config, metric_config, task, test_config

PROJECT_NAME = Project.CLOUD_TPU_INFERENCE_TEST.value
RUNTIME_IMAGE = RuntimeVersion.V2_ALPHA_TPUV6.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value

# Run once a day at 5 am UTC (9 pm PST)
SCHEDULED_TIME = "0 5 * * *" if composer_env.is_prod_env() else None


def get_mlperf_converter_script():
  return """cat << 'EOL' > convert_logs.py
import json
import re
import jsonlines
import pkg_resources
import os
from typing import Optional

def find_git_dir(start_path: str) -> Optional[str]:
    current_path = os.path.abspath(start_path)
    while current_path != "/":
        git_path = os.path.join(current_path, ".git")
        if os.path.exists(git_path) and os.path.isdir(git_path):
            return current_path
        current_path = os.path.dirname(current_path)
    return None

def get_git_commit(repo_path: Optional[str] = None) -> str:
    try:
        if repo_path is None:
            repo_path = find_git_dir(os.getcwd())
            if repo_path is None:
                return "unknown"

        head_path = os.path.join(repo_path, ".git", "HEAD")
        with open(head_path, "r") as f:
            head_content = f.read().strip()

        if head_content.startswith("ref: "):
            ref_path = head_content[5:]
            ref_full_path = os.path.join(repo_path, ".git", ref_path)
            with open(ref_full_path, "r") as f:
                return f.read().strip()
        return head_content
    except Exception as e:
        print(f"Warning: Could not get git commit: {str(e)}")
        return "unknown"

def get_package_version(package_name: str) -> str:
    try:
        return pkg_resources.get_distribution(package_name).version
    except:
        return "unknown"

def convert_mlperf_log_to_jsonlines(
    log_file_path: str,
    output_path: str,
    repo_path: Optional[str] = None
) -> dict:
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
    for line in log_content.split("\\n"):
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

    dimensions["maxtext_commit_id"] = get_git_commit(repo_path)
    dimensions["jax_version"] = get_package_version("jax")
    dimensions["libtpu_version"] = get_package_version("libtpu")
    dimensions["libtpu_nightly_version"] = get_package_version("libtpu-nightly")

    result = {"metrics": metrics, "dimensions": dimensions}

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with jsonlines.open(output_path, mode="w") as writer:
        writer.write(result)

    return result

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert MLPerf log to jsonlines format")
    parser.add_argument("--log-file", type=str, required=True,
                        help="Path to the MLPerf log file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path for the output jsonlines file")
    parser.add_argument("--repo-path", type=str, default=None,
                        help="Path to the git repository (optional, will auto-detect if not provided)")

    args = parser.parse_args()

    log_file = os.path.abspath(args.log_file)
    output_file = os.path.abspath(args.output_file)
    repo_path = os.path.abspath(args.repo_path) if args.repo_path else None

    result = convert_mlperf_log_to_jsonlines(
        log_file,
        output_file,
        repo_path
    )
    print(f"Conversion complete. Output written to: {output_file}")
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
      # Setup Loadgen
      "git clone https://github.com/mlcommons/inference.git",
      "cd inference/loadgen && pip install . && cd ../..",
      # Setup MaxText
      git_clone_maxtext,
      f"cd maxtext && bash setup.sh MODE={test_mode.value} && cd ..",
      "pip install -r maxtext/MaxText/inference_mlperf/requirements.txt",
      "cd maxtext/MaxText/inference_mlperf/trillium",
      # Copy Dataset
      "gcloud storage cp gs://cloud-tpu-inference-public/mlcommons/inference/language/llama2-70b/data/processed-openorca/open_orca_gpt4_tokenized_llama.sampled_24576.pkl /tmp/processed-data.pkl",
      "cp ../user100.conf ./",
  )

  add_accuracy_to_metrics = r"""tac evaluate_offline_accuracy_log.log | grep -m1 '{.*}' | \ #  read file in reverse, grep first json-like pattern
        tr -d "'" | \  # Removes all single quotes from the output
        tr -d '\000-\037' | \  # Removes all ASCII control characters (characters 0-31 in decimal)
        sed 's/\([a-zA-Z0-9_]*\):/"\1":/g' | \  # Adds double quotes around JSON keys that aren't already quoted
        sed 's/np\.[a-zA-Z0-9_]*(\([0-9.]*\))/\1/g' | \  # Converts numpy function calls with numbers (like np.float64(0.123)) to just the number
        sed 's/{/{"metrics":{/; s/}/}}/' | \  # Wraps the JSON object in a "metrics" field
        jq -sc '.[0].metrics += .[1].metrics | .[0]' acc_metric_report.jsonl - > acc_combined_output.jsonl"""  # Combines metrics objects

  run_performance = (
      "source .env/bin/activate",
      "export DATA_DISK_DIR=/tmp",
      "export CHECKPOINT=gs://inference-benchmarks/models/llama2-70b-chat/quant/int8_",
      "export TOKENIZER_PATH=/home/ml-auto-solutions/maxtext/assets/tokenizer.llama2",
      "export LOGLEVEL=WARNING",  # the logging at the INFO level was too much and hit some quotas
      "cd maxtext/MaxText/inference_mlperf/trillium",
      "bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b performance",
      'cp "$(ls -t /tmp/logs/*performance*/mlperf_log_detail.txt | head -n1)" ./perf_log.txt',
      get_mlperf_converter_script(),
      "python3 convert_logs.py --log-file perf_log.txt --output-file perf_metric_report.jsonl",
  )

  run_accuracy = (
      "export FAST_EVAL=true",
      "bash benchmarks_llama2-70b-trillium_2x4.sh -x -s -t -b accuracy",
      'cp "$(ls -t /tmp/logs/*accuracy*/mlperf_log_detail.txt | head -n1)" ./acc_log.txt',
      'cp "$(ls -t /tmp/logs/*accuracy*/evaluate_offline_accuracy_log.log | head -n1)" ./evaluate_offline_accuracy_log.log',
      "python3 convert_logs.py --log-file acc_log.txt --output-file acc_metric_report.jsonl",
      add_accuracy_to_metrics,
      'jq -c "." perf_metric_report.jsonl > temp_perf.jsonl',
      'jq -c "." acc_combined_output.jsonl > temp_acc.jsonl',
      "cat temp_perf.jsonl temp_acc.jsonl > combined_results.jsonl",
      f"gcloud storage cp combined_results.jsonl {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )

  run_model_cmds = run_performance + run_accuracy

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
      json_lines=metric_config.JSONLinesConfig("combined_results.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


USER_PREFIX = ""
gcs_subfolder_prefix = test_owner.Team.INFERENCE.value

tags = ["inference_team", "maxtext", "offline", "benchmark", "TPU", "v6e-8"]

if USER_PREFIX:
  dag_id = f"{USER_PREFIX}_maxtext_inference_offline_benchmark"
  tags.append(USER_PREFIX)
else:
  dag_id = "maxtext_inference_offline_benchmark"

with models.DAG(
    dag_id=dag_id,
    tags=tags,
    start_date=datetime.datetime(2024, 1, 19),
    schedule=SCHEDULED_TIME,
    catchup=False,
) as dag:
  test_name_prefix = dag_id
  maxtext_offline_benchmark = maxtext_inference_offline_benchmark_config(
      tpu_version=TpuVersion.TRILLIUM,
      tpu_cores=8,
      tpu_zone=Zone.EUROPE_WEST4_A.value,
      time_out_in_min=300,
      test_name="maxtext_inference_offline_benchmark",
      test_mode=SetupMode.STABLE,
      project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
      network=V6E_GCE_NETWORK,
      subnetwork=V6E_GCE_SUBNETWORK,
      is_tpu_reserved=True,
      maxtext_branch="",
  )
