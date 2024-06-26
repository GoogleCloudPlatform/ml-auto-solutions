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

"""Utilities to construct configs for maxtext inference DAG."""

import datetime
import json
from typing import Dict
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner
from dags.multipod.configs import common
from dags.vm_resource import TpuVersion, Project, RuntimeVersion

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value


def get_maxtext_inference_nightly_config(
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
    model_configs: Dict = {},
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = (
      "pip install --upgrade pip",
      # Download jetstream and maxtext
      "git clone https://github.com/google/maxtext.git",
      "git clone https://github.com/google/JetStream.git",
      # Create a python virtual environment
      "sudo apt-get -y update",
      "sudo apt-get -y install python3.10-venv",
      "sudo apt-get -y install jq",
      "python -m venv .env",
      "source .env/bin/activate",
      # Setup MaxText & JetStream
      f"cd maxtext && bash setup.sh MODE={test_mode.value} && cd ..",
      "cd JetStream && pip install -e . && cd benchmarks && pip install -r requirements.in",
      "pip install torch --index-url https://download.pytorch.org/whl/cpu",
  )

  additional_metadata_dict = {
      "model_mode": f"{model_configs['model_mode']}",
      "checkpoint": f"{model_configs['checkpoint']}",
      "scan_layers": f"{model_configs['scan_layers']}",
      "dataset": f"{model_configs['dataset']}",
      "max_prefill_predict_length": f"{model_configs['max_prefill_predict_length']}",
      "max_target_length": f"{model_configs['max_target_length']}",
      "max_output_length": f"{model_configs['max_output_length']}",
      "ici_fsdp_parallelism": f"{model_configs['ici_fsdp_parallelism']}",
      "ici_autoregressive_parallelism": f"{model_configs['ici_autoregressive_parallelism']}",
      "ici_tensor_parallelism": f"{model_configs['ici_tensor_parallelism']}",
      "per_device_batch_size": f"{model_configs['per_device_batch_size']}",
      "weight_dtype": f"{model_configs['weight_dtype']}",
  }

  # Let gcs path be directly used, else use maxtext/assets dir
  if not model_configs["tokenizer"].startswith("gs://"):
    tokenizer_path = f"assets/{model_configs['tokenizer']}"
    full_tokenizer_path = f"maxtext/assets/{model_configs['tokenizer']}"
  else:
    tokenizer_path = model_configs["tokenizer"]
    full_tokenizer_path = model_configs["tokenizer"]

  run_model_cmds = (
      # Start virtual environment
      "source .env/bin/activate",
      # Get commit hash of the maxtext and jetstream repos
      f"export METADATA_DICT='{json.dumps(additional_metadata_dict)}'",
      'cd maxtext && export MAXTEXT_COMMIT_HASH=$(git log -1 --format="%H") && cd ..',
      'cd JetStream && export JETSTREAM_COMMIT_HASH=$(git log -1 --format="%H") && cd ..',
      'export METADATA_DICT=$(jq -c \'. + { "maxtext_commit_hash": $newVal}\' --arg newVal ${MAXTEXT_COMMIT_HASH} <<<"$METADATA_DICT")',
      'export METADATA_DICT=$(jq -c \'. + { "jetstream_commit_hash": $newVal}\' --arg newVal ${JETSTREAM_COMMIT_HASH} <<<"$METADATA_DICT")',
      ### Benchmark
      "cd maxtext",
      # Configure flags
      f"export UNSCANNED_CKPT_PATH={model_configs['checkpoint']}",
      f"export TOKENIZER_PATH={tokenizer_path}",
      "export LOAD_PARAMETERS_PATH=${UNSCANNED_CKPT_PATH}",
      f"export MAX_PREFILL_PREDICT_LENGTH={model_configs['max_prefill_predict_length']}",
      f"export MAX_TARGET_LENGTH={model_configs['max_target_length']}",
      f"export MODEL_NAME={model_configs['model_name']}",
      f"export ICI_FSDP_PARALLELISM={model_configs['ici_fsdp_parallelism']}",
      f"export ICI_AUTOREGRESSIVE_PARALLELISM={model_configs['ici_autoregressive_parallelism']}",
      f"export ICI_TENSOR_PARALLELISM={model_configs['ici_tensor_parallelism']}",
      f"export SCAN_LAYERS={model_configs['scan_layers']}",
      f"export WEIGHT_DTYPE={model_configs['weight_dtype']}",
      f"export PER_DEVICE_BATCH_SIZE={model_configs['per_device_batch_size']}",
      f"export MOE_MATMUL={model_configs['moe_matmul']}",
      # Start JetStream MaxText server in the background
      """python MaxText/maxengine_server.py \
        MaxText/configs/inference_jetstream.yml \
        tokenizer_path=${TOKENIZER_PATH} \
        load_parameters_path=${LOAD_PARAMETERS_PATH} \
        max_prefill_predict_length=${MAX_PREFILL_PREDICT_LENGTH} \
        max_target_length=${MAX_TARGET_LENGTH} \
        model_name=${MODEL_NAME} \
        ici_fsdp_parallelism=${ICI_FSDP_PARALLELISM} \
        ici_autoregressive_parallelism=${ICI_AUTOREGRESSIVE_PARALLELISM} \
        ici_tensor_parallelism=${ICI_TENSOR_PARALLELISM} \
        scan_layers=${SCAN_LAYERS} \
        weight_dtype=${WEIGHT_DTYPE} \
        per_device_batch_size=${PER_DEVICE_BATCH_SIZE} \
        moe_matmul=${MOE_MATMUL} &""",
      "cd ..",
      # Give server time to start
      f"sleep {model_configs['sleep_time']}",
      # Run benchmark, run eval, save benchmark and eval results, and save predictions to /tmp/request-outputs.json
      f"""python JetStream/benchmarks/benchmark_serving.py \
      --tokenizer {full_tokenizer_path} \
      --model {model_configs['model_name']} \
      --num-prompts {model_configs['num_prompts']}  \
      --dataset {model_configs['dataset']} \
      --max-output-length {model_configs['max_output_length']} \
      --request-rate {model_configs['request_rate']} \
      --warmup-mode sampled \
      --save-result \
      --additional-metadata-metrics-to-save ${{METADATA_DICT}} \
      --save-request-outputs \
      --run-eval true""",
      'export BENCHMARK_OUTPUT=$(find . -name "*JetStream*" -type f -printf "%T@ %Tc %p\n" | sort -n | head -1 | awk \'NF>1{print $NF}\')',
      # Stop JetStream server
      "kill -9 %%",
      # Upload results (in jsonlines format) to GCS to be post-processed into
      # our BigQuery table
      "mv ${BENCHMARK_OUTPUT} metric_report.jsonl",
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
      task_owner=test_owner.ANDY_Y,
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
