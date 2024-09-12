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

"""Utilities to construct configs for jetstream-pytorch inference DAG."""

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


def get_jetstream_pytorch_inference_nightly_config(
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
      # Create a python virtual environment
      "sudo apt-get -y update",
      "sudo apt-get -y install python3.10-venv",
      "sudo apt-get -y install jq",
      "python -m venv .env",
      "source .env/bin/activate",
      # Setup jetstream-pytorch
      "git clone https://github.com/google/jetstream-pytorch.git",
      "cd jetstream-pytorch && source install_everything.sh",
  )

  additional_metadata_dict = model_configs.copy()
  additional_metadata_dict.pop("sleep_time")

  run_model_cmds = (
      # Start virtual environment
      "source .env/bin/activate",
      # Get commit hash of the jetstream-pytorch and jetstream repos
      f"export METADATA_DICT='{json.dumps(additional_metadata_dict)}'",
      'cd jetstream-pytorch && export JETSTREAM_PYTORCH_COMMIT_HASH=$(git log -1 --format="%H") && cd ..',
      'cd jetstream-pytorch/deps/JetStream && export JETSTREAM_COMMIT_HASH=$(git log -1 --format="%H") && cd ../../..',
      'export METADATA_DICT=$(jq -c \'. + { "jetstream_pytorch_commit_hash": $newVal}\' --arg newVal ${JETSTREAM_PYTORCH_COMMIT_HASH} <<<"$METADATA_DICT")',
      'export METADATA_DICT=$(jq -c \'. + { "jetstream_commit_hash": $newVal}\' --arg newVal ${JETSTREAM_COMMIT_HASH} <<<"$METADATA_DICT")',
      ### Benchmark
      "cd jetstream-pytorch",
      # Configure flags
      f"export MODEL_NAME={model_configs['model_name']}",
      f"export SIZE={model_configs['size']}",
      f"export BATCH_SIZE={model_configs['batch_size']}",
      f"export MAX_CACHE_LEN={model_configs['max_cache_length']}",
      f"export CKPT_PATH={model_configs['checkpoint']}",
      f"export TOKENIZER_PATH=$(pwd)/ckpt_dir/{model_configs['tokenizer']}",
      f"export SHARDING_CONFIG={model_configs['sharding_config']}",
      f"export QUANTIZE={str(model_configs['quantize'])}",
      f"export QUANTIZE_KV_CACHE={str(model_configs['quantize'])}",
      "mkdir /dev/shm/ckpt_dir",
      "gsutil cp -r ${CKPT_PATH}/* /dev/shm/ckpt_dir/",
      # Start jetstream-pytorch server in the background
      """python run_server.py \
        --model_name=${MODEL_NAME} \
        --size=${SIZE} \
        --batch_size=${BATCH_SIZE} \
        --max_cache_length=${MAX_CACHE_LEN} \
        --checkpoint_path=/dev/shm/ckpt_dir \
        --tokenizer_path=${TOKENIZER_PATH} \
        --quantize_weights=${QUANTIZE} \
        --quantize_kv_cache=${QUANTIZE_KV_CACHE} \
        --sharding_config=${SHARDING_CONFIG} &""",
      """pip install -r deps/JetStream/benchmarks/requirements.in \
                     -r deps/JetStream/requirements.in \
                     -r deps/JetStream/requirements.txt """,
      # redo the install everything script to keep jax library versions in accord
      "source ./install_everything.sh",
      "pip install --force-reinstall --no-deps nltk==3.8.1",
      # Give server time to start
      f"sleep {model_configs['sleep_time']}",
      # Run benchmark, run eval, save benchmark and eval results, and save predictions to /tmp/request-outputs.json
      f"""python deps/JetStream/benchmarks/benchmark_serving.py \
      --tokenizer ckpt_dir/{model_configs['tokenizer']} \
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
      task_owner=test_owner.XIANG_S,
      num_slices=num_slices,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/jetstream_pytorch",
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
