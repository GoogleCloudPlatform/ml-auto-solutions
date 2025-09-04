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

"""Utilities to construct configs for maxtext inference microbenchmarks DAG."""

import datetime
import json
from typing import Dict
from xlml.apis import gcp_config, metric_config, task, test_config
from dags.common import test_owner
from dags.multipod.configs import common
from dags.common.vm_resource import TpuVersion, Project, RuntimeVersion

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value


def config(
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
    maxtext_branch: str = "",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = (
      "pip install --upgrade pip",
      # Download maxtext
      f"if [ ! -d maxtext ]; then git clone {maxtext_branch} https://github.com/google/maxtext.git; fi",
      # Create a python virtual environment
      "sudo apt-get -y update",
      "sudo apt-get -y install python3.10-venv",
      "sudo apt-get -y install jq",
      "python -m venv .env",
      "source .env/bin/activate",
      # Setup MaxText
      f"cd maxtext && bash setup.sh MODE={test_mode.value} && cd ..",
      "pip install torch --index-url https://download.pytorch.org/whl/cpu",
  )

  additional_metadata_dict = {
      "quant_mode": f"{model_configs['quant_mode']}",
      "two_axis_order_product_id_list": f"{model_configs['two_axis_order_product_id_list']}",
      "prefill_cache_axis_order_list": f"{model_configs['prefill_cache_axis_order_list']}",
      "ar_cache_axis_order_list": f"{model_configs['ar_cache_axis_order_list']}",
      "accelerator": f"v{tpu_version.value}-{tpu_cores}",
      "flatten_microbenchmark_results": "true",
  }

  run_model_cmds = (
      # Start virtual environment
      "source .env/bin/activate",
      # Get commit hash of the maxtext and jetstream repos
      "cd maxtext",
      f"export METADATA_DICT='{json.dumps(additional_metadata_dict)}'",
      'export MAXTEXT_COMMIT_HASH=$(git log -1 --format="%H")',
      # 'export METADATA_DICT=$(jq -c \'. + { "maxtext_commit_hash": $newVal}\' --arg newVal ${MAXTEXT_COMMIT_HASH} <<<"$METADATA_DICT")',
      # "echo ${METADATA_DICT}",
      'jq \'. + { "maxtext_commit_hash": $newVal}\' --arg newVal ${MAXTEXT_COMMIT_HASH} <<<"$METADATA_DICT" > MaxText/metadata.json',
      "cat MaxText/metadata.json",
      ### Benchmark
      # Configure flags
      "export XLA_FLAGS='--xla_disable_hlo_passes=rematerialization'",
      f"""python3 -m MaxText.inference_microbenchmark_sweep \
          MaxText/configs/base.yml \
          model_name={model_configs['model_name']} \
          tokenizer_path=assets/{model_configs['tokenizer']} \
          weight_dtype={model_configs['weight_dtype']} \
          scan_layers={model_configs['scan_layers']} \
          max_prefill_predict_length={model_configs['max_prefill_predict_length']} \
          max_target_length={model_configs['max_target_length']} \
          attention={model_configs['attention']} \
          ici_fsdp_parallelism={model_configs['ici_fsdp_parallelism']} \
          ici_autoregressive_parallelism={model_configs['ici_autoregressive_parallelism']} \
          ici_tensor_parallelism={model_configs['ici_tensor_parallelism']} \
          quantization={model_configs['quantization']} \
          quantize_kvcache={model_configs['quantize_kvcache']} \
          per_device_batch_size={model_configs['per_device_batch_size']} \
          inference_microbenchmark_prefill_lengths={model_configs['inference_microbenchmark_prefill_lengths']} \
          inference_microbenchmark_stages={model_configs['inference_microbenchmark_stages']} \
          inference_microbenchmark_loop_iters={model_configs['inference_microbenchmark_loop_iters']} \
          base_output_directory={model_configs['base_output_directory']} \
          run_name={model_configs['run_name']} \
          profiler={model_configs['profiler']} \
          save_config_to_gcs={model_configs['save_config_to_gcs']} \
          reshape_q={model_configs['reshape_q']} \
          kv_quant_axis={model_configs['kv_quant_axis']} \
          compute_axis_order={model_configs['compute_axis_order']} \
          inference_metadata_file=MaxText/metadata.json""",
      "cat inference_microbenchmark_sweep_results.jsonl",
      "mv inference_microbenchmark_sweep_results.jsonl metric_report.jsonl",
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
      task_owner=test_owner.AIRFLOW,
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
