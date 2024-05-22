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

import json
from typing import Dict
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner
from dags.multipod.configs import common
from dags.vm_resource import TpuVersion, Project, RuntimeVersion

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value


def get_maxtext_inference_microbenchmark_nightly_config(
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
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = (
      "pip install --upgrade pip",
      # Download maxtext
      "git clone -b mor--kv-cache-layout-reformat-output https://github.com/google/maxtext.git",
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
      "base_output_directory": f"{model_configs['base_output_directory']}",
      "model_name": f"{model_configs['model_name']}",
      "tokenizer": f"{model_configs['tokenizer']}",
      "weight_dtype": f"{model_configs['weight_dtype']}",
      "inference_microbenchmark_prefill_lengths": f"{model_configs['inference_microbenchmark_prefill_lengths']}",
      "inference_microbenchmark_stages": f"{model_configs['inference_microbenchmark_stages']}",
      "inference_microbenchmark_loop_iters": f"{model_configs['inference_microbenchmark_loop_iters']}",
      "max_prefill_predict_length": f"{model_configs['max_prefill_predict_length']}",
      "max_target_length": f"{model_configs['max_target_length']}",
      "per_device_batch_size": f"{model_configs['per_device_batch_size']}",
      "ici_fsdp_parallelism": f"{model_configs['ici_fsdp_parallelism']}",
      "ici_autoregressive_parallelism": f"{model_configs['ici_autoregressive_parallelism']}",
      "ici_tensor_parallelism": f"{model_configs['ici_tensor_parallelism']}",
      "enable_profiler": f"{model_configs['enable_profiler']}",
      "scan_layers": f"{model_configs['scan_layers']}",
      "quantization": f"{model_configs['quantization']}",
      "quantize_kvcache": f"{model_configs['quantize_kvcache']}",
      "attention": f"{model_configs['attention']}",
  }

  run_model_cmds = (
      # Start virtual environment
      "source .env/bin/activate",
      # Get commit hash of the maxtext and jetstream repos
      f"export METADATA_DICT='{json.dumps(additional_metadata_dict)}'",
      'cd maxtext && export MAXTEXT_COMMIT_HASH=$(git log -1 --format="%H") && cd ..',
      'export METADATA_DICT=$(jq -c \'. + { "maxtext_commit_hash": $newVal}\' --arg newVal ${MAXTEXT_COMMIT_HASH} <<<"$METADATA_DICT")',
      "echo ${METADATA_DICT}",
      ### Benchmark
      "cd maxtext",
      # Configure flags
      "export XLA_FLAGS='--xla_disable_hlo_passes=rematerialization'",
      f"export run_name={model_configs['key_value_axis_order_product_id_list']}",
      f"""python MaxText/inference_microbenchmark_sweep.py \
          MaxText/configs/base.yml \
          base_output_directory={model_configs['base_output_directory']} \
          model_name={model_configs['model_name']} \
          tokenizer_path=assets/{model_configs['tokenizer']} \
          weight_dtype={model_configs['weight_dtype']} \
          inference_microbenchmark_prefill_lengths={model_configs['inference_microbenchmark_prefill_lengths']} \
          inference_microbenchmark_stages={model_configs['inference_microbenchmark_stages']} \
          inference_microbenchmark_loop_iters={model_configs['inference_microbenchmark_loop_iters']} \
          max_prefill_predict_length={model_configs['max_prefill_predict_length']} \
          max_target_length={model_configs['max_target_length']} \
          per_device_batch_size={model_configs['per_device_batch_size']} \
          ici_fsdp_parallelism={model_configs['ici_fsdp_parallelism']} \
          ici_tensor_parallelism={model_configs['ici_tensor_parallelism']} \
          ici_autoregressive_parallelism={model_configs['ici_autoregressive_parallelism']} \
          enable_profiler={model_configs['enable_profiler']} \
          scan_layers={model_configs['scan_layers']} \
          run_name=${{run_name}} \
          quantization={model_configs['quantization']} \
          quantize_kvcache={model_configs['quantize_kvcache']} \
          attention={model_configs['attention']} \
          inference_microbenchmark_sweep_key_value_axis_order_product_id_list={model_configs['key_value_axis_order_product_id_list']} \
          inference_microbenchmark_sweep_ar_key_axis_order_list={model_configs['ar_key_axis_order_list']} \
          inference_microbenchmark_sweep_ar_value_axis_order_list={model_configs['ar_value_axis_order_list']} \
          inference_microbenchmark_sweep_additional_metadata=${{METADATA_DICT}} \
          inference_microbenchmark_flatten_results=True""",
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
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.ANDY_Y,
      num_slices=num_slices,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/maxtext",
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
