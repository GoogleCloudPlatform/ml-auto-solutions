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

"""A DAG to run JetStream inference E2E test."""

import datetime
from airflow import models
from dags.common.vm_resource import TpuVersion
from dags.inference.maxtext_model_config_generator import generate_model_configs

"""A JetStream inference E2E test (JAX nightly, no schedule) DAG.

Usage:
gcloud composer environments run ml-automation-solutions \
  --project=cloud-ml-auto-solutions \
  --location=us-central1 dags trigger \
  -- \
  jetstream_e2e_inference

"""

LLAMA2_7B = "llama2-7b"
GEMMA_7B = "gemma-7b"

BASE_MODE = "base"

W_BF16_KV_BF16 = "w-b16-kv-b16"

CKPT = {
    LLAMA2_7B: {
        BASE_MODE: "gs://inference-benchmarks/models/llama2-7b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items",
    },
    GEMMA_7B: {
        BASE_MODE: "gs://inference-benchmarks/models/gemma-7b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items"
    },
}


with models.DAG(
    dag_id="jetstream_e2e_inference",
    schedule="0 0 * * *",
    tags=[
        "inference_team",
        "jetstream",
        "maxtext",
        "nightly",
        "e2e",
        "TPU",
        "v5e-8",
        "v6e-8",
    ],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "jetstream-e2e-inference"

  test_templates = {
      # LLAMA2_7B
      LLAMA2_7B: {
          "maxtext_branch": "",
          "jetstream_branch": "",
          "sleep_time": 360,
          "time_out_in_min": 60,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "model_name": LLAMA2_7B,
          "tokenizer": "tokenizer.llama2",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "reshape_q": True,
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "num_prompts": 200,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{LLAMA2_7B}-{W_BF16_KV_BF16}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-2013-2013",
          ],
      },
      # GEMMA_7B
      GEMMA_7B: {
          "maxtext_branch": "",
          "jetstream_branch": "",
          "sleep_time": 360,
          "time_out_in_min": 60,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "model_name": GEMMA_7B,
          "tokenizer": "tokenizer.gemma",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "reshape_q": True,
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "sharegpt",
          "dataset_path": "~/ShareGPT_V3_unfiltered_cleaned_split.json",
          "request_rate": [0.0],
          "num_prompts": 200,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{GEMMA_7B}-{W_BF16_KV_BF16}-autoselect": {
          "attention": "autoselected",
          "request_rate": [0.0],
          "axis_order": ["0123-1203-1203"],
      },
  }

  tests = {
      # LLAMA2_7B
      f"{LLAMA2_7B}-{BASE_MODE}-{W_BF16_KV_BF16}": test_templates[LLAMA2_7B]
      | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-dot-product"]
      | {
          "checkpoint": CKPT[LLAMA2_7B][BASE_MODE],
          "model_mode": BASE_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_size": 12,
          "kv_quant_axis": "",
          "run_eval": True,
      },
      # GEMMA_7B
      f"{GEMMA_7B}-{BASE_MODE}-{W_BF16_KV_BF16}": test_templates[GEMMA_7B]
      | test_templates[f"{GEMMA_7B}-{W_BF16_KV_BF16}-autoselect"]
      | {
          "checkpoint": CKPT[GEMMA_7B][BASE_MODE],
          "model_mode": BASE_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_size": 12,
          "kv_quant_axis": "",
          "run_eval": True,
      },
  }

  run_configs = [
      f"{LLAMA2_7B}-{BASE_MODE}-{W_BF16_KV_BF16}",
      f"{GEMMA_7B}-{BASE_MODE}-{W_BF16_KV_BF16}",
  ]

  skip_configs = []

  for model_config_name, sweep_model_configs in tests.items():
    if run_configs and model_config_name not in run_configs:
      continue
    if skip_configs and model_config_name in skip_configs:
      continue
    dags = []
    for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
      for axis_order in sweep_model_configs["axis_order"]:
        for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
          for request_rate in sweep_model_configs["request_rate"]:
            jetstream_benchmark_serving_kv_cache_layout = (
                generate_model_configs(
                    test_name_prefix=test_name_prefix,
                    model_config_name=model_config_name,
                    sweep_model_configs=sweep_model_configs,
                    axis_order=axis_order,
                    ici_parallelism=ici_parallelism,
                    request_rate=request_rate,
                    tpu_version=tpu_version,
                    tpu_cores=tpu_cores,
                )
            )
