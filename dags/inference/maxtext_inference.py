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

"""A DAG to run MaxText inference benchmarks with nightly version."""

import datetime
import numpy as np
from airflow import models
from dags import composer_env
from dags.common.vm_resource import TpuVersion
from dags.inference.maxtext_model_config_generator import generate_model_configs

USER_PREFIX = ""

MAXTEXT_BRANCH = ""
JETSTREAM_BRANCH = ""

maxtext_branch = "" if not MAXTEXT_BRANCH else f"-b {MAXTEXT_BRANCH}"
jetstream_branch = "" if not JETSTREAM_BRANCH else f"-b {JETSTREAM_BRANCH}"

# Run once a day at 8 am UTC (12 am PST)
SCHEDULED_TIME = "0 8 * * *" if composer_env.is_prod_env() else None

LLAMA2_7B = "llama2-7b"
LLAMA2_13B = "llama2-13b"
LLAMA2_70B = "llama2-70b"
GEMMA_7B = "gemma-7b"
MIXTRAL_8_7B = "mixtral-8x7b"

BASE_MODE = "base"
CHAT_MODE = "chat"
INSTRUCT_MODE = "instruct"

W_BF16_KV_BF16 = "w-b16-kv-b16"
W_INT8_KV_INT8 = "w-i8-kv-i8"

CKPT = {
    LLAMA2_7B: {
        BASE_MODE: "gs://inference-benchmarks/models/llama2-7b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items",
        CHAT_MODE: "gs://inference-benchmarks/models/llama2-7b-chat/2024-05-24-12-39/param-only-decode-ckpt-maxtext/checkpoints/0/items",
    },
    LLAMA2_13B: {
        BASE_MODE: "gs://inference-benchmarks/models/llama2-13b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items",
        CHAT_MODE: "gs://inference-benchmarks/models/llama2-13b-chat/2024-05-24-12-39/param-only-decode-ckpt-maxtext/checkpoints/0/items",
    },
    LLAMA2_70B: {
        CHAT_MODE: "gs://inference-benchmarks/models/llama2-70b-chat/2024-05-08-23-16/param-only-decode-ckpt-maxtext/checkpoints/0/items"
    },
    GEMMA_7B: {
        BASE_MODE: "gs://inference-benchmarks/models/gemma-7b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items"
    },
    MIXTRAL_8_7B: {
        # checkpoint created using these instructions - go/mixtral-inference-testing
        INSTRUCT_MODE: "gs://vipannalla_mixtral_ckpt/moe_matmul/moe_matmul_06_15_24/checkpoints/0/items/"
    },
}


dag_id = (
    "jetstream_benchmark_serving"
    if not USER_PREFIX
    else f"{USER_PREFIX}_jetstream_benchmark_serving"
)
tags = [
    "inference_team",
    "jetstream",
    "maxtext",
    "benchmark",
    "TPU",
    "v5e-8",
    "v6e-8",
]
if USER_PREFIX:
  tags.append(USER_PREFIX)


with models.DAG(
    dag_id=dag_id,
    tags=tags,
    start_date=datetime.datetime(2024, 1, 19),
    schedule=SCHEDULED_TIME,
    catchup=False,
) as dag:
  test_name_prefix = "max-js" if not USER_PREFIX else f"{USER_PREFIX}-max-js"

  # TODO: baseline layout can be deleted if meeting one of the below conditions:
  #   - default cache layout performs better
  #   - run layout tuning to get optimized cache layout for 0213

  llama2_7B_bf16_batch_sizes = [1, 2, 4, 8, 12]
  llama2_7B_int8_batch_sizes = [1, 2, 4, 8, 12, 24]

  llama2_13B_bf16_batch_sizes = [1, 2, 4, 8]
  llama2_13B_int8_batch_sizes = [1, 2, 4, 8]

  llama2_70B_bf16_batch_sizes = [1, 4, 8, 16, 24]
  llama2_70B_int8_batch_sizes = [1, 2, 4, 8, 11]

  gemma_7B_bf16_batch_sizes = [1, 2, 4, 8, 12]
  gemma_7B_int8_batch_sizes = [1, 2, 4, 8, 12, 24]

  test_templates = {
      # LLAMA2_7B
      LLAMA2_7B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 120,
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
          "num_prompts": 1000,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{LLAMA2_7B}-{W_BF16_KV_BF16}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-2013-2013",  # optimized layout for 0123
              "0213-0213-0213",  # default layout
              "0213-0213-0132",  # optimized layout for 0213
          ],
      },
      f"{LLAMA2_7B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0213-0213-0213",  # default layout
              "0213-0231-0213",  # optimized layout for 0213
          ],
      },
      # LLAMA2_13B
      LLAMA2_13B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "model_name": LLAMA2_13B,
          "tokenizer": "tokenizer.llama2",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "reshape_q": True,
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "request_rate": [0.0],
          "num_prompts": 1000,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{LLAMA2_13B}-{W_BF16_KV_BF16}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
      f"{LLAMA2_13B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
      # LLAMA2_70B
      LLAMA2_70B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 240,
          "tpu_version_cores": [(TpuVersion.V5P, 8), (TpuVersion.TRILLIUM, 8)],
          "model_name": LLAMA2_70B,
          "tokenizer": "tokenizer.llama2",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "reshape_q": True,
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "num_prompts": 1000,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{LLAMA2_70B}-{W_BF16_KV_BF16}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
      f"{LLAMA2_70B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
      # GEMMA_7B
      GEMMA_7B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 120,
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
          "num_prompts": 1000,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{GEMMA_7B}-{W_BF16_KV_BF16}-autoselect": {
          "attention": "autoselected",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
      f"{GEMMA_7B}-{W_INT8_KV_INT8}-autoselect": {
          "attention": "autoselected",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
      # MIXTRAL_8_7B
      MIXTRAL_8_7B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 240,
          "time_out_in_min": 240,
          "tpu_version_cores": [(TpuVersion.V5P, 8), (TpuVersion.TRILLIUM, 8)],
          "model_name": MIXTRAL_8_7B,
          "tokenizer": "gs://maxtext-external/mixtral-8x7B-v0.1-Instruct/tokenizer.mistral",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 2048,
          "max_target_length": 3072,
          "reshape_q": True,
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "num_prompts": 1000,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{MIXTRAL_8_7B}-{W_BF16_KV_BF16}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
      f"{MIXTRAL_8_7B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline layout
              "0213-0213-0213",  # default layout
          ],
      },
  }

  tests_llama2_7b_bf16_base_mode_tests = {}
  tests_llama2_7b_bf16_chat_mode_tests = {}

  tests_llama2_7b_int8_base_mode_tests = {}
  tests_llama2_7b_int8_chat_mode_tests = {}

  # llama 2 7B bfloat16 tests in both base and chat mode
  for bs in llama2_7B_bf16_batch_sizes:
    tests_llama2_7b_bf16_base_mode_tests[
        f"{LLAMA2_7B}-{BASE_MODE}-{W_BF16_KV_BF16}-{bs}"
    ] = (
        test_templates[LLAMA2_7B]
        | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_7B][BASE_MODE],
            "model_mode": BASE_MODE,
            "quant_mode": W_BF16_KV_BF16,
            "quantization": "",
            "quantize_kvcache": "false",
            "per_device_batch_size": bs,
            "kv_quant_axis": "",
            "run_eval": False,
        }
    )

    tests_llama2_7b_bf16_chat_mode_tests[
        f"{LLAMA2_7B}-{CHAT_MODE}-{W_BF16_KV_BF16}-{bs}"
    ] = (
        test_templates[LLAMA2_7B]
        | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_7B][CHAT_MODE],
            "model_mode": CHAT_MODE,
            "quant_mode": W_BF16_KV_BF16,
            "quantization": "",
            "quantize_kvcache": "false",
            "per_device_batch_size": bs,
            "kv_quant_axis": "",
            "run_eval": True,
        }
    )

  # llama 2 7B int8 tests in both base and chat mode
  for bs in llama2_7B_int8_batch_sizes:
    tests_llama2_7b_int8_base_mode_tests[
        f"{LLAMA2_7B}-{BASE_MODE}-{W_INT8_KV_INT8}-{bs}"
    ] = (
        test_templates[LLAMA2_7B]
        | test_templates[f"{LLAMA2_7B}-{W_INT8_KV_INT8}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_7B][BASE_MODE],
            "model_mode": BASE_MODE,
            "quant_mode": W_INT8_KV_INT8,
            "quantization": "int8",
            "quantize_kvcache": "true",
            "kv_quant_dtype": "int8",
            "per_device_batch_size": bs,
            "kv_quant_axis": "heads_and_dkv",
            "run_eval": False,
        }
    )

    tests_llama2_7b_int8_chat_mode_tests[
        f"{LLAMA2_7B}-{CHAT_MODE}-{W_INT8_KV_INT8}-{bs}"
    ] = (
        test_templates[LLAMA2_7B]
        | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_7B][CHAT_MODE],
            "model_mode": CHAT_MODE,
            "quant_mode": W_INT8_KV_INT8,
            "quantization": "int8",
            "quantize_kvcache": "true",
            "kv_quant_dtype": "int8",
            "per_device_batch_size": bs,
            "kv_quant_axis": "heads_and_dkv",
            "run_eval": True,
        }
    )

  tests_llama2_13b_bf16_base_mode_tests = {}
  tests_llama2_13b_bf16_chat_mode_tests = {}

  tests_llama2_13b_int8_base_mode_tests = {}
  tests_llama2_13b_int8_chat_mode_tests = {}

  # llama 13B 7B bfloat16 tests in both base and chat mode

  for bs in llama2_13B_bf16_batch_sizes:
    tests_llama2_13b_bf16_base_mode_tests[
        f"{LLAMA2_13B}-{BASE_MODE}-{W_BF16_KV_BF16}-{bs}"
    ] = (
        test_templates[LLAMA2_13B]
        | test_templates[f"{LLAMA2_13B}-{W_BF16_KV_BF16}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_13B][BASE_MODE],
            "model_mode": BASE_MODE,
            "quant_mode": W_BF16_KV_BF16,
            "quantization": "",
            "quantize_kvcache": "false",
            "per_device_batch_size": bs,
            "kv_quant_axis": "",
            "run_eval": False,
        }
    )

    tests_llama2_13b_bf16_chat_mode_tests[
        f"{LLAMA2_13B}-{BASE_MODE}-{W_BF16_KV_BF16}-{bs}"
    ] = (
        test_templates[LLAMA2_13B]
        | test_templates[f"{LLAMA2_13B}-{W_BF16_KV_BF16}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_13B][CHAT_MODE],
            "model_mode": CHAT_MODE,
            "quant_mode": W_BF16_KV_BF16,
            "quantization": "",
            "quantize_kvcache": "false",
            "per_device_batch_size": bs,
            "kv_quant_axis": "",
            "run_eval": True,
        }
    )

  for bs in llama2_13B_int8_batch_sizes:
    tests_llama2_13b_int8_base_mode_tests[
        f"{LLAMA2_13B}-{BASE_MODE}-{W_INT8_KV_INT8}-{bs}"
    ] = (
        test_templates[LLAMA2_13B]
        | test_templates[f"{LLAMA2_13B}-{W_INT8_KV_INT8}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_13B][BASE_MODE],
            "model_mode": BASE_MODE,
            "quant_mode": W_INT8_KV_INT8,
            "quantization": "int8",
            "quantize_kvcache": "true",
            "kv_quant_dtype": "int8",
            "per_device_batch_size": bs,
            "kv_quant_axis": "heads_and_dkv",
            "run_eval": False,
        }
    )

    tests_llama2_13b_int8_chat_mode_tests[
        f"{LLAMA2_13B}-{CHAT_MODE}-{W_INT8_KV_INT8}-{bs}"
    ] = (
        test_templates[LLAMA2_13B]
        | test_templates[f"{LLAMA2_13B}-{W_INT8_KV_INT8}-dot-product"]
        | {
            "checkpoint": CKPT[LLAMA2_13B][CHAT_MODE],
            "model_mode": CHAT_MODE,
            "quant_mode": W_INT8_KV_INT8,
            "quantization": "int8",
            "quantize_kvcache": "true",
            "kv_quant_dtype": "int8",
            "per_device_batch_size": bs,
            "kv_quant_axis": "heads_and_dkv",
            "run_eval": True,
        }
    )

  tests_gemma_7b_bf16_base_mode_tests = {}
  tests_gemma_7b_int8_base_mode_tests = {}

  for bs in gemma_7B_bf16_batch_sizes:
    tests_gemma_7b_bf16_base_mode_tests[
        f"{GEMMA_7B}-{BASE_MODE}-{W_BF16_KV_BF16}-{bs}"
    ] = (
        test_templates[GEMMA_7B]
        | test_templates[f"{GEMMA_7B}-{W_BF16_KV_BF16}-autoselect"]
        | {
            "checkpoint": CKPT[GEMMA_7B][BASE_MODE],
            "model_mode": BASE_MODE,
            "quant_mode": W_BF16_KV_BF16,
            "quantization": "",
            "quantize_kvcache": "false",
            "per_device_batch_size": bs,
            "kv_quant_axis": "",
            "run_eval": False,
        }
    )

  for bs in gemma_7B_int8_batch_sizes:
    tests_gemma_7b_int8_base_mode_tests[
        f"{GEMMA_7B}-{BASE_MODE}-{W_INT8_KV_INT8}-{bs}"
    ] = (
        test_templates[GEMMA_7B]
        | test_templates[f"{GEMMA_7B}-{W_INT8_KV_INT8}-autoselect"]
        | {
            "checkpoint": CKPT[GEMMA_7B][BASE_MODE],
            "model_mode": BASE_MODE,
            "quant_mode": W_INT8_KV_INT8,
            "quantization": "int8",
            "quantize_kvcache": "true",
            "kv_quant_dtype": "int8",
            "per_device_batch_size": bs,
            "kv_quant_axis": "heads_and_dkv",
            "run_eval": False,
        }
    )

  tests = (
      tests_llama2_7b_bf16_base_mode_tests
      | tests_llama2_7b_bf16_chat_mode_tests
      | tests_llama2_7b_int8_base_mode_tests
      | tests_llama2_7b_int8_chat_mode_tests
      | tests_llama2_13b_bf16_base_mode_tests
      | tests_llama2_13b_bf16_chat_mode_tests
      | tests_llama2_13b_int8_base_mode_tests
      | tests_llama2_13b_int8_chat_mode_tests
      | tests_gemma_7b_bf16_base_mode_tests
      | tests_gemma_7b_int8_base_mode_tests
      | {
          # LLAMA2_70B
          f"{LLAMA2_70B}-{CHAT_MODE}-{W_BF16_KV_BF16}": test_templates[
              LLAMA2_70B
          ]
          | test_templates[f"{LLAMA2_70B}-{W_BF16_KV_BF16}-dot-product"]
          | {
              "checkpoint": CKPT[LLAMA2_70B][CHAT_MODE],
              "model_mode": CHAT_MODE,
              "quant_mode": W_BF16_KV_BF16,
              "quantization": "",
              "quantize_kvcache": "false",
              "per_device_batch_size": 24,
              "kv_quant_axis": "",
              "run_eval": True,
          },
          f"{LLAMA2_70B}-{CHAT_MODE}-{W_INT8_KV_INT8}": test_templates[
              LLAMA2_70B
          ]
          | test_templates[f"{LLAMA2_70B}-{W_INT8_KV_INT8}-dot-product"]
          | {
              "checkpoint": CKPT[LLAMA2_70B][CHAT_MODE],
              "model_mode": CHAT_MODE,
              "quant_mode": W_INT8_KV_INT8,
              "quantization": "int8",
              "quantize_kvcache": "true",
              "kv_quant_dtype": "int8",
              "per_device_batch_size": 48,
              "kv_quant_axis": "heads_and_dkv",
              "run_eval": True,
          },
          # MIXTRAL_8_7B
          f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_BF16_KV_BF16}": test_templates[
              MIXTRAL_8_7B
          ]
          | test_templates[f"{MIXTRAL_8_7B}-{W_BF16_KV_BF16}-dot-product"]
          | {
              "checkpoint": CKPT[MIXTRAL_8_7B][INSTRUCT_MODE],
              "model_mode": INSTRUCT_MODE,
              "quant_mode": W_BF16_KV_BF16,
              "quantization": "",
              "quantize_kvcache": "false",
              "per_device_batch_size": 128,
              "kv_quant_axis": "",
              "run_eval": True,
          },
          f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_INT8_KV_INT8}": test_templates[
              MIXTRAL_8_7B
          ]
          | test_templates[f"{MIXTRAL_8_7B}-{W_INT8_KV_INT8}-dot-product"]
          | {
              "checkpoint": CKPT[MIXTRAL_8_7B][INSTRUCT_MODE],
              "model_mode": INSTRUCT_MODE,
              "quant_mode": W_INT8_KV_INT8,
              "quantization": "int8",
              "quantize_kvcache": "true",
              "kv_quant_dtype": "int8",
              "per_device_batch_size": 258,
              "kv_quant_axis": "heads_and_dkv",
              "run_eval": True,
          },
      }
  )

  # run_configs = [
  #     f"{LLAMA2_7B}-{BASE_MODE}-{W_BF16_KV_BF16}",
  #     f"{LLAMA2_7B}-{BASE_MODE}-{W_INT8_KV_INT8}",
  #     f"{LLAMA2_7B}-{CHAT_MODE}-{W_BF16_KV_BF16}",
  #     f"{LLAMA2_7B}-{CHAT_MODE}-{W_INT8_KV_INT8}",
  #     f"{LLAMA2_13B}-{BASE_MODE}-{W_BF16_KV_BF16}",
  #     f"{LLAMA2_13B}-{BASE_MODE}-{W_INT8_KV_INT8}",
  #     f"{LLAMA2_13B}-{CHAT_MODE}-{W_BF16_KV_BF16}",
  #     f"{LLAMA2_13B}-{CHAT_MODE}-{W_INT8_KV_INT8}",
  #     f"{LLAMA2_70B}-{CHAT_MODE}-{W_BF16_KV_BF16}",
  #     f"{LLAMA2_70B}-{CHAT_MODE}-{W_INT8_KV_INT8}",
  #     f"{GEMMA_7B}-{BASE_MODE}-{W_BF16_KV_BF16}",
  #     f"{GEMMA_7B}-{BASE_MODE}-{W_INT8_KV_INT8}",
  #     f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_BF16_KV_BF16}",
  #     f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_INT8_KV_INT8}",
  # ]

  skip_configs = []

  dags = []
  for model_config_name, sweep_model_configs in tests.items():
    # if run_configs and model_config_name not in run_configs:
    #   continue
    if skip_configs and model_config_name in skip_configs:
      continue
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
            dags.append(jetstream_benchmark_serving_kv_cache_layout)

  # Cap the number of simultaneously requested v5e-8 due to resource contraints
  n_parallel_jobs = 4
  chunks = np.array_split(dags, n_parallel_jobs)
  for chunk in chunks:
    for i in range(1, len(chunk)):
      chunk[i - 1] >> chunk[i]
