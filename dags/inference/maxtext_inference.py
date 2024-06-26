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
from airflow import models
from dags import composer_env
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import jetstream_benchmark_serving_gce_config
from dags.multipod.configs.common import SetupMode
from dags.inference.maxtext_model_config_generator import generate_model_configs

USER_PREFIX = ""

MAXTEXT_BRANCH = ""
JETSTREAM_BRANCH = ""

maxtext_branch = "" if not MAXTEXT_BRANCH else f"-b {MAXTEXT_BRANCH}"
jetstream_branch = "" if not JETSTREAM_BRANCH else f"-b {JETSTREAM_BRANCH}"

# Run once a day at 8 am UTC (12 pm PST)
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


# def generate_model_configs(
#     test_name_prefix,
#     model_config_name,
#     sweep_model_configs,
#     axis_order,
#     ici_parallelism,
#     request_rate,
#     tpu_version,
#     tpu_cores,
# ):
#   model_configs = {}
#   model_configs["model_config_name"] = model_config_name

#   (
#       compute_axis_order,
#       prefill_cache_axis_order,
#       ar_cache_axis_order,
#   ) = axis_order.split("-")
#   compute_axis_order = ",".join(compute_axis_order)
#   prefill_cache_axis_order = ",".join(prefill_cache_axis_order)
#   ar_cache_axis_order = ",".join(ar_cache_axis_order)

#   model_configs["compute_axis_order"] = compute_axis_order
#   model_configs["prefill_cache_axis_order"] = prefill_cache_axis_order
#   model_configs["ar_cache_axis_order"] = ar_cache_axis_order
#   (
#       model_configs["ici_fsdp_parallelism"],
#       model_configs["ici_autoregressive_parallelism"],
#       model_configs["ici_tensor_parallelism"],
#   ) = ici_parallelism

#   model_configs["request_rate"] = request_rate
#   model_configs["maxtext_branch"] = sweep_model_configs["maxtext_branch"]
#   model_configs["jetstream_branch"] = sweep_model_configs["jetstream_branch"]

#   model_configs["model_name"] = sweep_model_configs["model_name"]
#   model_configs["model_mode"] = sweep_model_configs["model_mode"]
#   model_configs["quant_mode"] = sweep_model_configs["quant_mode"]
#   model_configs["sleep_time"] = sweep_model_configs["sleep_time"]
#   model_configs["tokenizer"] = sweep_model_configs["tokenizer"]
#   model_configs["weight_dtype"] = sweep_model_configs["weight_dtype"]
#   model_configs["scan_layers"] = sweep_model_configs["scan_layers"]
#   model_configs["max_prefill_predict_length"] = sweep_model_configs[
#       "max_prefill_predict_length"
#   ]
#   model_configs["max_target_length"] = sweep_model_configs["max_target_length"]
#   model_configs["attention"] = sweep_model_configs["attention"]
#   model_configs["reshape_q"] = sweep_model_configs["reshape_q"]
#   model_configs["per_device_batch_size"] = sweep_model_configs[
#       "per_device_batch_size"
#   ]
#   model_configs["checkpoint"] = sweep_model_configs["checkpoint"]
#   model_configs["quantization"] = sweep_model_configs["quantization"]
#   model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
#   model_configs["kv_quant_axis"] = sweep_model_configs["kv_quant_axis"]

#   model_configs["dataset"] = sweep_model_configs["dataset"]
#   model_configs["num_prompts"] = sweep_model_configs["num_prompts"]
#   model_configs["max_output_length"] = sweep_model_configs["max_output_length"]
#   model_configs["warmup_mode"] = sweep_model_configs["warmup_mode"]
#   model_configs["run_eval"] = sweep_model_configs["run_eval"]
#   model_configs["moe_matmul"] = sweep_model_configs.get("moe_matmul", "false")

#   per_device_batch_size = model_configs["per_device_batch_size"]
#   attention = model_configs["attention"][:3]
#   kv_quant_axis = "".join(
#       [axis for axis in model_configs["kv_quant_axis"].split("_")]
#   )
#   test_run_tag = (
#       model_config_name
#       if not kv_quant_axis
#       else f"{model_config_name}-{kv_quant_axis}"
#   )
#   test_run_tag = f"{test_run_tag}-pdbs{per_device_batch_size}-{attention}-{compute_axis_order.replace(',', '')}-{prefill_cache_axis_order.replace(',', '')}-{ar_cache_axis_order.replace(',', '')}"

#   test_name = f"{test_name_prefix}-{test_run_tag}"

#   if tpu_version == TpuVersion.V5E:
#     # v5e benchmarks
#     project_name = Project.TPU_PROD_ENV_AUTOMATED.value
#     zone = Zone.US_EAST1_C.value
#     network = V5_NETWORKS
#     subnetwork = V5E_SUBNETWORKS
#     runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value
#   elif tpu_version == TpuVersion.V5P:
#     zone = Zone.US_EAST5_A.value
#     runtime_version = RuntimeVersion.V2_ALPHA_TPUV5.value
#     project_name = Project.TPU_PROD_ENV_AUTOMATED.value
#     network = V5_NETWORKS
#     subnetwork = V5P_SUBNETWORKS

#   jetstream_benchmark_serving = (
#       jetstream_benchmark_serving_gce_config.get_config(
#           tpu_version=tpu_version,
#           tpu_cores=tpu_cores,
#           tpu_zone=zone,
#           time_out_in_min=sweep_model_configs["time_out_in_min"],
#           test_name=test_name,
#           test_mode=SetupMode.STABLE,
#           project_name=project_name,
#           runtime_version=runtime_version,
#           network=network,
#           subnetwork=subnetwork,
#           is_tpu_reserved=True,
#           model_configs=model_configs,
#           maxtext_branch=model_configs["maxtext_branch"],
#           jetstream_branch=model_configs["jetstream_branch"],
#       )
#   )

#   return jetstream_benchmark_serving


dag_id = (
    "jetstream-benchmark-serving"
    if not USER_PREFIX
    else f"{USER_PREFIX}-jetstream-benchmark-serving"
)
tags = ["inference_team", "jetstream", "maxtext", "benchmark"]
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

  test_templates = {
      # LLAMA2_7B
      LLAMA2_7B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 4), (TpuVersion.V5E, 8)],
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
              "0123-2013-2013",
              "0213-0213-0213",
              "0213-0213-0132",
          ],
      },
      f"{LLAMA2_7B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0213-0213-0213",
              "0213-0231-0213",
          ],
      },
      # LLAMA2_13B
      LLAMA2_13B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 4), (TpuVersion.V5E, 8)],
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
              "0213-0213-0213",
          ],
      },
      f"{LLAMA2_13B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline
              "0213-0213-0213",  # default
          ],
      },
      # LLAMA2_70B
      LLAMA2_70B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 240,
          "tpu_version_cores": [(TpuVersion.V5P, 8)],
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
              "0123-1203-1203",  # baseline
              "0213-0213-0213",  # default
          ],
      },
      f"{LLAMA2_70B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline
              "0213-0213-0213",  # default
          ],
      },
      # GEMMA_7B
      GEMMA_7B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 360,
          "time_out_in_min": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
          "model_name": GEMMA_7B,
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
      f"{GEMMA_7B}-{W_BF16_KV_BF16}-autoselect": {
          "attention": "autoselected",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline
              "0213-0213-0213",  # default
          ],
      },
      f"{GEMMA_7B}-{W_INT8_KV_INT8}-autoselect": {
          "attention": "autoselected",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline
              "0213-0213-0213",  # default
          ],
      },
      # MIXTRAL_8_7B
      MIXTRAL_8_7B: {
          "maxtext_branch": maxtext_branch,
          "jetstream_branch": jetstream_branch,
          "sleep_time": 240,
          "time_out_in_min": 240,
          "tpu_version_cores": [(TpuVersion.V5P, 8)],
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
          # Only used for MoE models
          "moe_matmul": "true",
      },
      f"{MIXTRAL_8_7B}-{W_BF16_KV_BF16}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline
              "0213-0213-0213",  # default
          ],
      },
      f"{MIXTRAL_8_7B}-{W_INT8_KV_INT8}-dot-product": {
          "attention": "dot_product",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203",  # baseline
              "0213-0213-0213",  # default
          ],
      },
  }

  tests = {
      # LLAMA2_7B
      f"{LLAMA2_7B}-{CHAT_MODE}-{W_BF16_KV_BF16}": test_templates[LLAMA2_7B]
      | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-dot-product"]
      | {
          "checkpoint": CKPT[LLAMA2_7B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_size": 10,
          "kv_quant_axis": "",
          "run_eval": True,
      },
      f"{LLAMA2_7B}-{CHAT_MODE}-{W_INT8_KV_INT8}": test_templates[LLAMA2_7B]
      | test_templates[f"{LLAMA2_7B}-{W_INT8_KV_INT8}-dot-product"]
      | {
          "checkpoint": CKPT[LLAMA2_7B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_size": 24,
          "kv_quant_axis": "heads_and_dkv",
          "run_eval": True,
      },
      # LLAMA2_13B
      f"{LLAMA2_13B}-{CHAT_MODE}-{W_BF16_KV_BF16}": test_templates[LLAMA2_13B]
      | test_templates[f"{LLAMA2_13B}-{W_BF16_KV_BF16}-dot-product"]
      | {
          "checkpoint": CKPT[LLAMA2_13B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_size": 12,
          "kv_quant_axis": "",
          "run_eval": True,
      },
      f"{LLAMA2_13B}-{CHAT_MODE}-{W_INT8_KV_INT8}": test_templates[LLAMA2_13B]
      | test_templates[f"{LLAMA2_13B}-{W_INT8_KV_INT8}-dot-product"]
      | {
          "checkpoint": CKPT[LLAMA2_13B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_size": 24,
          "kv_quant_axis": "heads_and_dkv",
          "run_eval": True,
      },
      # LLAMA2_70B
      f"{LLAMA2_70B}-{CHAT_MODE}-{W_BF16_KV_BF16}": test_templates[LLAMA2_70B]
      | test_templates[f"{LLAMA2_70B}-{W_BF16_KV_BF16}-dot-product"]
      | {
          "checkpoint": CKPT[LLAMA2_70B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_size": 12,
          "kv_quant_axis": "",
          "run_eval": True,
      },
      f"{LLAMA2_70B}-{CHAT_MODE}-{W_INT8_KV_INT8}": test_templates[LLAMA2_70B]
      | test_templates[f"{LLAMA2_70B}-{W_INT8_KV_INT8}-dot-product"]
      | {
          "checkpoint": CKPT[LLAMA2_70B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_size": 24,
          "kv_quant_axis": "heads_and_dkv",
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
      f"{GEMMA_7B}-{BASE_MODE}-{W_INT8_KV_INT8}": test_templates[GEMMA_7B]
      | test_templates[f"{GEMMA_7B}-{W_INT8_KV_INT8}-autoselect"]
      | {
          "checkpoint": CKPT[GEMMA_7B][BASE_MODE],
          "model_mode": BASE_MODE,
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_size": 24,
          "kv_quant_axis": "heads_and_dkv",
          "run_eval": True,
      },
      # MIXTRAL_8_7B
      f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_BF16_KV_BF16}": test_templates[MIXTRAL_8_7B]
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
      f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_INT8_KV_INT8}": test_templates[MIXTRAL_8_7B]
      | test_templates[f"{MIXTRAL_8_7B}-{W_INT8_KV_INT8}-dot-product"]
      | {
          "checkpoint": CKPT[MIXTRAL_8_7B][INSTRUCT_MODE],
          "model_mode": INSTRUCT_MODE,
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_size": 64,
          "kv_quant_axis": "heads_and_dkv",
          "run_eval": True,
      },
  }

  run_configs = [
      f"{LLAMA2_7B}-{CHAT_MODE}-{W_BF16_KV_BF16}",
      f"{LLAMA2_7B}-{CHAT_MODE}-{W_INT8_KV_INT8}",
      f"{LLAMA2_13B}-{CHAT_MODE}-{W_BF16_KV_BF16}",
      f"{LLAMA2_13B}-{CHAT_MODE}-{W_INT8_KV_INT8}",
      f"{LLAMA2_70B}-{CHAT_MODE}-{W_BF16_KV_BF16}",
      f"{LLAMA2_70B}-{CHAT_MODE}-{W_INT8_KV_INT8}",
      f"{GEMMA_7B}-{BASE_MODE}-{W_BF16_KV_BF16}",
      f"{GEMMA_7B}-{BASE_MODE}-{W_INT8_KV_INT8}",
      f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_BF16_KV_BF16}",
      f"{MIXTRAL_8_7B}-{INSTRUCT_MODE}-{W_INT8_KV_INT8}"
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
            dags.append(jetstream_benchmark_serving_kv_cache_layout)

    for i in range(1, len(dags)):
      dags[i - 1] >> dags[i]
