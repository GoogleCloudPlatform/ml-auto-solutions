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
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import jetstream_benchmark_serving_gce_config
from dags.multipod.configs.common import SetupMode
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


with models.DAG(
    dag_id="jetstream_e2e_inference",
    schedule=None,
    tags=["inference_team", "jetstream", "maxtext", "nightly", "e2e"],
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
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
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
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
          "model_name": GEMMA_7B,
          "tokenizer": "tokenizer.gemma",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "reshape_q": True,
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "request_rate": [0.0],
          "num_prompts": 200,
          "max_output_length": 1024,
          "warmup_mode": "full",
      },
      f"{GEMMA_7B}-{W_BF16_KV_BF16}-autoselect": {
          "attention": "autoselected",
          "request_rate": [0.0],
          "axis_order": [
              "0123-1203-1203"
          ],
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

  # skip_configs = []

  # for model_config_name, sweep_model_configs in tests.items():
  #   if run_configs and model_config_name not in run_configs:
  #     continue
  #   if skip_configs and model_config_name in skip_configs:
  #     continue

  #   for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
  #     for max_output_length in sweep_model_configs["max_output_length"]:
  #       for axis_order in sweep_model_configs["axis_order"]:
  #         for attention in sweep_model_configs["attention"]:
  #           for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
  #             for request_rate in sweep_model_configs["request_rate"]:
  #               for warmup_mode in sweep_model_configs["warmup_mode"]:
  #                 for per_device_batch_size in sweep_model_configs[
  #                     "per_device_batch_sizes"
  #                 ]:
  #                   jetstream_benchmark_serving_kv_cache_layout = (
  #                       generate_model_configs(
  #                           test_name_prefix=test_name_prefix,
  #                           model_config_name=model_config_name,
  #                           sweep_model_configs=sweep_model_configs,
  #                           axis_order=axis_order,
  #                           max_output_length=max_output_length,
  #                           attention=attention,
  #                           ici_parallelism=ici_parallelism,
  #                           per_device_batch_size=per_device_batch_size,
  #                           request_rate=request_rate,
  #                           warmup_mode=warmup_mode,
  #                           tpu_version=tpu_version,
  #                           tpu_cores=tpu_cores,
  #                       )
  #                   )
