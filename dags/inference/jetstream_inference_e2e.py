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
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import jetstream_benchmark_serving_gce_config
from dags.multipod.configs.common import SetupMode

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


def generate_model_configs(
    test_name_prefix,
    model_config_name,
    sweep_model_configs,
    axis_order,
    max_output_length,
    attention,
    ici_parallelism,
    per_device_batch_size,
    request_rate,
    warmup_mode,
    tpu_version,
    tpu_cores,
):
  model_configs = {}
  model_configs["model_config_name"] = model_config_name

  model_configs["attention"] = attention
  (
      model_configs["ici_fsdp_parallelism"],
      model_configs["ici_autoregressive_parallelism"],
      model_configs["ici_tensor_parallelism"],
  ) = ici_parallelism

  prefill_axis_order, ar_axis_order = axis_order.split("-")
  prefill_axis_order = ",".join(prefill_axis_order)
  ar_axis_order = ",".join(ar_axis_order)

  model_configs["prefill_key_axis_order"] = prefill_axis_order
  model_configs["prefill_value_axis_order"] = prefill_axis_order
  model_configs["ar_key_axis_order"] = ar_axis_order
  model_configs["ar_value_axis_order"] = ar_axis_order

  model_configs["per_device_batch_size"] = per_device_batch_size
  model_configs["request_rate"] = request_rate
  model_configs["warmup_mode"] = warmup_mode
  model_configs["max_output_length"] = max_output_length

  model_configs["maxtext_branch"] = sweep_model_configs["maxtext_branch"]
  model_configs["jetstream_branch"] = sweep_model_configs["jetstream_branch"]

  model_configs["model_name"] = sweep_model_configs["model_name"]
  model_configs["model_mode"] = sweep_model_configs["model_mode"]
  model_configs["quant_mode"] = sweep_model_configs["quant_mode"]
  model_configs["sleep_time"] = sweep_model_configs["sleep_time"]
  model_configs["tokenizer"] = sweep_model_configs["tokenizer"]
  model_configs["weight_dtype"] = sweep_model_configs["weight_dtype"]
  model_configs["scan_layers"] = sweep_model_configs["scan_layers"]
  model_configs["max_prefill_predict_length"] = sweep_model_configs[
      "max_prefill_predict_length"
  ]
  model_configs["max_target_length"] = sweep_model_configs["max_target_length"]
  model_configs["checkpoint"] = sweep_model_configs["checkpoint"]
  model_configs["quantization"] = sweep_model_configs["quantization"]
  model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
  model_configs["dataset"] = sweep_model_configs["dataset"]
  model_configs["num_prompts"] = sweep_model_configs["num_prompts"]
  model_configs["run_eval"] = sweep_model_configs["run_eval"]

  test_run_tag = f"{model_config_name}-{axis_order}-bs{per_device_batch_size}-{attention[:4]}-mol{max_output_length}"

  test_name = f"{test_name_prefix}-{test_run_tag}"

  if tpu_version == TpuVersion.V5E:
    # v5e benchmarks
    project_name = Project.TPU_PROD_ENV_AUTOMATED.value
    zone = Zone.US_EAST1_C.value
    network = V5_NETWORKS
    subnetwork = V5E_SUBNETWORKS
    runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value

  jetstream_benchmark_serving = (
      jetstream_benchmark_serving_gce_config.get_config(
          tpu_version=tpu_version,
          tpu_cores=tpu_cores,
          tpu_zone=zone,
          time_out_in_min=sweep_model_configs["time_out_in_min"],
          test_name=test_name,
          test_mode=SetupMode.STABLE,
          project_name=project_name,
          runtime_version=runtime_version,
          network=network,
          subnetwork=subnetwork,
          is_tpu_reserved=True,
          model_configs=model_configs,
          maxtext_branch=model_configs["maxtext_branch"],
          jetstream_branch=model_configs["jetstream_branch"],
      )
  )

  return jetstream_benchmark_serving


with models.DAG(
    dag_id="jetstream_e2e_inference",
    schedule=None,
    tags=["inference_team", "jetstream", "maxtext", "nightly", "e2e"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "a-max-js"

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
          "attention": ["autoselected"],
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "request_rate": [0.0],
          "num_prompts": 200,
          "warmup_mode": ["full"],
      },
      f"{LLAMA2_7B}-{W_BF16_KV_BF16}-axis-order": {
          "axis_order": [
              "2103-2013",
          ]
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
          "attention": ["autoselected"],
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "request_rate": [0.0],
          "num_prompts": 200,
          "warmup_mode": ["full"],
      },
      f"{GEMMA_7B}-{W_BF16_KV_BF16}-axis-order": {
          "axis_order": [
              "2103-2013",
          ]
      },
  }

  tests = {
      # LLAMA2_7B
      f"{LLAMA2_7B}-{BASE_MODE}-{W_BF16_KV_BF16}": test_templates[LLAMA2_7B]
      | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-axis-order"]
      | {
          "checkpoint": CKPT[LLAMA2_7B][BASE_MODE],
          "model_mode": BASE_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_sizes": [12],
          "max_output_length": [1024],
          "run_eval": False,
      },
      # GEMMA_7B
      f"{GEMMA_7B}-{BASE_MODE}-{W_BF16_KV_BF16}": test_templates[GEMMA_7B]
      | test_templates[f"{GEMMA_7B}-{W_BF16_KV_BF16}-axis-order"]
      | {
          "checkpoint": CKPT[GEMMA_7B][BASE_MODE],
          "model_mode": BASE_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_sizes": [12],
          "max_output_length": [0, 1024],
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

    for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
      for max_output_length in sweep_model_configs["max_output_length"]:
        for axis_order in sweep_model_configs["axis_order"]:
          for attention in sweep_model_configs["attention"]:
            for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
              for request_rate in sweep_model_configs["request_rate"]:
                for warmup_mode in sweep_model_configs["warmup_mode"]:
                  for per_device_batch_size in sweep_model_configs[
                      "per_device_batch_sizes"
                  ]:
                    jetstream_benchmark_serving_kv_cache_layout = (
                        generate_model_configs(
                            test_name_prefix=test_name_prefix,
                            model_config_name=model_config_name,
                            sweep_model_configs=sweep_model_configs,
                            axis_order=axis_order,
                            max_output_length=max_output_length,
                            attention=attention,
                            ici_parallelism=ici_parallelism,
                            per_device_batch_size=per_device_batch_size,
                            request_rate=request_rate,
                            warmup_mode=warmup_mode,
                            tpu_version=tpu_version,
                            tpu_cores=tpu_cores,
                        )
                    )
