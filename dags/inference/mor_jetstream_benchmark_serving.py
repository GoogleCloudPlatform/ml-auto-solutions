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
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import mor_jetstream_benchmark_serving_gce_config
from dags.multipod.configs.common import SetupMode

LLAMA2_7B = "llama2-7b"
LLAMA2_13B = "llama2-13b"

BASE_MODE = "base"
CHAT_MODE = "chat"

W_BF16_KV_BF16 = "w-b16_kv-b16"
W_INT8_KV_INT8 = "w-i8_kv-i8"

CKPT = {
  LLAMA2_7B: {
    BASE_MODE: "gs://inference-benchmarks/models/llama2-7b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items",
    CHAT_MODE: "gs://inference-benchmarks/models/llama2-7b-chat/2024-05-24-12-39/param-only-decode-ckpt-maxtext/checkpoints/0/items"
  },
  LLAMA2_13B: {
    BASE_MODE: "gs://inference-benchmarks/models/llama2-13b/2024-04-25-14-01/param-only-decode-ckpt-maxtext/checkpoints/0/items",
    CHAT_MODE: "gs://inference-benchmarks/models/llama2-13b-chat/2024-05-24-12-39/param-only-decode-ckpt-maxtext/checkpoints/0/items"
  },
}

def generate_model_configs(
    test_name_prefix,
    model_config_name,
    sweep_model_configs,
    cache_axis_order,
    compute_axis_order,
    reshape_q,
    attention,
    ici_parallelism,
    per_device_batch_size,
    kv_quant_axis,
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
    model_configs["ici_tensor_parallelism"]
  ) = ici_parallelism

  prefill_cache_axis_order, ar_cache_axis_order = cache_axis_order.split("-")
  prefill_cache_axis_order = ",".join(prefill_cache_axis_order)
  ar_cache_axis_order = ",".join(ar_cache_axis_order)

  model_configs["prefill_cache_axis_order"] = prefill_cache_axis_order
  model_configs["ar_cache_axis_order"] = ar_cache_axis_order
  model_configs["compute_axis_order"] = ",".join(compute_axis_order)
  model_configs["reshape_q"] = reshape_q

  model_configs["per_device_batch_size"] = per_device_batch_size
  model_configs["kv_quant_axis"] = kv_quant_axis

  model_configs["request_rate"] = request_rate
  model_configs["warmup_mode"] = warmup_mode

  model_configs["maxtext_branch"] = sweep_model_configs["maxtext_branch"]
  model_configs["jetstream_branch"] = sweep_model_configs["jetstream_branch"]

  model_configs["model_name"] = sweep_model_configs["model_name"]
  model_configs["model_mode"] = sweep_model_configs["model_mode"]
  model_configs["quant_mode"] = sweep_model_configs["quant_mode"]
  model_configs["sleep_time"] = sweep_model_configs["sleep_time"]
  model_configs["tokenizer"] = sweep_model_configs["tokenizer"]
  model_configs["weight_dtype"] = sweep_model_configs["weight_dtype"]
  model_configs["scan_layers"] = sweep_model_configs["scan_layers"]
  model_configs["max_prefill_predict_length"] = sweep_model_configs["max_prefill_predict_length"]
  model_configs["max_target_length"] = sweep_model_configs["max_target_length"]
  model_configs["max_output_length"] = sweep_model_configs["max_output_length"]

  model_configs["checkpoint"] = sweep_model_configs["checkpoint"]
  model_configs["quantization"] = sweep_model_configs["quantization"]
  model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
  model_configs["dataset"] = sweep_model_configs["dataset"]
  model_configs["num_prompts"] = sweep_model_configs["num_prompts"]
  model_configs["run_eval"] = sweep_model_configs["run_eval"]

  test_run_tag=f"{model_config_name}-bs{per_device_batch_size}-{kv_quant_axis}-{compute_axis_order}-cache{cache_axis_order}"
  run_prefix = sweep_model_configs["run_prefix"]

  if run_prefix:
    test_name_prefix = f"{test_name_prefix}-{run_prefix}"
  test_name = f"{test_name_prefix}-{test_run_tag}"

  if tpu_version == TpuVersion.V5E:
    # v5e benchmarks
    project_name = Project.TPU_PROD_ENV_AUTOMATED.value
    zone = Zone.US_EAST1_C.value
    network = V5_NETWORKS
    subnetwork = V5E_SUBNETWORKS
    runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value

  mor_jetstream_benchmark_serving = mor_jetstream_benchmark_serving_gce_config.config(
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

  return mor_jetstream_benchmark_serving


with models.DAG(
    dag_id="mor-jetstream-benchmark-serving",
    tags=["mor", "inference_team", "jetstream", "maxtext", "benchmark"],
    start_date=datetime.datetime(2024, 1, 19),
    schedule=None,
    catchup=False,
) as dag:

  test_name_prefix = "mor-max-js"

  test_templates = {
      LLAMA2_7B: {
          # "maxtext_branch": "-b jetstream-v0.2.2",
          "run_prefix": None,
          "maxtext_branch": "-b mor--compute-axis-order-n-quantize-kv-over-hd",
          "jetstream_branch": "",
          "time_out_in_min": 60,
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
          "model_name": LLAMA2_7B,
          "tokenizer": "tokenizer.llama2",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "attention": ["dot_product"],
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "dataset": "openorca",
          "request_rate": [0.],
          "num_prompts": 1000,
          # "warmup_mode": ["sampled", "full"]
          "warmup_mode": ["full"],
          "max_output_length": 1024,
          "compute_axis_order": ["0213"],
          "reshape_q": [True],
      },
      f"{LLAMA2_7B}-{W_BF16_KV_BF16}-cache-layout": {
          "cache_axis_order": [
            "0213-0213",
            # "0213-0132",
            # "0213-1023",
            # "0213-0123",
            # "0213-1302",
            # "2013-0132"
            ],

      },
      f"{LLAMA2_7B}-{W_INT8_KV_INT8}-cache-layout": {
          "cache_axis_order": [
            "0213-0213",
            # "0231-0213",
            # "0231-0231",
            ],
      },
  }

  tests = {
      f"{LLAMA2_7B}-{BASE_MODE}-{W_BF16_KV_BF16}": test_templates[LLAMA2_7B] | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-cache-layout"] |{
          "sleep_time": 120,
          "checkpoint": CKPT[LLAMA2_7B][BASE_MODE],
          "model_mode": BASE_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_sizes": [10],
          "kv_quant_axis": [""],
          "run_eval": False,
      },
      f"{LLAMA2_7B}-{BASE_MODE}-{W_INT8_KV_INT8}": test_templates[LLAMA2_7B] | test_templates[f"{LLAMA2_7B}-{W_INT8_KV_INT8}-cache-layout"] | {
          "sleep_time": 300,
          "checkpoint": CKPT[LLAMA2_7B][BASE_MODE],
          "model_mode": BASE_MODE,
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_sizes": [24],
          "kv_quant_axis": ["d", "d-b", "hd"],
          "run_eval": False,
      },
      f"{LLAMA2_7B}-{CHAT_MODE}-{W_BF16_KV_BF16}": test_templates[LLAMA2_7B] | test_templates[f"{LLAMA2_7B}-{W_BF16_KV_BF16}-cache-layout"] | {
          "sleep_time": 120,
          "checkpoint": CKPT[LLAMA2_7B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_sizes": [10],
          "kv_quant_axis": [""],
          "run_eval": True,
      },
      f"{LLAMA2_7B}-{CHAT_MODE}-{W_INT8_KV_INT8}": test_templates[LLAMA2_7B] | test_templates[f"{LLAMA2_7B}-{W_INT8_KV_INT8}-cache-layout"] | {
          "sleep_time": 300,
          "checkpoint": CKPT[LLAMA2_7B][CHAT_MODE],
          "model_mode": CHAT_MODE,
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_sizes": [24],
          "kv_quant_axis": ["d", "d-b", "hd"],
          "run_eval": True,
      },
  }

  chat_configs = [
    f"{LLAMA2_7B}-{CHAT_MODE}-{W_BF16_KV_BF16}",
    f"{LLAMA2_7B}-{CHAT_MODE}-{W_INT8_KV_INT8}",
  ]

  base_configs = [
    f"{LLAMA2_7B}-{BASE_MODE}-{W_BF16_KV_BF16}",
    f"{LLAMA2_7B}-{BASE_MODE}-{W_INT8_KV_INT8}",
  ]

  skip_configs = [
  ]

  run_configs = [
  ] + chat_configs + base_configs

  for model_config_name, sweep_model_configs in tests.items():

    if run_configs and model_config_name not in run_configs:
      continue
    if skip_configs and model_config_name in skip_configs:
      continue

    for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
      for cache_axis_order in sweep_model_configs["cache_axis_order"]:
        for compute_axis_order in sweep_model_configs["compute_axis_order"]:
          for reshape_q in sweep_model_configs["reshape_q"]:
            for attention in sweep_model_configs["attention"]:
              for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
                for per_device_batch_size in sweep_model_configs["per_device_batch_sizes"]:
                  for kv_quant_axis in sweep_model_configs["kv_quant_axis"]:
                    for request_rate in sweep_model_configs["request_rate"]:
                      for warmup_mode in sweep_model_configs["warmup_mode"]:

                        mor_jetstream_benchmark_serving_kv_cache_layout = generate_model_configs(
                          test_name_prefix=test_name_prefix,
                          model_config_name=model_config_name,
                          sweep_model_configs=sweep_model_configs,
                          cache_axis_order=cache_axis_order,
                          compute_axis_order=compute_axis_order,
                          reshape_q=reshape_q,
                          attention=attention,
                          ici_parallelism=ici_parallelism,
                          per_device_batch_size=per_device_batch_size,
                          kv_quant_axis=kv_quant_axis,
                          request_rate=request_rate,
                          warmup_mode=warmup_mode,
                          tpu_version=tpu_version,
                          tpu_cores=tpu_cores,
                        )
