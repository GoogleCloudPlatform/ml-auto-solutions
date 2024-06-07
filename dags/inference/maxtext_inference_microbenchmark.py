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

"""A DAG to run MaxText inference microbenchmarks with nightly version."""

import datetime
import pytz
import itertools
import numpy
from airflow import models
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import maxtext_inference_microbenchmark_gce_config
from dags.multipod.configs.common import SetupMode

LLAMA2_7B = "llama2-7b"
LLAMA2_13B = "llama2-13b"

W_BF16_KV_BF16 = "w-b16-kv-b16"
W_INT8_KV_INT8 = "w-i8-kv-i8"

BASE_OUTPUT_DIRECTORY = (
    "gs://inference-benchmarks/logs/maxtext-inference-microbenchmark"
)
test_run_datetime = datetime.datetime.now(
    pytz.timezone("America/Los_Angeles")
).strftime("%Y%m%d-%H%M%S")


def get_concatenated_list_of_params(sweep_vm_count=1):
  cache_rank = 4
  cache_permu_values = list(itertools.permutations(range(cache_rank)))
  cache_permu_strs = [
      ",".join([str(i) for i in value]) for value in cache_permu_values
  ]
  cache_permu_idx_strs = {
      cache_permu_idx: cache_permu_str
      for cache_permu_idx, cache_permu_str in enumerate(cache_permu_strs)
  }
  num_cache_permu = len(cache_permu_strs)
  key_value_cache_idx_product_values = list(
      itertools.product(range(num_cache_permu), range(num_cache_permu))
  )
  key_value_cache_idx_product_idx_values = {
      key_value_cache_idx_product_idx: key_value_cache_idx_product_value
      for key_value_cache_idx_product_idx, key_value_cache_idx_product_value in enumerate(
          key_value_cache_idx_product_values
      )
  }
  key_value_axis_order_product_id_list = []
  ar_key_axis_order_str_list = []
  ar_value_axis_order_str_list = []
  for key_value_axis_order_product_id in range(
      len(key_value_cache_idx_product_idx_values)
  ):
    (
        key_axis_order_idx,
        value_axis_order_idx,
    ) = key_value_cache_idx_product_idx_values[
        int(key_value_axis_order_product_id)
    ]
    ar_key_axis_order_str = cache_permu_idx_strs[key_axis_order_idx]
    ar_value_axis_order_str = cache_permu_idx_strs[value_axis_order_idx]
    key_value_axis_order_product_id_list.append(key_value_axis_order_product_id)
    ar_key_axis_order_str_list.append(ar_key_axis_order_str)
    ar_value_axis_order_str_list.append(ar_value_axis_order_str)
  key_value_axis_order_product_id_split = numpy.array_split(
      key_value_axis_order_product_id_list, sweep_vm_count
  )
  ar_key_axis_order_str_split = numpy.array_split(
      ar_key_axis_order_str_list, sweep_vm_count
  )
  ar_value_axis_order_str_split = numpy.array_split(
      ar_value_axis_order_str_list, sweep_vm_count
  )
  key_value_axis_order_product_id_concat_list = [
      ":".join(list(str(y) for y in x))
      for x in key_value_axis_order_product_id_split
  ]
  prefill_key_axis_order_concat_list = [
      ":".join(list(x)) for x in ar_key_axis_order_str_split
  ]
  prefill_value_axis_order_concat_list = [
      ":".join(list(x)) for x in ar_key_axis_order_str_split
  ]
  ar_key_axis_order_concat_list = [
      ":".join(list(x)) for x in ar_value_axis_order_str_split
  ]
  ar_value_axis_order_concat_list = [
      ":".join(list(x)) for x in ar_value_axis_order_str_split
  ]
  return (
      key_value_axis_order_product_id_concat_list,
      prefill_key_axis_order_concat_list,
      prefill_value_axis_order_concat_list,
      ar_key_axis_order_concat_list,
      ar_value_axis_order_concat_list,
  )


def generate_model_configs(
    test_name_prefix,
    model_config_name,
    sweep_model_configs,
    attention,
    ici_parallelism,
    per_device_batch_size,
    vm_number,
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

  model_configs["per_device_batch_size"] = per_device_batch_size

  model_configs["maxtext_branch"] = sweep_model_configs["maxtext_branch"]

  model_configs["model_name"] = sweep_model_configs["model_name"]
  model_configs["quant_mode"] = sweep_model_configs["quant_mode"]
  model_configs["sleep_time"] = sweep_model_configs["sleep_time"]

  model_configs["tokenizer"] = sweep_model_configs["tokenizer"]
  model_configs["weight_dtype"] = sweep_model_configs["weight_dtype"]
  model_configs["scan_layers"] = sweep_model_configs["scan_layers"]
  model_configs["max_prefill_predict_length"] = sweep_model_configs[
      "max_prefill_predict_length"
  ]
  model_configs["max_target_length"] = sweep_model_configs["max_target_length"]
  model_configs["quantization"] = sweep_model_configs["quantization"]
  model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
  model_configs["base_output_directory"] = sweep_model_configs[
      "base_output_directory"
  ]

  model_configs[
      "inference_microbenchmark_prefill_lengths"
  ] = sweep_model_configs["inference_microbenchmark_prefill_lengths"]
  model_configs["inference_microbenchmark_stages"] = sweep_model_configs[
      "inference_microbenchmark_stages"
  ]
  model_configs["inference_microbenchmark_loop_iters"] = sweep_model_configs[
      "inference_microbenchmark_loop_iters"
  ]
  model_configs["profiler"] = sweep_model_configs["profiler"]
  model_configs["save_config_to_gcs"] = sweep_model_configs[
      "save_config_to_gcs"
  ]
  model_configs[
      "key_value_axis_order_product_id_list"
  ] = key_value_axis_order_product_id_concat_list[vm_number]
  model_configs[
      "prefill_key_axis_order_list"
  ] = prefill_key_axis_order_concat_list[vm_number]
  model_configs[
      "prefill_value_axis_order_list"
  ] = prefill_value_axis_order_concat_list[vm_number]
  model_configs["ar_key_axis_order_list"] = ar_key_axis_order_concat_list[
      vm_number
  ]
  model_configs["ar_value_axis_order_list"] = ar_value_axis_order_concat_list[
      vm_number
  ]

  test_run_tag = f"{model_config_name}-bs{per_device_batch_size}-{attention[:4]}-vm{vm_number}"
  test_name = f"{test_name_prefix}-{test_run_tag}"
  model_configs["run_name"] = test_run_tag

  if tpu_version == TpuVersion.V5E:
    # v5e benchmarks
    project_name = Project.TPU_PROD_ENV_AUTOMATED.value
    zone = Zone.US_EAST1_C.value
    network = V5_NETWORKS
    subnetwork = V5E_SUBNETWORKS
    runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value

  maxtext_kv_cache_layout_optimization = (
      maxtext_inference_microbenchmark_gce_config.config(
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
      )
  )

  return maxtext_kv_cache_layout_optimization


with models.DAG(
    dag_id="maxtext_inference_microbenchmark",
    tags=["mor", "inference_team", "maxtext", "benchmark"],
    start_date=datetime.datetime(2024, 1, 19),
    schedule=None,
    catchup=False,
) as dag:
  test_name_prefix = "max-micro"

  sweep_vm_count = 12
  (
      key_value_axis_order_product_id_concat_list,
      prefill_key_axis_order_concat_list,
      prefill_value_axis_order_concat_list,
      ar_key_axis_order_concat_list,
      ar_value_axis_order_concat_list,
  ) = get_concatenated_list_of_params(sweep_vm_count=sweep_vm_count)

  test_templates = {
      LLAMA2_7B: {
          "maxtext_branch": "-b mor--kv-cache-sweep",
          "sleep_time": 60,
          "tpu_version_cores": [(TpuVersion.V5E, 4)],
          "model_name": LLAMA2_7B,
          "tokenizer": "tokenizer.llama2",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "attention": ["autoselected"],
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "inference_microbenchmark_prefill_lengths": "64,128,256,512,1024",
          "inference_microbenchmark_stages": "prefill,generate",
          "inference_microbenchmark_loop_iters": 10,
          "base_output_directory": f"{BASE_OUTPUT_DIRECTORY}/{test_name_prefix}/kv_cache_layout_optimization/{test_run_datetime}",
          "profiler": "xplane",
          "save_config_to_gcs": "true",
      },
  }

  tests = {
      f"{LLAMA2_7B}-{W_BF16_KV_BF16}": test_templates[LLAMA2_7B]
      | {
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_sizes": [10],
          "time_out_in_min": 330,
      },
      f"{LLAMA2_7B}-{W_INT8_KV_INT8}": test_templates[LLAMA2_7B]
      | {
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_sizes": [24],
          "time_out_in_min": 360,
      },
  }

  run_configs = [
      f"{LLAMA2_7B}-{W_INT8_KV_INT8}",
  ]

  skip_configs = [
      f"{LLAMA2_7B}-{W_BF16_KV_BF16}",
  ]

  for model_config_name, sweep_model_configs in tests.items():
    if run_configs and model_config_name not in run_configs:
      continue
    if skip_configs and model_config_name in skip_configs:
      continue

    for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
      for attention in sweep_model_configs["attention"]:
        for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
          for per_device_batch_size in sweep_model_configs[
              "per_device_batch_sizes"
          ]:
            for vm_number in range(sweep_vm_count):
              maxtext_kv_cache_layout_optimization = generate_model_configs(
                  test_name_prefix=test_name_prefix,
                  model_config_name=model_config_name,
                  sweep_model_configs=sweep_model_configs,
                  attention=attention,
                  ici_parallelism=ici_parallelism,
                  per_device_batch_size=per_device_batch_size,
                  vm_number=vm_number,
                  tpu_version=tpu_version,
                  tpu_cores=tpu_cores,
              )
