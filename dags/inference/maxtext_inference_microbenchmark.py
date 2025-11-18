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
from dags.common.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion, V6E_GCE_NETWORK, V6E_GCE_SUBNETWORK
from dags.inference.configs import maxtext_inference_microbenchmark_gce_config
from dags.multipod.configs.common import SetupMode

USER_PREFIX = ""
MAXTEXT_BRANCH = ""

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
  two_cache_idx_product_values = list(
      itertools.product(range(num_cache_permu), range(num_cache_permu))
  )
  two_cache_idx_product_idx_values = {
      two_cache_idx_product_idx: two_cache_idx_product_value
      for two_cache_idx_product_idx, two_cache_idx_product_value in enumerate(
          two_cache_idx_product_values
      )
  }
  two_axis_order_product_id_list = []
  prefill_cache_axis_order_str_list = []
  ar_cache_axis_order_str_list = []
  for two_axis_order_product_id in range(len(two_cache_idx_product_idx_values)):
    (
        prefill_cache_axis_order_idx,
        ar_cache_axis_order_idx,
    ) = two_cache_idx_product_idx_values[int(two_axis_order_product_id)]
    prefill_cache_axis_order_str = cache_permu_idx_strs[
        prefill_cache_axis_order_idx
    ]
    ar_cache_axis_order_str = cache_permu_idx_strs[ar_cache_axis_order_idx]
    two_axis_order_product_id_list.append(two_axis_order_product_id)
    prefill_cache_axis_order_str_list.append(prefill_cache_axis_order_str)
    ar_cache_axis_order_str_list.append(ar_cache_axis_order_str)
  two_axis_order_product_id_split = numpy.array_split(
      two_axis_order_product_id_list, sweep_vm_count
  )
  prefill_cache_axis_order_str_split = numpy.array_split(
      prefill_cache_axis_order_str_list, sweep_vm_count
  )
  ar_cache_axis_order_str_split = numpy.array_split(
      ar_cache_axis_order_str_list, sweep_vm_count
  )
  two_axis_order_product_id_concat_list = [
      ":".join(list(str(y) for y in x)) for x in two_axis_order_product_id_split
  ]
  prefill_cache_axis_order_concat_list = [
      ":".join(list(x)) for x in prefill_cache_axis_order_str_split
  ]
  ar_cache_axis_order_concat_list = [
      ":".join(list(x)) for x in ar_cache_axis_order_str_split
  ]
  return (
      two_axis_order_product_id_concat_list,
      prefill_cache_axis_order_concat_list,
      ar_cache_axis_order_concat_list,
  )


def generate_model_configs(
    test_name_prefix,
    model_config_name,
    sweep_model_configs,
    compute_axis_order,
    ici_parallelism,
    vm_number,
    tpu_version,
    tpu_cores,
):
  model_configs = {}
  model_configs["model_config_name"] = model_config_name

  model_configs["compute_axis_order"] = compute_axis_order
  (
      model_configs["ici_fsdp_parallelism"],
      model_configs["ici_autoregressive_parallelism"],
      model_configs["ici_tensor_parallelism"],
  ) = ici_parallelism

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
  model_configs["attention"] = sweep_model_configs["attention"]
  model_configs["per_device_batch_size"] = sweep_model_configs[
      "per_device_batch_size"
  ]
  model_configs["quantization"] = sweep_model_configs["quantization"]
  model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
  model_configs["kv_quant_axis"] = sweep_model_configs["kv_quant_axis"]

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
  model_configs["reshape_q"] = sweep_model_configs["reshape_q"]

  model_configs[
      "two_axis_order_product_id_list"
  ] = two_axis_order_product_id_concat_list[vm_number]
  model_configs[
      "prefill_cache_axis_order_list"
  ] = prefill_cache_axis_order_concat_list[vm_number]
  model_configs["ar_cache_axis_order_list"] = ar_cache_axis_order_concat_list[
      vm_number
  ]

  attention = sweep_model_configs["attention"]
  per_device_batch_size = sweep_model_configs["per_device_batch_size"]
  compute_axis_order_tag = model_configs["compute_axis_order"].replace(",", "")
  test_run_tag = f"{model_config_name}-bs{per_device_batch_size}-{attention[:3]}-{compute_axis_order_tag}-vm{vm_number}"
  test_name = f"{test_name_prefix}-{test_run_tag}"
  model_configs["run_name"] = test_run_tag

  if tpu_version == TpuVersion.V5E:
    # v5e benchmarks
    project_name = Project.TPU_PROD_ENV_AUTOMATED.value
    zone = Zone.US_EAST1_C.value
    network = V5_NETWORKS
    subnetwork = V5E_SUBNETWORKS
    runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value
  if tpu_version == TpuVersion.TRILLIUM:
    project_name = Project.CLOUD_ML_AUTO_SOLUTIONS.value
    zone = Zone.EUROPE_WEST4_A.value
    network = V6E_GCE_NETWORK
    subnetwork = V6E_GCE_SUBNETWORK
    runtime_version = RuntimeVersion.V2_ALPHA_TPUV6.value

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


dag_id = (
    "maxtext-inference-microbenchmark"
    if not USER_PREFIX
    else f"{USER_PREFIX}-maxtext-inference-microbenchmark"
)
tags = ["inference_team", "maxtext", "microbenchmark", "TPU", "v5e-8", "v6e-8"]

if USER_PREFIX:
  dag_id = f"{USER_PREFIX}-maxtext-inference-microbenchmark"
  tags.append(USER_PREFIX)

with models.DAG(
    dag_id=dag_id,
    tags=tags,
    start_date=datetime.datetime(2024, 1, 19),
    schedule=None,
    catchup=False,
) as dag:
  test_name_prefix = (
      "max-micro" if not USER_PREFIX else f"{USER_PREFIX}-max-micro"
  )

  sweep_vm_count = 8
  (
      two_axis_order_product_id_concat_list,
      prefill_cache_axis_order_concat_list,
      ar_cache_axis_order_concat_list,
  ) = get_concatenated_list_of_params(sweep_vm_count=sweep_vm_count)

  test_templates = {
      LLAMA2_7B: {
          "maxtext_branch": ""
          if not MAXTEXT_BRANCH
          else f"-b {MAXTEXT_BRANCH}",
          "sleep_time": 60,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "model_name": LLAMA2_7B,
          "tokenizer": "tokenizer.llama2",
          "weight_dtype": "bfloat16",
          "scan_layers": "false",
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "attention": "dot_product",
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "inference_microbenchmark_prefill_lengths": "64,128,256,512,1024",
          "inference_microbenchmark_stages": "prefill, generate",
          "inference_microbenchmark_loop_iters": 10,
          "base_output_directory": f"{BASE_OUTPUT_DIRECTORY}/{test_name_prefix}/kv_cache_layout_optimization/{test_run_datetime}",
          "profiler": "xplane",
          "save_config_to_gcs": "true",
          "reshape_q": "true",
          "compute_axis_order": ["0,2,1,3"],
      },
  }

  tests = {
      f"{LLAMA2_7B}-{W_BF16_KV_BF16}": test_templates[LLAMA2_7B]
      | {
          "quant_mode": W_BF16_KV_BF16,
          "quantization": "",
          "quantize_kvcache": "false",
          "per_device_batch_size": 10,
          "kv_quant_axis": "",
          "time_out_in_min": 330,
      },
      f"{LLAMA2_7B}-{W_INT8_KV_INT8}": test_templates[LLAMA2_7B]
      | {
          "quant_mode": W_INT8_KV_INT8,
          "quantization": "int8",
          "quantize_kvcache": "true",
          "per_device_batch_size": 24,
          "kv_quant_axis": "heads_and_dkv",
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
      for compute_axis_order in sweep_model_configs["compute_axis_order"]:
        for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
          for vm_number in range(sweep_vm_count):
            maxtext_kv_cache_layout_optimization = generate_model_configs(
                test_name_prefix=test_name_prefix,
                model_config_name=model_config_name,
                sweep_model_configs=sweep_model_configs,
                compute_axis_order=compute_axis_order,
                ici_parallelism=ici_parallelism,
                vm_number=vm_number,
                tpu_version=tpu_version,
                tpu_cores=tpu_cores,
            )
