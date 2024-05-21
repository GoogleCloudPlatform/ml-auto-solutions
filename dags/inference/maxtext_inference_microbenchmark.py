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
import itertools
import numpy
from airflow import models
from airflow.models.baseoperator import chain
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import maxtext_inference_microbenchmark_gce_config
from dags.multipod.configs.common import SetupMode, Platform


def get_concatenated_list_of_params(sweep_vm_count=1):
  cache_rank = 4
  cache_permu_values = list(itertools.permutations(range(cache_rank)))
  cache_permu_strs = [",".join([str(i) for i in value]) for value in cache_permu_values]
  cache_permu_idx_strs = {cache_permu_idx: cache_permu_str for cache_permu_idx, cache_permu_str in enumerate(cache_permu_strs)}
  num_cache_permu = len(cache_permu_strs)
  key_value_cache_idx_product_values = list(itertools.product(range(num_cache_permu), range(num_cache_permu)))
  key_value_cache_idx_product_idx_values = {key_value_cache_idx_product_idx: key_value_cache_idx_product_value for key_value_cache_idx_product_idx, key_value_cache_idx_product_value in enumerate(key_value_cache_idx_product_values)}

  key_value_axis_order_product_id_list = []
  ar_key_axis_order_str_list = []
  ar_value_axis_order_str_list = []
  for key_value_axis_order_product_id in range(len(key_value_cache_idx_product_idx_values)):
    key_axis_order_idx, value_axis_order_idx = key_value_cache_idx_product_idx_values[int(key_value_axis_order_product_id)]
    ar_key_axis_order_str = cache_permu_idx_strs[key_axis_order_idx]
    ar_value_axis_order_str = cache_permu_idx_strs[value_axis_order_idx]

    key_value_axis_order_product_id_list.append(key_value_axis_order_product_id)
    ar_key_axis_order_str_list.append(ar_key_axis_order_str)
    ar_value_axis_order_str_list.append(ar_value_axis_order_str)

  key_value_axis_order_product_id_split = numpy.array_split(key_value_axis_order_product_id_list, sweep_vm_count)
  ar_key_axis_order_str_split = numpy.array_split(ar_key_axis_order_str_list, sweep_vm_count)
  ar_value_axis_order_str_split = numpy.array_split(ar_value_axis_order_str_list, sweep_vm_count)
  key_value_axis_order_product_id_concat_list = [':'.join(list(str(y) for y in x)) for x in key_value_axis_order_product_id_split]
  ar_key_axis_order_concat_list = [':'.join(list(x)) for x in ar_key_axis_order_str_split]
  ar_value_axis_order_concat_list = [':'.join(list(x)) for x in ar_value_axis_order_str_split]

  return (
    key_value_axis_order_product_id_concat_list,
    ar_key_axis_order_concat_list,
    ar_value_axis_order_concat_list,
  )

# Run once a day at 12 pm UTC (4 am PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_inference_microbenchmark",
    schedule=SCHEDULED_TIME,
    tags=["inference_team", "maxtext", "nightly", "benchmark", "microbenchmark"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  sweep_vm_count=32
  (
    key_value_axis_order_product_id_concat_list,
    ar_key_axis_order_concat_list,
    ar_value_axis_order_concat_list,
  ) = get_concatenated_list_of_params(sweep_vm_count=sweep_vm_count)
  test_name_prefix = "maxtext-inf-microbench"
  test_models = {
      "llama2-7b": {
          "tpu_version_cores": [(TpuVersion.V5E, 4)],
          "base_output_directory": "gs://inference-benchmarks/logs/llama2-7b/microbenchmark/int8",
          "tokenizer": "tokenizer.llama2",
          "weight_dtype": "bfloat16",
          "inference_microbenchmark_prefill_lengths": 1024,
          "inference_microbenchmark_stages": "generate",
          "inference_microbenchmark_loop_iters": 10,
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "per_device_batch_sizes": [24],
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "enable_profiler": "false",
          "scan_layers": "false",
          "quantization": "int8",
          "quantize_kvcache": "true",
          "attention": "dot_product",
          "sleep_time": 120,
      },
  }

  for model, sweep_model_configs in test_models.items():
    for per_device_batch_size in sweep_model_configs["per_device_batch_sizes"]:
      for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
        for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
          for vm_number in range(sweep_vm_count):
            model_configs = {}
            model_configs["model_name"] = model
            model_configs["base_output_directory"] = sweep_model_configs["base_output_directory"]
            model_configs["tokenizer"] = sweep_model_configs["tokenizer"]
            model_configs["weight_dtype"] = sweep_model_configs["weight_dtype"]
            model_configs["inference_microbenchmark_prefill_lengths"] = sweep_model_configs["inference_microbenchmark_prefill_lengths"]
            model_configs["inference_microbenchmark_stages"] = sweep_model_configs["inference_microbenchmark_stages"]
            model_configs["inference_microbenchmark_loop_iters"] = sweep_model_configs["inference_microbenchmark_loop_iters"]
            model_configs["max_target_length"] = sweep_model_configs["max_target_length"]
            model_configs["max_prefill_predict_length"] = sweep_model_configs["max_prefill_predict_length"]
            model_configs["per_device_batch_size"] = per_device_batch_size
            ici_fsdp = ici_parallelism[0]
            ici_ar = ici_parallelism[1]
            ici_tensor = ici_parallelism[2]
            model_configs["ici_fsdp_parallelism"] = ici_fsdp
            model_configs["ici_autoregressive_parallelism"] = ici_ar
            model_configs["ici_tensor_parallelism"] = ici_tensor
            model_configs["enable_profiler"] = sweep_model_configs["enable_profiler"]
            model_configs["scan_layers"] = sweep_model_configs["scan_layers"]
            model_configs["quantization"] = sweep_model_configs["quantization"]
            model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
            model_configs["attention"] = sweep_model_configs["attention"]
            model_configs["key_value_axis_order_product_id_list"] = key_value_axis_order_product_id_concat_list[vm_number]
            model_configs["ar_key_axis_order_list"] = ar_key_axis_order_concat_list[vm_number]
            model_configs["ar_value_axis_order_list"] = ar_value_axis_order_concat_list[vm_number]
            model_configs["sleep_time"] = sweep_model_configs["sleep_time"]

            if tpu_version == TpuVersion.V5E:
              # v5e benchmarks
              project_name = Project.TPU_PROD_ENV_AUTOMATED.value
              zone = Zone.US_EAST1_C.value
              network = V5_NETWORKS
              subnetwork = V5E_SUBNETWORKS
              runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value
            elif tpu_version == TpuVersion.V5P:
              zone = Zone.US_EAST5_A.value
              runtime_version = RuntimeVersion.V2_ALPHA_TPUV5.value
              project_name = Project.TPU_PROD_ENV_AUTOMATED.value
              network = V5_NETWORKS
              subnetwork = V5P_SUBNETWORKS

            maxtext_stable_1slice = maxtext_inference_microbenchmark_gce_config.get_maxtext_inference_microbenchmark_nightly_config(
                tpu_version=tpu_version,
                tpu_cores=tpu_cores,
                tpu_zone=zone,
                runtime_version=runtime_version,
                project_name=project_name,
                time_out_in_min=60,
                is_tpu_reserved=True,
                test_name=f"{test_name_prefix}-stable-{model}-batch-{per_device_batch_size}-ici-fsdp{ici_fsdp}-ar{ici_ar}-tensor{ici_tensor}-vm-{vm_number}",
                test_mode=SetupMode.STABLE,
                network=network,
                subnetwork=subnetwork,
                model_configs=model_configs,
            ).run()
            maxtext_stable_1slice
