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
from airflow.models.baseoperator import chain
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import maxtext_inference_microbenchmark_gce_config
from dags.multipod.configs.common import SetupMode, Platform


# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_inference_microbenchmark",
    schedule=SCHEDULED_TIME,
    tags=["inference_team", "maxtext", "nightly", "benchmark", "microbenchmark"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "maxtext-inference-microbenchmark"
  test_models = {
      "llama2-7b": {
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 4)],
          "maxtext_logs": "gs://inference-benchmarks/models/llama2-7b/2024-04-25-14-01/",
          "scan_layers": "false",
          "weight_dtype": "bfloat16",
          "quantization": "int8",
          "quantize_kvcache": "true",
          "attention": "dot_product",
          "tokenizer": "tokenizer.llama2",
          "per_device_batch_sizes": [24],
          # (ici_fsdp_parallelism, ici_autoregressive_parallelism, ici_tensor_parallelism)
          "ici_parallelisms": [(1, 1, -1)],
          "max_prefill_predict_length": 1024,
          "max_target_length": 2048,
          "inference_microbenchmark_prefill_lengths": 1024,
          "inference_microbenchmark_stages": "generate",
          "inference_microbenchmark_loop_iters": 10,
          "enable_profiler": "false",
          "key_value_axis_order_product_id": "",
          "ar_key_axis_order": "1,2,0,3",
          "ar_value_axis_order": "1,2,0,3",
      },
  }

  for model, sweep_model_configs in test_models.items():
    # tasks_per_model = []
    for per_device_batch_size in sweep_model_configs["per_device_batch_sizes"]:
      for ici_parallelism in sweep_model_configs["ici_parallelisms"]:
        for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
          # Set per_device_batch_size to a single value, not a list
          model_configs = {}
          model_configs["model_name"] = model
          model_configs["sleep_time"] = sweep_model_configs["sleep_time"]
          model_configs["maxtext_logs"] = sweep_model_configs["maxtext_logs"]
          model_configs["scan_layers"] = sweep_model_configs["scan_layers"]
          model_configs["weight_dtype"] = sweep_model_configs["weight_dtype"]
          model_configs["quantization"] = sweep_model_configs["quantization"]
          model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
          model_configs["attention"] = sweep_model_configs["attention"]
          model_configs["tokenizer"] = sweep_model_configs["tokenizer"]
          model_configs["per_device_batch_size"] = per_device_batch_size
          ici_fsdp = ici_parallelism[0]
          ici_ar = ici_parallelism[1]
          ici_tensor = ici_parallelism[2]
          model_configs["ici_fsdp_parallelism"] = ici_fsdp
          model_configs["ici_autoregressive_parallelism"] = ici_ar
          model_configs["ici_tensor_parallelism"] = ici_tensor
          model_configs["max_target_length"] = sweep_model_configs[
              "max_target_length"
          ]
          model_configs["max_prefill_predict_length"] = sweep_model_configs[
              "max_prefill_predict_length"
          ]
          model_configs["inference_microbenchmark_prefill_lengths"] = sweep_model_configs["inference_microbenchmark_prefill_lengths"]
          model_configs["inference_microbenchmark_stages"] = sweep_model_configs["inference_microbenchmark_stages"]
          model_configs["inference_microbenchmark_loop_iters"] = sweep_model_configs["inference_microbenchmark_loop_iters"]
          model_configs["enable_profiler"] = sweep_model_configs["enable_profiler"]
          model_configs["key_value_axis_order_product_id"] = sweep_model_configs["key_value_axis_order_product_id"]
          model_configs["ar_key_axis_order"] = sweep_model_configs["ar_key_axis_order"]
          model_configs["ar_value_axis_order"] = sweep_model_configs["ar_value_axis_order"]

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
              test_name=f"{test_name_prefix}-stable-{model}-per_device_batch_size-{per_device_batch_size}-ici-fsdp{ici_fsdp}-ar{ici_ar}-tensor{ici_tensor}",
              test_mode=SetupMode.STABLE,
              network=network,
              subnetwork=subnetwork,
              model_configs=model_configs,
          ).run()
          # maxtext_nightly_1slice = maxtext_inference_microbenchmark_gce_config.get_maxtext_inference_microbenchmark_nightly_config(
          #     tpu_version=tpu_version,
          #     tpu_cores=tpu_cores,
          #     tpu_zone=zone,
          #     runtime_version=runtime_version,
          #     project_name=project_name,
          #     time_out_in_min=60,
          #     is_tpu_reserved=True,
          #     test_name=f"{test_name_prefix}-nightly-{model}-per_device_batch_size-{per_device_batch_size}-ici-fsdp{ici_fsdp}-ar{ici_ar}-tensor{ici_tensor}",
          #     test_mode=SetupMode.NIGHTLY,
          #     network=network,
          #     subnetwork=subnetwork,
          #     model_configs=model_configs,
          # ).run()
          # maxtext_stable_1slice >> maxtext_nightly_1slice
          maxtext_stable_1slice
