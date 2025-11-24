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

"""A helper to generate maxtext model configs."""

from dags.common.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion, V6E_GCE_NETWORK, V6E_GCE_SUBNETWORK
from dags.inference.configs import jetstream_benchmark_serving_gce_config
from dags.multipod.configs.common import SetupMode


def generate_model_configs(
    test_name_prefix,
    model_config_name,
    sweep_model_configs,
    axis_order,
    ici_parallelism,
    request_rate,
    tpu_version,
    tpu_cores,
):
  model_configs = {}
  model_configs["model_config_name"] = model_config_name

  (
      compute_axis_order,
      prefill_cache_axis_order,
      ar_cache_axis_order,
  ) = axis_order.split("-")
  compute_axis_order = ",".join(compute_axis_order)
  prefill_cache_axis_order = ",".join(prefill_cache_axis_order)
  ar_cache_axis_order = ",".join(ar_cache_axis_order)

  model_configs["compute_axis_order"] = compute_axis_order
  model_configs["prefill_cache_axis_order"] = prefill_cache_axis_order
  model_configs["ar_cache_axis_order"] = ar_cache_axis_order
  (
      model_configs["ici_fsdp_parallelism"],
      model_configs["ici_autoregressive_parallelism"],
      model_configs["ici_tensor_parallelism"],
  ) = ici_parallelism

  model_configs["request_rate"] = request_rate
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
  model_configs["attention"] = sweep_model_configs["attention"]
  model_configs["reshape_q"] = sweep_model_configs["reshape_q"]
  model_configs["per_device_batch_size"] = sweep_model_configs[
      "per_device_batch_size"
  ]
  model_configs["checkpoint"] = sweep_model_configs["checkpoint"]
  model_configs["quantization"] = sweep_model_configs["quantization"]
  model_configs["quantize_kvcache"] = sweep_model_configs["quantize_kvcache"]
  model_configs["kv_quant_dtype"] = sweep_model_configs.get(
      "kv_quant_dtype", ""
  )
  model_configs["kv_quant_axis"] = sweep_model_configs["kv_quant_axis"]

  model_configs["dataset"] = sweep_model_configs["dataset"]
  model_configs["dataset_path"] = sweep_model_configs.get("dataset_path", "")
  model_configs["num_prompts"] = sweep_model_configs["num_prompts"]
  model_configs["max_output_length"] = sweep_model_configs["max_output_length"]
  model_configs["warmup_mode"] = sweep_model_configs["warmup_mode"]
  model_configs["run_eval"] = sweep_model_configs["run_eval"]

  per_device_batch_size = model_configs["per_device_batch_size"]
  attention = model_configs["attention"][:3]
  kv_quant_axis = "".join(
      [axis for axis in model_configs["kv_quant_axis"].split("_")]
  )
  test_run_tag = (
      model_config_name
      if not kv_quant_axis
      else f"{model_config_name}-{kv_quant_axis}"
  )
  test_run_tag = f"{test_run_tag}-rate{str(request_rate).replace('.', '_')}-pdbs{per_device_batch_size}-{attention}-{compute_axis_order.replace(',', '')}-{prefill_cache_axis_order.replace(',', '')}-{ar_cache_axis_order.replace(',', '')}"

  test_name = f"{test_name_prefix}-{test_run_tag}"

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
  elif tpu_version == TpuVersion.TRILLIUM:
    zone = Zone.US_EAST5_A.value
    runtime_version = RuntimeVersion.V2_ALPHA_TPUV6.value
    project_name = Project.TPU_PROD_ENV_AUTOMATED.value
    network = V6E_GCE_NETWORK
    subnetwork = V6E_GCE_SUBNETWORK
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
