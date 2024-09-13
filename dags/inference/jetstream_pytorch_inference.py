"""A DAG to run jetstream-pytorch inference benchmarks with nightly version."""

import datetime
from airflow import models
from airflow.models.baseoperator import chain
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.inference.configs import jetstream_pytorch_gce_config
from dags.multipod.configs.common import SetupMode, Platform


# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="jetstream_pytorch_inference",
    schedule=SCHEDULED_TIME,
    tags=["inference_team", "jetstream_pytorch", "nightly"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "jetstream-pytorch-inference"
  test_models = {
      "llama3-8b": {
          "model_name": "llama-3",
          "size": "8b",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama3-8b",
          "checkpoint_quantized": "gs://inference-benchmarks/models/llama3-8b-quantized",
          "dataset": "openorca",
          "tokenizer": "tokenizer.model",
          "batch_sizes": [128],
          "request_rate": 100,
          "num_prompts": 200,
          "quantize": [True, False],
          "max_output_length": 1024,
          "max_cache_length": 2048,
          "sharding_config": "default_shardings/llama.yaml",
      },
      "llama2-7b": {
          "model_name": "llama-2",
          "size": "7b",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama2-7b-chat/pytorch/llama-2-7b-chat-merged",
          "checkpoint_quantized": "gs://inference-benchmarks/models/llama2-7b-chat/pytorch/llama-2-7b-chat-merged-int8-per-channel",
          "dataset": "openorca",
          "tokenizer": "tokenizer.llama2",
          "batch_sizes": [96],
          "request_rate": 100,
          "num_prompts": 200,
          "quantize": [True, False],
          "max_output_length": 1024,
          "max_cache_length": 2048,
          "sharding_config": "default_shardings/llama.yaml",
      },
      "llama2-13b": {
          "model_name": "llama-2",
          "size": "13b",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama2-13b-chat/pytorch/llama-2-13b-chat-merged",
          "checkpoint_quantized": "gs://inference-benchmarks/models/llama2-13b-chat/pytorch/llama-2-13b-chat-merged-int8-per-channel",
          "dataset": "openorca",
          "tokenizer": "tokenizer.llama2",
          "batch_sizes": [96],
          "request_rate": 100,
          "num_prompts": 200,
          "quantize": [True, False],
          "max_output_length": 1024,
          "max_cache_length": 2048,
          "sharding_config": "default_shardings/llama.yaml",
      },
      "llama2-70b": {
          "model_name": "llama-2",
          "size": "70b",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama2-70b-chat/pytorch/llama-2-70b-chat-merged-quantized",
          "checkpoint_quantized": "gs://inference-benchmarks/models/llama2-70b-chat/pytorch/llama-2-70b-chat-merged-quantized",
          "dataset": "openorca",
          "tokenizer": "tokenizer.model",
          "batch_sizes": [96],
          "request_rate": 100,
          "num_prompts": 200,
          "quantize": [True],
          "max_output_length": 1024,
          "max_cache_length": 2048,
          "sharding_config": "default_shardings/llama.yaml",
      },
  }

  for model, sweep_model_configs in test_models.items():
    for batch_size in sweep_model_configs["batch_sizes"]:
      for quantize in sweep_model_configs["quantize"]:
        for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
          # Set batch_size to a single value, not a list
          model_configs = {}
          model_configs["model_name"] = sweep_model_configs["model_name"]
          model_configs["size"] = sweep_model_configs["size"]
          model_configs["sleep_time"] = sweep_model_configs["sleep_time"]
          if quantize:
            model_configs["checkpoint"] = sweep_model_configs[
                "checkpoint_quantized"
            ]
          else:
            model_configs["checkpoint"] = sweep_model_configs["checkpoint"]
          model_configs["dataset"] = sweep_model_configs["dataset"]
          model_configs["tokenizer"] = sweep_model_configs["tokenizer"]
          model_configs["batch_size"] = batch_size
          model_configs["request_rate"] = sweep_model_configs["request_rate"]
          model_configs["num_prompts"] = sweep_model_configs["num_prompts"]
          model_configs["quantize"] = str(quantize)
          model_configs["max_output_length"] = sweep_model_configs[
              "max_output_length"
          ]
          model_configs["max_cache_length"] = sweep_model_configs[
              "max_cache_length"
          ]
          model_configs["sharding_config"] = sweep_model_configs[
              "sharding_config"
          ]

          # v5e e2e test with benchmarks
          project_name = Project.TPU_PROD_ENV_AUTOMATED.value
          zone = Zone.US_EAST1_C.value
          network = V5_NETWORKS
          subnetwork = V5E_SUBNETWORKS
          runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value

          jetstream_pytorch_nightly_1slice = jetstream_pytorch_gce_config.get_jetstream_pytorch_inference_nightly_config(
              tpu_version=tpu_version,
              tpu_cores=tpu_cores,
              tpu_zone=zone,
              runtime_version=runtime_version,
              project_name=project_name,
              time_out_in_min=60,
              is_tpu_reserved=True,
              test_name=f"{test_name_prefix}-nightly-{model}-batch_size-{batch_size}-quantized-{quantize}",
              test_mode=SetupMode.NIGHTLY,
              network=network,
              subnetwork=subnetwork,
              model_configs=model_configs,
          )
          jetstream_pytorch_nightly_1slice
