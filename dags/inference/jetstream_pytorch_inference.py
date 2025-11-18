"""A DAG to run jetstream-pytorch inference benchmarks with nightly version."""

import datetime
from airflow import models
from airflow.models.baseoperator import chain
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion, V6E_GCE_NETWORK, V6E_GCE_SUBNETWORK
from dags.inference.configs import jetstream_pytorch_gce_config
from dags.multipod.configs.common import SetupMode, Platform
import numpy as np

# Run once a day at 4 am UTC (8 pm PST)
SCHEDULED_TIME = "0 4 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="jetstream_pytorch_inference",
    schedule=SCHEDULED_TIME,
    tags=[
        "inference_team",
        "jetstream_pytorch",
        "nightly",
        "TPU",
        "v5e-8",
        "v6e-8",
    ],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "jetstream-pytorch-inference"
  test_models = {
      "llama3-8b": {
          "model_name": "llama-3",
          "size": "8b",
          "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama3-8b-instruct/pytorch/llama3-8b-instruct-hf",
          "dataset": "openorca",
          "batch_sizes": [8, 32, 64, 128],
          "request_rate": 100,
          "num_prompts": 1000,
          "max_output_length": 1024,
          "quantize": [True, False],
      },
      "llama2-7b": {
          "model_name": "llama-2",
          "size": "7b",
          "model_id": "meta-llama/Llama-2-7b-chat-hf",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama2-7b-chat/pytorch/llama-2-7b-chat-hf",
          "dataset": "openorca",
          "batch_sizes": [8, 32, 64, 96, 128],
          "request_rate": 100,
          "num_prompts": 1000,
          "max_output_length": 1024,
          "quantize": [True, False],
      },
      "gemma-7b": {
          "model_name": "gemma",
          "size": "7b",
          "model_id": "google/gemma-7b",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "checkpoint": "gs://inference-benchmarks/models/gemma-7b-it/pytorch/gemma-7b-it-hf",
          "dataset": "openorca",
          "tokenizer": "tokenizer.model",
          "batch_sizes": [8, 32, 64, 128],
          "request_rate": 100,
          "num_prompts": 1000,
          "max_output_length": 1024,
          "quantize": [True, False],
      },
      "llama2-13b": {
          "model_name": "llama-2",
          "size": "13b",
          "model_id": "meta-llama/Llama-2-13b-chat-hf",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama2-13b-chat/pytorch/llama-2-13b-chat-hf",
          "dataset": "openorca",
          "tokenizer": "tokenizer.llama2",
          "batch_sizes": [8, 32, 64, 96],
          "request_rate": 100,
          "num_prompts": 1000,
          "max_output_length": 1024,
          "quantize": [True, False],
      },
      "llama2-70b": {
          "model_name": "llama-2",
          "size": "70b",
          "model_id": "meta-llama/Llama-2-70b-chat-hf",
          "sleep_time": 120,
          "tpu_version_cores": [(TpuVersion.V5E, 8), (TpuVersion.TRILLIUM, 8)],
          "checkpoint": "gs://inference-benchmarks/models/llama2-70b-chat/pytorch/llama-2-70b-chat-hf",
          "dataset": "openorca",
          "tokenizer": "tokenizer.model",
          "batch_sizes": [8, 32, 64, 96],
          "request_rate": 100,
          "num_prompts": 1000,
          "max_output_length": 1024,
          "quantize": [True],
      },
  }
  skip_settings = (
      ("llama-2", "13b", 96, "False"),
      ("llama-2", "7b", 128, "False"),
  )
  dags = []
  for model, sweep_model_configs in test_models.items():
    for batch_size in sweep_model_configs["batch_sizes"]:
      for quantize in sweep_model_configs["quantize"]:
        for tpu_version, tpu_cores in sweep_model_configs["tpu_version_cores"]:
          # Set batch_size to a single value, not a list
          model_configs = {}
          model_configs["model_name"] = sweep_model_configs["model_name"]
          model_configs["size"] = sweep_model_configs["size"]
          model_configs["model_id"] = sweep_model_configs["model_id"]
          model_configs["sleep_time"] = sweep_model_configs["sleep_time"]
          model_configs["checkpoint"] = sweep_model_configs["checkpoint"]
          model_configs["dataset"] = sweep_model_configs["dataset"]
          model_configs["batch_size"] = batch_size
          model_configs["per_device_batch_size"] = batch_size // tpu_cores
          model_configs["request_rate"] = sweep_model_configs["request_rate"]
          model_configs["num_prompts"] = sweep_model_configs["num_prompts"]
          model_configs["quantize"] = str(quantize)
          model_configs["max_output_length"] = sweep_model_configs[
              "max_output_length"
          ]
          # Llama-2 13b unquantized with bs 96 cannot hold in v5e-8
          if (
              model_configs["model_name"],
              model_configs["size"],
              model_configs["batch_size"],
              model_configs["quantize"],
          ) in skip_settings:
            continue

          # v5e e2e test with benchmarks
          if tpu_version == TpuVersion.TRILLIUM:
            project_name = Project.CLOUD_ML_AUTO_SOLUTIONS.value
            zone = Zone.EUROPE_WEST4_A.value
            network = V6E_GCE_NETWORK
            subnetwork = V6E_GCE_SUBNETWORK
            runtime_version = RuntimeVersion.V2_ALPHA_TPUV6.value

          else:
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
          dags.append(jetstream_pytorch_nightly_1slice)
  n_parallel_jobs = 10
  chunks = np.array_split(dags, n_parallel_jobs)
  for chunk in chunks:
    for i in range(1, len(chunk)):
      chunk[i - 1] >> chunk[i]
