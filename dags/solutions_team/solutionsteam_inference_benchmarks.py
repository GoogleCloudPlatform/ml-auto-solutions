"""A DAG to run inference benchmarks with nightly version."""

import datetime
from airflow import models
from airflow.models.baseoperator import chain
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, RuntimeVersion
from dags.multipod.configs.common import SetupMode, Platform
from dags.solutions_team.configs.inference import inference_benchmark_config


# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="solutionsteam_inference_benchmark",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "nightly", "supported", "xlml"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "solutionsteam-inference-benchmark"
  test_models = {
      "llama3-8b": {
          # TODO: Add support for GPUs
          "accelerator_version_cores": [(TpuVersion.V5E, 8)],
          # TODO: Support other backends
          "backend": ["vllm"],
          "model_id": ["meta-llama/Meta-Llama-3.1-8B"],
          "dataset": "ShareGPT_V3_unfiltered_cleaned_split.json",
          "request_rates": "1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32",
          "num_prompts": 1000,
          "max_output_length": 1024,
          "max_cache_length": 2048,
          # TODO: Add support for quantization
          "sharding_config": "default_shardings/llama.yaml",
      },
  }

  for model, sweep_model_configs in test_models.items():
    for backend in sweep_model_configs["backend"]:
      for model_id in sweep_model_configs["model_id"]:
        for tpu_version, tpu_cores in sweep_model_configs["accelerator_version_cores"]:
          #for request_rate in sweep_model_configs["request_rates"]:
            model_configs = {}
            model_configs["backend"] = backend
            model_configs["model_id"] = model_id
            model_configs["dataset"] = sweep_model_configs["dataset"]
            #model_configs["request_rates"] = request_rate
            model_configs["request_rates"] = sweep_model_configs["request_rates"]
            model_configs["num_prompts"] = sweep_model_configs["num_prompts"]
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

            tpu_inference_benchmark = inference_benchmark_config.get_tpu_inference_gce_config(
                tpu_version=tpu_version,
                tpu_cores=tpu_cores,
                tpu_zone=zone,
                backend=backend,
                runtime_version=runtime_version,
                project_name=project_name,
                time_out_in_min=120,
                is_tpu_reserved=True,
                #test_name=f"{test_name_prefix}-nightly-{model}-{backend}-{request_rate}qps",
                test_name=f"{test_name_prefix}-nightly-{model}-{backend}",
                test_mode=SetupMode.NIGHTLY,
                network=network,
                subnetwork=subnetwork,
                model_configs=model_configs,
            )
            tpu_inference_benchmark
