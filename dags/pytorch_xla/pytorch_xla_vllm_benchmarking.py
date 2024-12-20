"""A DAG to run vllm benchmarks with nightly version."""

import datetime
import enum
from airflow import models
from airflow.models.baseoperator import chain
from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import AcceleratorType, GpuVersion, TpuVersion, Region, Zone, Project, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, BM_NETWORKS, A100_BM_SUBNETWORKS, ImageProject, ImageFamily, MachineVersion, RuntimeVersion
from dags.multipod.configs.common import SetupMode, Platform
from dags.solutions_team.configs.vllm import vllm_benchmark_config


# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="solutionsteam_vllm_benchmark",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "nightly", "supported", "xlml"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "solutionsteam-vllm-benchmark"

  # Generate a test run id in the format YYYYMMDD-HHmm to group
  # tests together.
  now = datetime.datetime.now()
  test_run_id = now.strftime("%Y%m%d-%H%M")

  test_models = {
      "llama3-8b": {
          "accelerator_specs": [
              (
                  AcceleratorType.GPU,
                  (MachineVersion.A2_HIGHGPU_1G, GpuVersion.A100, 1),
              ),
              (AcceleratorType.TPU, (TpuVersion.V5E, 8)),
          ],
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
        for accelerator_type, accelerator_spec in sweep_model_configs[
            "accelerator_specs"
        ]:
          model_configs = {}
          model_configs["backend"] = backend
          model_configs["model_id"] = model_id
          model_configs["dataset"] = sweep_model_configs["dataset"]
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

          if accelerator_type == AcceleratorType.TPU:
            project = Project.TPU_PROD_ENV_AUTOMATED
            zone = Zone.US_EAST1_C
            tpu_version, tpu_cores = accelerator_spec
            runtime_version = RuntimeVersion.V2_ALPHA_TPUV5_LITE.value
            network = V5_NETWORKS
            subnetwork = V5E_SUBNETWORKS

            vllm_benchmark_config.get_tpu_vllm_gce_config(
                tpu_version=tpu_version,
                tpu_cores=tpu_cores,
                tpu_zone=zone,
                backend=backend,
                runtime_version=runtime_version,
                project=project,
                time_out_in_min=120,
                is_tpu_reserved=True,
                test_name=f"{test_name_prefix}-tpu-nightly-{model}-{backend}",
                test_run_id=test_run_id,
                network=network,
                subnetwork=subnetwork,
                model_configs=model_configs,
            )
          elif accelerator_type == AcceleratorType.GPU:
            project = Project.CLOUD_ML_BENCHMARKING
            zone = Zone.US_WEST4_B
            machine_version, gpu_version, count = accelerator_spec
            image_project = ImageProject.DEEP_LEARNING_PLATFORM_RELEASE
            image_family = ImageFamily.COMMON_CU121_DEBIAN_11
            network = BM_NETWORKS
            subnetwork = A100_BM_SUBNETWORKS

            vllm_benchmark_config.get_gpu_vllm_gce_config(
                machine_version=machine_version,
                image_project=image_project,
                image_family=image_family,
                gpu_version=gpu_version,
                count=count,
                backend=backend,
                project=project,
                gpu_zone=zone,
                time_out_in_min=120,
                test_name=f"{test_name_prefix}-gpu-nightly-{model}-{backend}",
                test_run_id=test_run_id,
                network=network,
                subnetwork=subnetwork,
                model_configs=model_configs,
            ).run()
