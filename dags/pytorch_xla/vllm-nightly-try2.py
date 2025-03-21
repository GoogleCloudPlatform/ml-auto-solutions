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
from xlml.utils import tpu, metric, name_format, ssh


# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None


def get_tpu_vllm_gce_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: Zone,
    backend: str,
    time_out_in_min: int,
    test_name: str,
    test_run_id: str,
    project: Project,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = True,
    num_slices: int = 1,
    model_configs: Dict = {},
):
  # job_gcp_config = gcp_config.GCPConfig(
  #     project_name=project.value,
  #     zone=tpu_zone.value,
  #     dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  # )

  # set_up_cmds = get_vllm_tpu_setup_cmds()
  # model_configs["instance_type"] = tpu_version.value

  # run_model_cmds = get_tpu_vllm_benchmark_cmds(
  #     model_id=model_configs["model_id"],
  #     num_chips=tpu_cores,
  #     test_run_id=test_run_id,
  #     model_configs=model_configs,
  # )

  # job_test_config = test_config.TpuVmTest(
  #     test_config.Tpu(
  #         version=tpu_version,
  #         cores=tpu_cores,
  #         runtime_version=runtime_version,
  #         reserved=is_tpu_reserved,
  #         network=network,
  #         subnetwork=subnetwork,
  #     ),
  #     test_name=test_name,
  #     set_up_cmds=set_up_cmds,
  #     run_model_cmds=run_model_cmds,
  #     timeout=datetime.timedelta(minutes=time_out_in_min),
  #     task_owner=test_owner.RICHARD_L,
  #     num_slices=num_slices,
  #     gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/vllm_benchmark",
  # )

  # job_metric_config = metric_config.MetricConfig(
  #     json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
  #     use_runtime_generated_gcs_folder=True,
  # )

  with TaskGroup(
      group_id=task_test_config.benchmark_id, prefix_group_id=True
  ) as test:
    ssh_keys = ssh.generate_ssh_keys()

    # queued_resource_op, queued_resource_name = tpu.create_queued_resource(
    #       tpu_name,
    #       task_gcp_config,
    #       ssh_keys,
    #       tpu_create_timeout,
    #       task_test_config,
    #   )

    run_model = tpu.ssh_tpu.override(
        task_id="run_vllm_code",
        execution_timeout=3600,
    )(
        "manfei-2025-v6e-4", # queued_resource_name,
        task_test_config.test_script,
        ssh_keys,
        True, # all_workers,
        env={metric_config.SshEnvVars.GCS_OUTPUT.name: output_location},
    )

    run_model

  return test


  # return task.run_queued_resource_test(
  #     task_test_config=job_test_config,
  #     task_gcp_config=job_gcp_config,
  #     task_metric_config=job_metric_config,
  # )



with models.DAG(
    dag_id="pytorch_vllm_nightly_2",
    schedule=SCHEDULED_TIME,
    tags=["solutions_team", "nightly", "supported", "xlml"],
    start_date=datetime.datetime(2024, 1, 19),
    catchup=False,
) as dag:
  test_name_prefix = "pytorch_vllm_nightly_2"

  # Generate a test run id in the format YYYYMMDD-HHmm to group
  # tests together.
  now = datetime.datetime.now()
  test_run_id = now.strftime("%Y%m%d-%H%M")

  test_models = {
      "llama3-8b": {
          "accelerator_specs": [
              # (
              #     AcceleratorType.GPU,
              #     (MachineVersion.A2_HIGHGPU_1G, GpuVersion.A100, 1),
              # ),
              (AcceleratorType.TPU, (TpuVersion.TRILLIUM, 4)),
          ],
          # TODO: Support other backends
          "backend": ["vllm"],
          "model_id": ["meta-llama/Meta-Llama-3.1-8B"],
          "dataset": "ShareGPT_V3_unfiltered_cleaned_split.json",
          "request_rates": "1,2,4,8,16,32",
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
            project = Project.CLOUD_ML_BENCHMARKING
            zone = Zone.US_EAST5_B
            tpu_version, tpu_cores = accelerator_spec
            # tpu_version = TpuVersion.TRILLIUM
            # tpu_cores = 4
            runtime_version = RuntimeVersion.V2_ALPHA_TPUV6.value
            network = V6E_GCE_NETWORK
            subnetwork = V6E_GCE_SUBNETWORK

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
