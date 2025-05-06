from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket
from dags.common import test_owner
import datetime
import dags.common.vm_resource as resource


def get_microbenchmark_config(
    tpu_version: resource.TpuVersion,
    tpu_cores: int,
    tpu_zone: resource.Zone,
    time_out_in_min: int,
    runtime_version: resource.RuntimeVersion,
    project: resource.Project,
    network: str = "default",
    subnetwork: str = "default",
    extraFlags: str = "",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project.value,
      zone=tpu_zone.value,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = (
      "pip install --upgrade pip",
      (
          "pip install jax[tpu] -f"
          " https://storage.googleapis.com/jax-releases/libtpu_releases.html"
      ),
      "JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true ",
  )

  benchmark_config = f"xlml_v{tpu_version.value}_{tpu_cores}.yaml"
  metrics_report = "/tmp/microbenchmarks/outputs/metrics_report.jsonl"

  # Initial commands
  run_model_cmds = (
      # Create the output directory
      "mkdir -p /tmp/microbenchmarks/outputs ",
      # Remove any existing metrics report
      (f"if [ -f {metrics_report} ]; then " f"rm -rf {metrics_report}; " "fi"),
  )

  # Run the benchmark tests.
  run_model_cmds += (
      " rm -rf accelerator-microbenchmarks ",
      "git clone https://github.com/AI-Hypercomputer/accelerator-microbenchmarks.git  ",
      "cd accelerator-microbenchmarks ",
      "pip install -r requirements.txt ",
      # Run the benchmark script
      f"python3 src/run_benchmark.py " f"--config=configs/{benchmark_config} ",
  )

  # Check if the metrics report exists, and if so, upload it to GCS
  run_model_cmds += (
      f"if [ -f {metrics_report} ]; then "
      f"gsutil cp {metrics_report} {metric_config.SshEnvVars.GCS_OUTPUT.value}; "
      "fi",
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version.value,
          network=network,
          subnetwork=subnetwork,
          reserved=True,
      ),
      test_name="framework-microbenchmark",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.QINY_Y,
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metrics_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


def get_microbenchmark_xpk_config(
    time_out_in_min: int,
    test_name: str,
    docker_image: str,
    test_owner: str,
    cluster: resource.XpkClusterConfig,
    num_slices: int = 1,
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.XLML_DATASET,
    dataset_project: str = resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    composer_project: str = resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value,
) -> task.XpkTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=dataset_name,
      dataset_project=dataset_project,
      composer_project=composer_project,
  )

  benchmark_config = (
      f"xlml_v{cluster.device_version.value}_{cluster.core_count}.yaml"
  )
  metrics_report = "/tmp/microbenchmarks/outputs/metrics_report.jsonl"

  # Initial commands
  run_model_cmds = (
      # Create the output directory
      "mkdir -p /tmp/microbenchmarks/outputs ",
      # Remove any existing metrics report
      (f"if [ -f {metrics_report} ]; then " f"rm -rf {metrics_report}; " "fi"),
  )

  # Run the benchmark tests.
  run_model_cmds += (
      "cd /app/accelerator-microbenchmarks ",
      # Run the benchmark script
      f"python3 src/run_benchmark.py " f"--config=configs/{benchmark_config} ",
  )

  # Check if the metrics report exists, and if so, upload it to GCS
  run_model_cmds += (
      f"if [ -f {metrics_report} ]; then "
      f"gsutil cp {metrics_report} {metric_config.SshEnvVars.GCS_OUTPUT.value} ; "
      "fi ",
  )

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=cluster.device_version,
          cores=cluster.core_count,
      ),
      test_name=test_name,
      set_up_cmds=None,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner,
      num_slices=num_slices,
      cluster_name=cluster.name,
      docker_image=docker_image,
  )
  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metrics_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )
  return task.XpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
