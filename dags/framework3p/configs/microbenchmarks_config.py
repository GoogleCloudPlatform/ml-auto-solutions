from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket, test_owner
import datetime
import dags.vm_resource as resource

    
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
        "pip install --upgrade clu tensorflow tensorflow-datasets ",
        "pip install jsonlines ",
        "JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true ",
        # TODO(qinyiyan): Clone maxtext from google repo when code is merged.
        "git clone https://github.com/qinyiyan/maxtext.git /tmp/maxtext "
    )

    # List of benchmark scripts to run
    benchmark_scripts = ["all_reduce", "all_gather"]

    metrics_report = "/tmp/microbenchmark/outputs/metrics_report.jsonl"

    # Initial commands
    run_model_cmds = (
        # Create the output directory
        "mkdir -p /tmp/microbenchmark/outputs",
        
        # Remove any existing metrics report
        (
            f"if [ -f {metrics_report} ]; then "
            f"rm -rf {metrics_report}; "
            "fi"
        ),
    )

    # Loop through the benchmark scripts and create commands dynamically
    for script in benchmark_scripts:
        run_model_cmds += (
            # Run the benchmark script (either all_reduce or all_gather)
            f"python3 /tmp/maxtext/microbenchmarks/{script}.py "
            "--metrics_jsonl_dir=/tmp/microbenchmark/outputs ",
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
    tpu_version: resource.TpuVersion,
    tpu_cores: int,
    tpu_zone: resource.Zone,
    time_out_in_min: int,
    test_name: str,
    docker_image: str,
    test_owner: str,
    cluster: resource.XpkClusterConfig,
    project: resource.Project,
    num_slices: int = 1,
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.XLML_DATASET,
    dataset_project: str = resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    composer_project: str = resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value,
) -> task.XpkTask:
    job_gcp_config = gcp_config.GCPConfig(
      project_name=project.value,
      zone=tpu_zone.value,
      dataset_name=dataset_name,
      dataset_project=dataset_project,
      composer_project=composer_project,
    )

    set_up_cmds = (
        "pip install --upgrade pip",
        (
            "pip install jax[tpu] -f"
            " https://storage.googleapis.com/jax-releases/libtpu_releases.html"
        ),
        "pip install --upgrade clu tensorflow tensorflow-datasets ",
        "pip install jsonlines ",
        "JAX_PLATFORMS=tpu,cpu ENABLE_PJRT_COMPATIBILITY=true ",
        # TODO(qinyiyan): clone from Google's maxtext when code is merged.
        "git clone https://github.com/qinyiyan/maxtext.git /tmp/maxtext ",
    )

    benchmark_scripts = ["all_reduce", "all_gather"]

    metrics_report = "/tmp/microbenchmark/outputs/metrics_report.jsonl"

    # Initial commands
    run_model_cmds = set_up_cmds + (
        # Create the output directory
        "mkdir -p /tmp/microbenchmark/outputs",
        
        # Remove any existing metrics report
        (
            f"if [ -f {metrics_report} ]; then "
            f"rm -rf {metrics_report}; "
            "fi"
        ),
    )

    # Loop through the benchmark scripts and create commands dynamically
    for script in benchmark_scripts:
        run_model_cmds += (
            # Run the benchmark script (either all_reduce or all_gather)
            f"python3 /tmp/maxtext/microbenchmarks/{script}.py "
            "--metrics_jsonl_dir=/tmp/microbenchmark/outputs ",
        )

    # Check if the metrics report exists, and if so, upload it to GCS
    run_model_cmds += (
        f"if [ -f {metrics_report} ]; then "
        f"gsutil cp {metrics_report} {metric_config.SshEnvVars.GCS_OUTPUT.value} ; "
        "fi ",
    )


    job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
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

