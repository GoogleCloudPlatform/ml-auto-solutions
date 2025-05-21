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

"""Workload functions for AOTC reproducibility benchmarks."""

import os
import logging
import tempfile
from typing import Dict, Any

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.operators.python import get_current_context

from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from dags.map_reproducibility.utils.common_utils import (
    calculate_maxtext_metrics,
    calculate_metrics,
    cleanup_all_runs_cmds,
    cleanup_cmds,
    clone_internal_recipes_gob,
    clone_recipes_gob,
    configure_project_and_cluster,
    copy_bucket_cmds_maxtext,
    copy_bucket_cmds_workload,
    extract_batch_size_and_seq_len,
    get_accelerator_type,
    get_bq_writer_path,
    get_bq_writer_repo,
    get_cluster,
    get_gpu_recipe_cmd,
    get_internal_pre_workload_cmds,
    get_internal_pre_workload_job_name,
    get_internal_recipe_repo_path,
    get_job_gcs_bucket_folder,
    get_metrics_cmd,
    get_patheon_job_link,
    get_recipe_repo_path,
    get_skip_steps_for_metrics_calculation,
    get_values_file_path,
    git_cookie_authdaemon,
    helm_apply_cmds_internal_run,
    helm_apply_cmds_workload,
    install_helm_cmds,
    internal_wait_for_jobs_cmds,
    namespace_cmds,
    parse_internal_config_content,
    parse_internal_config_filename,
    wait_for_jobsets_cmds,
    get_internal_run_type_and_comment,
)
from dags.map_reproducibility.utils.sample_workload_utils import handle_profiler, assemble_sample_united_workload_commands, execute_workload_commands
from dags.map_reproducibility.utils.constants import Optimizer, KUEUE_NAME, NUM_STEPS, BUCKET_NAME

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@task
def run_internal_aotc_workload(
    relative_config_yaml_path,
    test_run=False,
    backfill=False,
    timeout=None,
    image_version=None,
):
  """Runs the AOTC workload benchmark.

  Args:
    relative_config_yaml_path: Path to the config YAML relative to the repo root
  """
  # Get the current context to access DAG ID
  context = get_current_context()
  dag_id = context["dag"].dag_id

  # Parse config from filename
  config_yaml_name = relative_config_yaml_path.rsplit("/", maxsplit=1)[
      -1
  ].replace(".yaml", "")
  config = parse_internal_config_filename(config_yaml_name)

  # Get derived configuration
  cluster, cluster_region = get_cluster(config.HYPERCOMPUTER)
  docker_image = image_version
  values_name = f"{config.HYPERCOMPUTER}_{config.FRAMEWORK}_values"

  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                git_cookie_authdaemon()
                + clone_recipes_gob()
                + clone_internal_recipes_gob()
                + get_bq_writer_repo()
            ),
        ],
        cwd=tmpdir,
    )

    recipe_repo_root = get_recipe_repo_path(tmpdir)
    bq_writer_repo_root = get_bq_writer_path(tmpdir)

    # Update paths now that we have the repo paths
    internal_recipe_repo_root = get_internal_recipe_repo_path(tmpdir)
    values_file_path = f"{internal_recipe_repo_root}/values/{values_name}.yaml"
    model_specific_values_file_path = (
        f"{internal_recipe_repo_root}/values/{config_yaml_name}_values.yaml"
    )
    if os.path.exists(model_specific_values_file_path):
      # Use model-specific values file
      values_file_path = model_specific_values_file_path
      print(
          f"Using model-specific values file: {model_specific_values_file_path}"
      )
    else:
      print(
          f"Model-specific values file not found, using general values file: {values_file_path}"
      )

    full_config_yaml_path = (
        f"{internal_recipe_repo_root}/{relative_config_yaml_path}"
    )
    print(f"values_file_path is {values_file_path}")
    print(f"full_config_yaml_path is {full_config_yaml_path}")

    # Parse the config content now that we have the file path
    config = parse_internal_config_content(full_config_yaml_path, config=config)
    job_name = get_internal_pre_workload_job_name(
        model_id=config.MODEL_ID,
        precision=config.PRECISION,
        num_gpus=config.NUM_GPUS,
        framework=config.FRAMEWORK,
        cluster=config.HYPERCOMPUTER,
    )
    pantheon_link = get_patheon_job_link(
        region=cluster_region, cluster_name=cluster, job_name=job_name
    )

    # Print DAG ID with job name
    print(f"Running job '{job_name}' in DAG '{dag_id}'")

    container_timeout = int(timeout) - 4
    print(f"container timeout is {container_timeout}")
    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(cluster, cluster_region)
                + get_gpu_recipe_cmd(
                    config.HYPERCOMPUTER,
                    config.MODEL_ID,
                    config.FRAMEWORK,
                    recipe_repo_root,
                )
                + install_helm_cmds()
                + namespace_cmds()
                + get_internal_pre_workload_cmds(job_name=job_name)
                + helm_apply_cmds_internal_run(
                    config.FRAMEWORK,
                    config.HYPERCOMPUTER,
                    full_config_yaml_path,
                    internal_recipe_repo_root,
                    values_file_path,
                    docker_image,
                    cluster_name=cluster,
                    kueue_name=None,  # not enabled until kueue-tas is fixed
                    additional_cmds=f" --set workload.gpus={config.NUM_GPUS} ",
                )
                + internal_wait_for_jobs_cmds(timeout=container_timeout)
                + copy_bucket_cmds_maxtext(tmpdir)
                + cleanup_cmds()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    log_location = os.path.join(tmpdir, "tflog/metrics")

    comment = (
        "internal recipes regression tests"
        if not backfill
        else "internal recipes regression tests backfill"
    )
    is_db_test_run = False if backfill else test_run
    gcs_bucket = get_job_gcs_bucket_folder(job_name)
    print(f"GCS bucket is {gcs_bucket}")

    # calculate mfu based on the config
    skip_first_n_steps = get_skip_steps_for_metrics_calculation(config)
    mfu, step_time = calculate_maxtext_metrics(
        log_location,
        config.HYPERCOMPUTER,
        skip_first=skip_first_n_steps,
    )
    print(f"mfu: {mfu}")
    print(f"step_time: {step_time}")

    write_run(
        model_id=config.HELM_NAME_MODEL_ID,
        hardware_id=config.HYPERCOMPUTER,
        software_id=config.SOFTWARE_ID,
        number_of_nodes=config.NUM_GPUS / 8,
        number_of_chips=config.NUM_GPUS,
        container_image_name=docker_image,
        global_batch_size=config.per_device_batch_size * config.NUM_GPUS,
        precision=config.PRECISION,
        optimizer=Optimizer.ADAM,
        seq_length=config.max_target_length,
        median_step_time=step_time,
        e2e_time=step_time * NUM_STEPS,
        number_of_steps=NUM_STEPS,
        mfu=mfu,
        tokens_per_second=1,
        writer_path=bq_writer_repo_root,
        run_type="internal_perf_regression",
        topology="",
        comment=comment,
        is_test=is_db_test_run,
        gcs_metrics_bucket=gcs_bucket,
        workload_others=str(config),
        experiment_id=job_name,
    )


@task
def cleanup_cml_workloads(cluster, cluster_region):
  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()
    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(cleanup_all_runs_cmds(cluster, cluster_region)),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


def assemble_dag_united_workload_commands(
    config,
    cluster,
    region,
    job_name,
    launcher_path,
    helm_repo_root,
    full_config_path,
    values_file_path,
    bucket_name,
    timeout,
    tmpdir,
):
  metrics_cmd = get_metrics_cmd(
      config,
      get_accelerator_type(config.HYPERCOMPUTER),
      tmpdir,
      start_step=2,
      end_step=NUM_STEPS - 3,
  )
  return (
      configure_project_and_cluster(cluster, region)
      + install_helm_cmds()
      + namespace_cmds()
      + get_internal_pre_workload_cmds(job_name)
      + helm_apply_cmds_workload(
          config.FRAMEWORK,
          config.HYPERCOMPUTER,
          full_config_path,
          helm_repo_root,
          workload_launcher=launcher_path,
          kueue_name=None,
          additional_cmds=f" --set workload.gpus={config.NUM_GPUS} ",
          bucket_name=bucket_name,
          values_file_path=values_file_path,
      )
      + wait_for_jobsets_cmds(timeout)
      + copy_bucket_cmds_workload(
          recipe_repo_root=helm_repo_root,
          tmpdir=tmpdir,
          framework=config.FRAMEWORK,
          bucket_name=bucket_name,
      )
      + metrics_cmd
      + cleanup_cmds()
  )


def run_internal_united_workload(
    relative_config_yaml_path: str,
    base_recipe_repo_root: str,
    base_helm_repo_root: str,
    timeout: int,
    image_version: str,
    gcs_bucket_name: str,
    workload_launcher: str,
    is_dag_run: bool = False,
    backfill: bool = False,
    test_run: bool = True,
) -> Dict[str, Any]:
  """Run a sample or DAG internal workload (NeMo or MaxText)."""

  with tempfile.TemporaryDirectory() as tmpdir:
    # Clone repos if paths are not provided
    if not base_helm_repo_root and not base_recipe_repo_root:
      hook = SubprocessHook()
      hook.run_command(
          [
              "bash",
              "-c",
              ";".join(
                  git_cookie_authdaemon()
                  + clone_recipes_gob()
                  + clone_internal_recipes_gob()
                  + get_bq_writer_repo()
              ),
          ],
          cwd=tmpdir,
      )
      base_helm_repo_root = get_recipe_repo_path(tmpdir)
      base_recipe_repo_root = get_internal_recipe_repo_path(tmpdir)

    print("start workload")
    config_yaml_name = os.path.basename(relative_config_yaml_path).replace(
        ".yaml", ""
    )
    config = parse_internal_config_filename(config_yaml_name)

    full_config_path = os.path.join(
        base_recipe_repo_root, relative_config_yaml_path
    )
    config = parse_internal_config_content(full_config_path, config=config)

    cluster, region = get_cluster(config.HYPERCOMPUTER)
    values_file_path = get_values_file_path(
        base_recipe_repo_root,
        config_yaml_name,
        config.HYPERCOMPUTER,
        config.FRAMEWORK,
    )

    job_name = get_internal_pre_workload_job_name(
        model_id=config.MODEL_ID,
        precision=config.PRECISION,
        num_gpus=config.NUM_GPUS,
        framework=config.FRAMEWORK,
        cluster=config.HYPERCOMPUTER,
        is_sample_run=not is_dag_run,
    )

    get_patheon_job_link(
        region=region, cluster_name=cluster, job_name=job_name, is_jobset=True
    )

    launcher_path = os.path.join(
        base_helm_repo_root, f"src/launchers/{workload_launcher}"
    )

    global_batch_size, seq_length = extract_batch_size_and_seq_len(config)
    container_timeout = f"{timeout - 4}m"

    logger.info(f"Container timeout: {container_timeout}")
    logger.info(f"Global batch size: {global_batch_size}")

    assemble_fn = (
        assemble_dag_united_workload_commands
        if is_dag_run
        else assemble_sample_united_workload_commands
    )
    commands = assemble_fn(
        config,
        cluster,
        region,
        job_name,
        launcher_path,
        base_helm_repo_root,
        full_config_path,
        values_file_path,
        gcs_bucket_name,
        container_timeout,
        tmpdir,
    )

    success, error = execute_workload_commands(commands, cwd=tmpdir)
    if not success:
      return {"success": False, "error": error}

    gcs_bucket = get_job_gcs_bucket_folder(
        job_name,
        bucket_name=gcs_bucket_name,
        framework=config.FRAMEWORK,
        gcs_experiment_folder_name="maxtext-experiments",
    )
    logger.info(f"GCS bucket: {gcs_bucket}")

    mfu, average_step_time = calculate_metrics(config, tmpdir)
    logger.info(f"MFU: {mfu}, Step time: {average_step_time}")

    logs_profile = handle_profiler(
        config, gcs_bucket, tmpdir, is_sample_run=not is_dag_run
    )
    run_type, comment = get_internal_run_type_and_comment(is_dag_run, backfill)
    is_db_test_run = False if backfill else test_run

    write_run(
        model_id=config.HELM_NAME_MODEL_ID,
        hardware_id=config.HYPERCOMPUTER,
        software_id=config.SOFTWARE_ID,
        number_of_nodes=config.NUM_GPUS / 8,
        number_of_chips=config.NUM_GPUS,
        container_image_name=image_version,
        global_batch_size=global_batch_size,
        precision=config.PRECISION,
        optimizer=Optimizer.ADAM,
        seq_length=seq_length,
        median_step_time=average_step_time,
        e2e_time=average_step_time * NUM_STEPS,
        number_of_steps=NUM_STEPS,
        mfu=mfu,
        tokens_per_second=1,
        writer_path=get_bq_writer_path(tmpdir),
        run_type=run_type,
        comment=comment,
        is_test=is_db_test_run,
        logs_profile=logs_profile,
        gcs_metrics_bucket=gcs_bucket,
        workload_others=str(config),
        experiment_id=job_name,
    )

    return {"success": True, "job_name": job_name, "metrics_bucket": gcs_bucket}


@task
def run_internal_dag_united_workload(
    relative_config_yaml_path: str,
    test_run: bool,
    backfill: bool,
    timeout: int,
    image_version: str,
    workload_launcher: str,
) -> Dict[str, Any]:
  """Airflow Task wrapper."""
  return run_internal_united_workload(
      relative_config_yaml_path=relative_config_yaml_path,
      base_recipe_repo_root=None,
      base_helm_repo_root=None,
      timeout=timeout,
      image_version=image_version,
      gcs_bucket_name=BUCKET_NAME,
      workload_launcher=workload_launcher,
      is_dag_run=True,
      backfill=backfill,
      test_run=test_run,
  )
