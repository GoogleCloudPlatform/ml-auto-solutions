# Copyright 2025 Google LLC
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
import tempfile
import subprocess
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


from dags.map_reproducibility.utils.common_utils import (
    namespace_cmds,
    internal_wait_for_jobs_cmds,
    cleanup_cmds,
    helm_apply_cmds_internal_run,
    get_internal_pre_workload_cmds,
    get_internal_pre_workload_job_name,
    get_bq_writer_path,
    get_cluster,
    calculate_maxtext_metrics,
    copy_bucket_cmds_maxtext,
    get_job_gcs_bucket_folder,
    parse_internal_config_filename,
    parse_internal_config_content,
    get_patheon_job_link,
    find_xprof_gcs_path,
    get_skip_steps_for_metrics_calculation,
    copy_bucket_cmds_nemo,
    get_accelerator_type,
    get_nemo_metrics_cmds,
    get_nemo_metrics,
)

from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from dags.map_reproducibility.utils.constants import Optimizer, KUEUE_NAME, NUM_STEPS

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PROJECT = "supercomputer-testing"
BUCKET_NAME = "regression-testing-xlml"


@dataclass
class WorkloadResult:
  """Container for workload execution results."""

  mfu: float
  step_time: float
  gcs_bucket: str
  success: bool
  error_message: Optional[str] = None


def get_values_file_path(
    base_recipe_repo_root: str,
    config_yaml_name: str,
    hypercomputer: str,
    framework: str,
) -> str:
  """Determine the appropriate values file path.

  Args:
      base_recipe_repo_root: Root directory of the recipe repository
      config_yaml_name: Name of the config YAML file
      hypercomputer: Type of hypercomputer
      framework: Framework name

  Returns:
      Path to the values file
  """
  # Default values file based on hypercomputer and framework
  values_name = f"{hypercomputer}_{framework}_values"
  values_file_path = f"{base_recipe_repo_root}/values/{values_name}.yaml"

  # Check for model-specific values file
  model_specific_values_file_path = (
      f"{base_recipe_repo_root}/values/{config_yaml_name}_values.yaml"
  )
  if os.path.exists(model_specific_values_file_path):
    # Use model-specific values file
    values_file_path = model_specific_values_file_path

  logger.info(f"Using values file: {values_file_path}")
  return values_file_path


def execute_workload_commands(commands: list, cwd: str) -> Tuple[bool, list]:
  """Execute shell commands and capture their outputs.

  Args:
      commands: List of shell commands to execute
      cwd: Current working directory

  Returns:
      Tuple of (success, list of command results)
  """
  logger.info(f"Executing commands: {commands} in directory: {cwd}")

  # Join commands with semicolons for sequential execution
  combined_command = ";".join(commands)

  # Run the combined command
  process = subprocess.Popen(
      ["bash", "-c", combined_command],
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      cwd=cwd,
  )

  # Capture output
  stdout, stderr = process.communicate()
  exit_code = process.returncode

  # Create result for the combined execution
  command_results = [{
      "command": combined_command,
      "stdout": stdout,
      "stderr": stderr,
      "output": stdout + ("\n\nSTDERR:\n" + stderr if stderr else ""),
      "exit_code": exit_code,
  }]

  # Log results
  if stdout:
    logger.info(f"Stdout for combined commands:\n{stdout}")
  if stderr:
    logger.warning(f"Stderr for combined commands:\n{stderr}")
  if exit_code != 0:
    logger.error("Command execution failed")

  return exit_code == 0, command_results


def sample_job_configure_project_and_cluster(cluster: str, cluster_region: str):
  set_project_command = (
      f"gcloud config set project {PROJECT}",
      "gcloud container clusters get-credentials "
      f"{cluster} --region {cluster_region}",
  )
  return set_project_command


def sample_workload_gcs_to_cns_cmds(log_file_in_gcs, output_file=None):
  # This function only works for glinux or cloudtop because it is using fileutil_bs
  # If output_file is not provided, use the same name as the input file
  log_file_in_gcs = log_file_in_gcs.removeprefix("gs://")
  if not output_file:
    output_file = os.path.basename(log_file_in_gcs)
  print(f"output_file name is: {output_file}")

  cmds = (
      f"LOG_FILE_IN_GCS={log_file_in_gcs} ",
      f"filename={output_file} ",
      "CNS_PATH=/cns/pi-d/home/${USER}/tensorboard/multislice ",
      "/google/data/ro/projects/cloud/bigstore/mpm/fileutil_bs/stable/bin/fileutil_bs cp /bigstore/${LOG_FILE_IN_GCS} ${CNS_PATH}/${filename} ",
      "echo file to put into xprof: ${CNS_PATH}/${filename}",
  )
  return cmds


def write_run_results(
    config: Any,
    result: WorkloadResult,
    docker_image: str,
    bq_writer_repo_root: str,
    job_name: str,
    test_run: bool,
) -> None:
  """Write run results to the benchmark database.

  Args:
      config: Configuration object
      result: WorkloadResult object
      docker_image: Docker image name
      bq_writer_repo_root: Path to BigQuery writer repository
      job_name: Name of the job
      test_run: Whether this is a test run
  """
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
      median_step_time=result.step_time,
      e2e_time=result.step_time * NUM_STEPS,
      number_of_steps=NUM_STEPS,
      mfu=result.mfu,
      tokens_per_second=1,  # Consider calculating this properly
      writer_path=bq_writer_repo_root,
      run_type="sample_perf_regression",
      topology="",
      comment="sample benchmarking run",
      is_test=test_run,
      logs_profile=result.gcs_bucket,
      workload_others=str(config),
      experiment_id=job_name,
  )


def run_internal_sample_aotc_workload(
    relative_config_yaml_path: str,
    base_recipe_repo_root: str,
    timeout: int,
    image_version: str,
    sample_run_bucket_name: str,
) -> Dict[str, Any]:
  """Run the internal sample AOTC workload.

  Args:
      relative_config_yaml_path: Relative path to config YAML
      base_recipe_repo_root: Root directory of the recipe repository
      test_run: Whether this is a test run
      timeout: Timeout in seconds
      image_version: Docker image version

  Returns:
      Dictionary with results
  """
  # Parse config from filename
  config_yaml_name = relative_config_yaml_path.rsplit("/", maxsplit=1)[
      -1
  ].replace(".yaml", "")
  config = parse_internal_config_filename(config_yaml_name)

  # Get derived configuration
  cluster, cluster_region = get_cluster(config.HYPERCOMPUTER)
  docker_image = image_version

  # Locate values file
  values_file_path = get_values_file_path(
      base_recipe_repo_root,
      config_yaml_name,
      config.HYPERCOMPUTER,
      config.FRAMEWORK,
  )

  # Locate config yaml
  full_config_yaml_path = f"{base_recipe_repo_root}/{relative_config_yaml_path}"
  logger.info(f"Config YAML path: {full_config_yaml_path}")

  # Parse the config content now that we have the file path
  config = parse_internal_config_content(full_config_yaml_path, config=config)
  job_name = get_internal_pre_workload_job_name(
      model_id=config.MODEL_ID,
      precision=config.PRECISION,
      num_gpus=config.NUM_GPUS,
      framework=config.FRAMEWORK,
      cluster=config.HYPERCOMPUTER,
      is_sample_run=True,
  )
  pantheon_link = get_patheon_job_link(
      region=cluster_region, cluster_name=cluster, job_name=job_name
  )

  # Adjust timeout for the container
  container_timeout = int(timeout) - 4
  logger.info(f"Container timeout: {container_timeout}")

  with tempfile.TemporaryDirectory() as tmpdir:
    # Prepare commands
    commands = (
        sample_job_configure_project_and_cluster(cluster, cluster_region)
        + namespace_cmds()
        + get_internal_pre_workload_cmds(job_name=job_name)
        + helm_apply_cmds_internal_run(
            config.FRAMEWORK,
            config.HYPERCOMPUTER,
            full_config_yaml_path,
            base_recipe_repo_root,
            values_file_path,
            docker_image,
            cluster_name=cluster,
            kueue_name=None,
            additional_cmds=f" --set workload.gpus={config.NUM_GPUS} ",
            bucket_name=sample_run_bucket_name,
        )
        + internal_wait_for_jobs_cmds(timeout=container_timeout)
        + copy_bucket_cmds_maxtext(tmpdir, bucket_name=sample_run_bucket_name)
        + cleanup_cmds()
    )

    # Execute commands
    success, error_message = execute_workload_commands(commands, tmpdir)

    # Process results
    if success:
      bq_writer_repo_root = get_bq_writer_path(tmpdir)
      log_location = os.path.join(tmpdir, "tflog/metrics")

      comment = "sample benchmarking run"
      gcs_bucket = get_job_gcs_bucket_folder(
          job_name, bucket_name=sample_run_bucket_name
      )
      print(f"GCS bucket is {gcs_bucket}")
      logs_profile = None

      if hasattr(config, "profiler"):
        logs_profile = find_xprof_gcs_path(gcs_bucket)
        if not logs_profile:
          logger.error(f"No xprof file found in {gcs_bucket}")
        else:
          print(f"logs_profile is {logs_profile}")
          profiler_cmds = sample_workload_gcs_to_cns_cmds(logs_profile)
          profile_success, profiler_error_message = execute_workload_commands(
              profiler_cmds, tmpdir
          )
          if not profile_success:
            logger.error(
                f"Profile command failed with error: {profiler_error_message}"
            )

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
          run_type="sample_helm_workload",
          topology="",
          comment=comment,
          is_test=True,
          logs_profile=logs_profile,
          gcs_metrics_bucket=gcs_bucket,
          workload_others=str(config),
          experiment_id=job_name,
      )


def run_internal_sample_aotc_workload_nemo(
    relative_config_yaml_path: str,
    base_recipe_repo_root: str,
    timeout: int,
    image_version: str,
    sample_run_bucket_name: str,
) -> Dict[str, Any]:
  """Run the internal sample AOTC workload for NeMo framework.

  Args:
      relative_config_yaml_path: Relative path to config YAML
      base_recipe_repo_root: Root directory of the recipe repository
      timeout: Timeout in seconds
      image_version: Docker image version
      sample_run_bucket_name: GCS bucket for storing run data
      two_node: Whether to run with 2-node configuration

  Returns:
      Dictionary with results
  """
  # Parse config from filename
  config_yaml_name = relative_config_yaml_path.rsplit("/", maxsplit=1)[
      -1
  ].replace(".yaml", "")
  config = parse_internal_config_filename(config_yaml_name)

  # Get derived configuration
  cluster, cluster_region = get_cluster(config.HYPERCOMPUTER)
  docker_image = image_version

  # Locate values file
  values_file_path = get_values_file_path(
      base_recipe_repo_root,
      config_yaml_name,
      config.HYPERCOMPUTER,
      config.FRAMEWORK,
  )

  # Locate config yaml
  full_config_yaml_path = f"{base_recipe_repo_root}/{relative_config_yaml_path}"
  logger.info(f"Config YAML path: {full_config_yaml_path}")
  logger.info(f"base_recipe_repo_root is: {base_recipe_repo_root}")

  # Parse the config content now that we have the file path
  config = parse_internal_config_content(full_config_yaml_path, config=config)
  logger.info(f"sequence length is: {config.model.data.seq_length}")

  job_name = get_internal_pre_workload_job_name(
      model_id=config.MODEL_ID,
      precision=config.PRECISION,
      num_gpus=config.NUM_GPUS,
      framework=config.FRAMEWORK,
      cluster=config.HYPERCOMPUTER,
      is_sample_run=True,
  )
  accelerator_type = get_accelerator_type(config.HYPERCOMPUTER)
  pantheon_link = get_patheon_job_link(
      region=cluster_region, cluster_name=cluster, job_name=job_name
  )

  # Adjust timeout for the container
  container_timeout = int(timeout) - 4
  logger.info(f"Container timeout: {container_timeout}")
  logger.info(f"global_batch_size is: {config.model.global_batch_size}")

  with tempfile.TemporaryDirectory() as tmpdir:
    # Prepare commands
    commands = (
        sample_job_configure_project_and_cluster(cluster, cluster_region)
        + namespace_cmds()
        + get_internal_pre_workload_cmds(job_name=job_name)
        + helm_apply_cmds_internal_run(
            config.FRAMEWORK,
            config.HYPERCOMPUTER,
            full_config_yaml_path,
            base_recipe_repo_root,
            values_file_path,
            docker_image,
            cluster_name=cluster,
            kueue_name=None,
            additional_cmds=f" --set workload.gpus={config.NUM_GPUS} ",
            bucket_name=sample_run_bucket_name,
        )
        + internal_wait_for_jobs_cmds(timeout=container_timeout)
        + copy_bucket_cmds_nemo(
            base_recipe_repo_root,
            hypercomputer=config.HYPERCOMPUTER,
            bucket_name=sample_run_bucket_name,
        )
        + get_nemo_metrics_cmds(
            batch_size=config.model.global_batch_size,
            num_accelerators=config.NUM_GPUS,
            precision=config.PRECISION,
            model_id=config.MODEL_ID,
            accelertator_type=accelerator_type,
            temdir=tmpdir,
            two_node=False,
            start_step=2,
            end_step=int(NUM_STEPS) - 3,
        )
        + cleanup_cmds()
    )

    # Execute commands
    success, error_message = execute_workload_commands(commands, tmpdir)

    # Process results
    if success:
      bq_writer_repo_root = get_bq_writer_path(tmpdir)

      comment = "sample benchmarking run"
      gcs_bucket = get_job_gcs_bucket_folder(
          job_name,
          bucket_name=sample_run_bucket_name,
          framework=config.FRAMEWORK,
      )
      print(f"GCS bucket is {gcs_bucket}")
      logs_profile = None

      # calculate mfu based on the config
      mfu, step_time = get_nemo_metrics(tmpdir)
      print(f"mfu: {mfu}")
      print(f"step_time: {step_time}")

      write_run(
          model_id=config.HELM_NAME_MODEL_ID,
          hardware_id=config.HYPERCOMPUTER,
          software_id=config.SOFTWARE_ID,
          number_of_nodes=config.NUM_GPUS / 8,
          number_of_chips=config.NUM_GPUS,
          container_image_name=docker_image,
          global_batch_size=config.model.global_batch_size,
          precision=config.PRECISION,
          optimizer=Optimizer.ADAM,
          seq_length=config.model.data.seq_length,
          median_step_time=step_time,
          e2e_time=step_time * NUM_STEPS,
          number_of_steps=NUM_STEPS,
          mfu=mfu,
          tokens_per_second=1,
          writer_path=bq_writer_repo_root,
          run_type="sample_helm_workload",
          topology="",
          comment=comment,
          is_test=True,
          logs_profile=logs_profile,
          gcs_metrics_bucket=gcs_bucket,
          workload_others=str(config),
          experiment_id=job_name,
      )
