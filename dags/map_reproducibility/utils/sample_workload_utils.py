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
  """Execute each command individually while preserving bash context.

  Args:
      commands: List of shell commands to execute
      cwd: Current working directory

  Returns:
      Tuple of (success, list of command results)
  """
  logger.info(
      f"Executing commands sequentially: {commands} in directory: {cwd}"
  )

  command_results = []

  # Start a bash process that we'll keep alive
  process = subprocess.Popen(
      ["bash"],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
      cwd=cwd,
  )

  try:
    # Execute setup commands to enable command tracing
    process.stdin.write("set -e\n")  # Exit on first error
    process.stdin.write(
        "cd " + cwd + "\n"
    )  # Ensure we're in the right directory
    process.stdin.flush()

    for i, cmd in enumerate(commands):
      logger.info(f"Executing command {i+1}: {cmd}")

      # Create unique markers for this command
      cmd_id = f"CMD_{i}"

      # Script to capture both stdout and stderr separately
      capture_script = f"""
            # Start marker
            echo '{cmd_id}_START'
            
            # Create temporary files for stdout and stderr
            STDOUT_FILE=$(mktemp)
            STDERR_FILE=$(mktemp)
            
            # Execute the command, capturing stdout and stderr
            {{ {cmd} > $STDOUT_FILE 2> $STDERR_FILE; }}
            CMD_EXIT_CODE=$?
            
            # Output stdout with marker
            echo '{cmd_id}_STDOUT_BEGIN'
            cat $STDOUT_FILE
            echo '{cmd_id}_STDOUT_END'
            
            # Output stderr with marker
            echo '{cmd_id}_STDERR_BEGIN'
            cat $STDERR_FILE
            echo '{cmd_id}_STDERR_END'
            
            # Output exit code with marker
            echo '{cmd_id}_EXIT_'$CMD_EXIT_CODE
            
            # Clean up temp files
            rm -f $STDOUT_FILE $STDERR_FILE
            """

      # Write the capture script
      process.stdin.write(capture_script)
      process.stdin.flush()

      # Process output with state machine
      current_state = "WAITING_FOR_START"
      stdout_lines = []
      stderr_lines = []
      exit_code = None

      while True:
        line = process.stdout.readline().rstrip("\n")

        if not line:
          continue

        if line == f"{cmd_id}_START":
          current_state = "STARTED"
        elif line == f"{cmd_id}_STDOUT_BEGIN":
          current_state = "READING_STDOUT"
        elif line == f"{cmd_id}_STDOUT_END":
          current_state = "STDOUT_COMPLETE"
        elif line == f"{cmd_id}_STDERR_BEGIN":
          current_state = "READING_STDERR"
        elif line == f"{cmd_id}_STDERR_END":
          current_state = "STDERR_COMPLETE"
        elif line.startswith(f"{cmd_id}_EXIT_"):
          exit_code = int(line.split("_")[-1])
          break
        elif current_state == "READING_STDOUT":
          stdout_lines.append(line)
        elif current_state == "READING_STDERR":
          stderr_lines.append(line)

      # Combine stdout and stderr
      stdout_content = "\n".join(stdout_lines)
      stderr_content = "\n".join(stderr_lines)
      combined_output = stdout_content
      if stderr_content:
        if combined_output:
          combined_output += "\n\nSTDERR:\n" + stderr_content
        else:
          combined_output = "STDERR:\n" + stderr_content

      # Store the command result
      cmd_result = {
          "command": cmd,
          "stdout": stdout_content,
          "stderr": stderr_content,
          "output": combined_output,  # Combined for backward compatibility
          "exit_code": exit_code,
      }
      command_results.append(cmd_result)

      # Log the command result - no longer printing exit code
      if stdout_content:
        logger.info(f"Stdout for command {i+1}:\n{stdout_content}")
      if stderr_content:
        logger.warning(f"Stderr for command {i+1}:\n{stderr_content}")

      # If a command failed and we're using set -e, stop execution
      if exit_code != 0:
        logger.error(f"Command {i+1} failed")
        break

    # Close the process properly
    process.stdin.write("exit\n")
    process.stdin.flush()
    process.wait()

    # Check if all commands succeeded
    all_succeeded = all(result["exit_code"] == 0 for result in command_results)
    return all_succeeded, command_results

  except Exception as e:
    # Get detailed exception information including stack trace
    import traceback

    stack_trace = traceback.format_exc()
    error_message = (
        f"Error executing commands: {e}\n\nStack trace:\n{stack_trace}"
    )
    logger.error(error_message)

    # Kill the process if it's still running
    if process.poll() is None:
      process.terminate()

    return False, [{
        "command": "unknown",
        "stdout": "",
        "stderr": error_message,
        "output": error_message,
        "exit_code": -1,
    }]


def sample_job_configure_project_and_cluster(cluster: str, cluster_region: str):
  set_project_command = (
      f"gcloud config set project {PROJECT}",
      "gcloud container clusters get-credentials "
      f"{cluster} --region {cluster_region}",
  )
  return set_project_command


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
      config.MODEL_ID, config.FRAMEWORK, is_sample_run=True
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
            kueue_name=KUEUE_NAME,
            additional_cmds=f" --set workload.gpus={config.NUM_GPUS} ",
            test_run=True,
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

      mfu, step_time = calculate_maxtext_metrics(
          log_location, config.HYPERCOMPUTER
      )

      print(f"mfu: {mfu}")
      print(f"step_time: {step_time}")
      comment = "sample benchmarking run"
      gcs_bucket = get_job_gcs_bucket_folder(job_name)
      print(f"GCS bucket is {gcs_bucket}")

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
          is_test=True,
          logs_profile=gcs_bucket,
          workload_others=str(config),
          experiment_id=job_name,
      )
