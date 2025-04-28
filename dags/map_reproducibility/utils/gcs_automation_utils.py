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

"""Bash helper commands for GCS automation repo."""

from collections import namedtuple


def gcs_automation_cmds(
    gcs_results_generator: bool,
    run_details: namedtuple,
    logs_bucket: str,
    gcs_metrics_bucket: str,
    recipe_repo_root: str,
    gcs_automation_repo_root: str,
):
  """Get the commands for GCS automation.

  Args:
      gcs_results_generator: True if enabling GCS run results generator.
      run_details: The namedtuple which stores the workload configs.
      logs_bucket: The name of the logs bucket.
      gcs_metrics_bucket: The GCS bucket in which the metrics files are stored.
      recipe_repo_root: The root path to the recipe repo.
      gcs_automation_repo_root: The root path to the gcs automation repo.

  Returns:
      A command to run GCS automation pipeline.
  """
  if not gcs_results_generator:
    return ()

  cmds = _get_generated_job_name(
      logs_bucket=logs_bucket,
  ) + _logs_scraper_cmds(
      logs_bucket=logs_bucket,
      gcs_metrics_bucket=gcs_metrics_bucket,
      recipe_repo_root=recipe_repo_root,
  ) + _run_results_generator_cmds(
      run_details=run_details,
      gcs_automation_repo_root=gcs_automation_repo_root,
  )

  return cmds


def _get_generated_job_name(logs_bucket: str = None):
  """Get the commands for exporting the generated job name.

  Args:
      logs_bucket: The name of the logs bucket.

  Returns:
      A command to export the generated job name.
  """
  cmds = (
      "export GENERATED_JOB_NAME=$(gcloud storage ls "
      f"gs://{logs_bucket}/nemo-experiments/$JOB_NAME*/ "
      "| head -n 1 | awk -F '/' '{print $5}')",
      'echo "GENERATED_JOB_NAME ${GENERATED_JOB_NAME}"',
  )
  return cmds


def _logs_scraper_cmds(
    logs_bucket: str,
    gcs_metrics_bucket: str,
    recipe_repo_root: str,
):
  """Get the commands for GCS log scraper.

  Args:
      logs_bucket: The name of the logs bucket.
      gcs_metrics_bucket: The GCS bucket in which the metrics files are stored.
      recipe_repo_root: The root path to the recipe repo.

  Returns:
      A command to run GCS log scraper.
  """
  cmds = (
      "python3 "
      f"{recipe_repo_root}/src/utils/checkpointing_metrics"
      f"/log_scraper_nemo.py --logs_bucket={logs_bucket} "
      f"--raw_metrics_bucket={gcs_metrics_bucket}",
  )
  return cmds


def _run_results_generator_cmds(
    run_details: namedtuple,
    gcs_automation_repo_root: str,
):
  """Get the commands for GCS run results generator.

  Args:
      run_details: The namedtuple which stores the workload configs.
      gcs_automation_repo_root: The root path to the gcs automation repo.

  Returns:
      A command to run GCS results generator.
  """
  cmds = (
      "export AVG_STEP_TIME=$(sed -n '1s/.*: //p' $METRICS_FILE)",
      "export MFU=$(sed -n '2s/.*: //p' $METRICS_FILE)",
      "echo $AVG_STEP_TIME",
      "echo $MFU",
  )
  metrics_cmds = f"python3 {gcs_automation_repo_root}/metrics.py "
  ckpt_cmds = "checkpointing "
  for name, val in run_details._asdict().items():
    name = name.replace("_", "-")
    if not val or name == "benchmark-type":
      continue
    elif "checkpointing" in name:
      ckpt_cmds += f"--{name}={val} "
    else:
      metrics_cmds += f"--{name}={val} "
  if run_details.benchmark_type == "checkpointing":
    metrics_cmds += ckpt_cmds

  cmds += (
      metrics_cmds,
  )
  return cmds
