# Copyright 2026 Google LLC
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

"""Common Goodput measurement utilities and tasks."""

import os
from typing import Any

from absl import logging
from airflow.decorators import task
from ml_goodput_measurement import goodput


def log_goodput_summary(
    job_goodput: float,
    badput_breakdown: dict[Any, Any],
    last_step: int,
    scenario_name: str | None = None,
) -> None:
  """Logs Goodput and per-category Badput breakdown for a workload or scenario.

  Args:
    job_goodput: Computed cumulative Goodput percentage.
    badput_breakdown: Mapping from BadputType to percentage (or nested dict of
      event_name -> percentage for CUSTOM_BADPUT_EVENTS).
    last_step: Last productive training step recorded.
    scenario_name: Optional label prefixed to each log line (e.g.,
      "[steady_state:pre_training]").
  """
  prefix = f"[{scenario_name}] " if scenario_name else ""
  logging.info("%sLast step recorded: %d", prefix, last_step)
  logging.info("%sGoodput (%%): %.2f%%", prefix, job_goodput)
  logging.info("%s--- Badput Breakdown ---", prefix)

  for badput_type, percentage in badput_breakdown.items():
    if badput_type == goodput.BadputType.CUSTOM_BADPUT_EVENTS:
      if isinstance(percentage, dict) and percentage:
        logging.info("%sBadput due to %s:", prefix, badput_type.name)
        for event_name, event_percentage in percentage.items():
          logging.info(
              "%s  - %s: %.2f%%",
              prefix,
              event_name,
              event_percentage,
          )
      else:
        logging.info(
            "%sBadput due to %s: 0.00%%",
            prefix,
            badput_type.name,
        )
    else:
      logging.info(
          "%sBadput due to %s: %.2f%%",
          prefix,
          badput_type.name,
          percentage,
      )


@task
def check_workload_goodput(
    workload_id: str,
    project_id: str,
    using_pathways: bool,
) -> None:
  """Queries and logs Goodput and Badput breakdown metrics for a workload.

  Queries Cloud Logging via `ml_goodput_measurement.goodput.GoodputCalculator`
  for Goodput events recorded under logger name `goodput_<workload_id>` in
  `project_id`, then logs the last recorded training step, overall Goodput
  percentage, and per-category Badput breakdown (including any nested custom
  Badput events).

  Contract & Side Effects:
    - Assumes the target workload has emitted Goodput logs to Cloud Logging
      under the log name `goodput_<workload_id>` in `project_id`.
    - Sets `os.environ["GOOGLE_CLOUD_PROJECT"] = project_id` within the task
      execution environment so `GoodputCalculator` queries the target GCP
      project.
    - Returns `None`; metrics are emitted to the task logs for observability.
      Raises an exception (failing the Airflow task) if querying Cloud Logging
      or computing Goodput fails.

  Args:
    workload_id: Unique workload or job identifier (`job_name`) used when
      recording Goodput metrics (resolves to logger `goodput_<workload_id>`).
    project_id: Google Cloud project ID where the workload's Cloud Logging
      entries are stored.
    using_pathways: Forwarded to `GoodputCalculator(using_pathways=...)` to
      indicate whether the workload ran under the Pathways single-controller
      runtime (`True`) or standard multi-controller McJAX (`False`). When
      `True`, `GoodputCalculator` is configured for Pathways execution semantics
      (where worker disruptions and elastic recoveries are handled in-place by
      the Pathways controller and surface as anomalous step-time intervals
      rather than full job restarts); when `False`, it evaluates standard
      job-restart disruption boundaries across training segments.
  """
  goodput_logger_name = f"goodput_{workload_id}"
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
  goodput_calculator = goodput.GoodputCalculator(
      job_name=workload_id,
      logger_name=goodput_logger_name,
      using_pathways=using_pathways,
  )
  (
      current_goodput,
      badput_breakdown,
      last_step,
  ) = goodput_calculator.get_job_goodput(include_badput_breakdown=True)

  log_goodput_summary(
      job_goodput=current_goodput,
      badput_breakdown=badput_breakdown,
      last_step=last_step,
  )
