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

from absl import logging
from airflow.decorators import task
from ml_goodput_measurement import goodput


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

  logging.info(f"Last step recorded: {last_step}")
  logging.info(f"Goodput (%): {current_goodput:.2f}%")
  logging.info("\n--- Badput Breakdown ---")

  for badput_type, percentage in badput_breakdown.items():
    if badput_type == goodput.BadputType.CUSTOM_BADPUT_EVENTS:
      logging.info(f"Badput due to {badput_type}:")
      custom_events = percentage
      if isinstance(custom_events, dict):
        for event_name, event_percentage in custom_events.items():
          logging.info(f"  - {event_name}: {event_percentage:.2f}%")
    else:
      # Access the name attribute of the enum member
      logging.info(f"Badput due to {badput_type.name}: {percentage:.2f}%")
