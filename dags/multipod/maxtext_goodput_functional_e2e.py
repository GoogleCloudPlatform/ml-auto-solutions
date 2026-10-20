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

"""E2E Functional Tests DAG for MaxText Goodput Measurement.

Validates GoodputRecorder, GoodputCalculator, and GoodputMonitor end-to-end
against Cloud Logging, TensorBoard, and Cloud Monitoring:
1. Steady-State Goodput Regression: queries all pre-training and
   post-training (SFT, Multimodal SFT, LoRA, RL) model runs executed by
   maxtext_e2e_tpu_pre_training and maxtext_e2e_tpu_post_training in Cloud
   Logging and verifies Goodput against calibrated baseline thresholds.
2. Badput Attribution & Breakdown: verifies every BadputType bucket
   (TPU_INITIALIZATION, TRAINING_PREP, DATA_LOADING_SYNC, DATA_LOADING_ASYNC,
   PROGRAM_STARTUP, UNPRODUCTIVE_CHECKPOINT_SAVE_TIME,
   UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME, CUSTOM_BADPUT_EVENTS, OTHER).
3. Disruption / Recovery & Step Deviation: verifies mid-run restart badput
   (WASTED_PROGRESS_FROM_DISRUPTION, INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION)
   and step-time anomaly detection via get_step_deviation().
4. Goodput Monitor & Final Flush: verifies GoodputMonitor background uploaders
   (TensorBoard + Cloud Monitoring via GCPOptions) and clean shutdown on
   stop_goodput_uploader().
"""

import datetime
import os
import tempfile
import time
from typing import Any

from absl import logging
from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.utils.task_group import TaskGroup
from google.cloud import logging as gcp_logging
from ml_goodput_measurement import goodput
from ml_goodput_measurement import monitoring

from dags import composer_env
from dags.common import test_owner
from dags.common.goodput_utils import log_goodput_summary
from dags.common.vm_resource import GkeClusters
from xlml.utils import gcp_util, time_util

DAG_ID = "maxtext_goodput_functional_e2e"
SCHEDULED_TIME = "0 10 * * *" if composer_env.is_prod_env() else None
DEFAULT_CLUSTER = GkeClusters.TPU_V5P_BODABORG_NAP_CLUSTER

# Calibrated Goodput regression thresholds (%) across all 19 model workloads in
# maxtext_e2e_tpu_pre_training and maxtext_e2e_tpu_post_training on TPU v5p,
# derived from multi-day production E2E runs sampled from 2026-10-07 to
# 2026-10-18:
#   - Pre-training (5 models, McJAX, profile mean: 56.93%)
#   - Post-training SFT (5 models, Pathways, profile mean: 62.67%)
#   - Post-training Multimodal SFT (2 models, Pathways, profile mean: 59.55%)
#   - Post-training LoRA (1 model, Pathways, profile mean: 41.13%)
#   - Post-training RL (6 models, Pathways, profile mean: 39.23%)
#
# In `"models"`, each model maps to `(baseline_goodput_pct, min_threshold_pct)`:
#   - `baseline_goodput_pct`: Empirical multi-day mean Goodput (%) across the
#     sampled runs (logged for reference when a regression occurs).
#   - `min_threshold_pct`: Minimum acceptable Goodput (%) regression floor (~3
#     standard deviations below the empirical mean) asserted by Scenario 1.
GOODPUT_BASELINE_PROFILES: dict[str, dict[str, Any]] = {
    "pre_training": {
        "mode": "pre",
        "using_pathways": False,
        "total_steps": 2,
        "models": {
            "gemma3-4b": (59.49, 50.0),
            "gemma4-26b": (52.96, 45.0),
            "llama3_1-70b": (57.08, 45.0),
            "qwen3-30b": (62.24, 50.0),
            "gpt-oss-20b": (48.88, 40.0),
        },
    },
    "post_training_sft": {
        "mode": "sft",
        "using_pathways": True,
        "total_steps": 2,
        "models": {
            "gemma3-4b": (61.46, 50.0),
            "gemma4-26b": (71.97, 60.0),
            "llama3_1-70b": (74.74, 60.0),
            "qwen3-30b": (55.02, 45.0),
            "gpt-oss-20b": (48.57, 30.0),
        },
    },
    "post_training_multimodal_sft": {
        "mode": "multimodal_sft",
        "using_pathways": True,
        "total_steps": 2,
        "models": {
            "gemma3-4b": (58.20, 50.0),
            "qwen3-vl-2b": (60.89, 50.0),
        },
    },
    "post_training_lora": {
        "mode": "lora",
        "using_pathways": True,
        "total_steps": 2,
        "models": {
            "gemma3-4b": (41.13, 35.0),
        },
    },
    "post_training_rl": {
        "mode": "rl",
        "using_pathways": True,
        "total_steps": 2,
        "models": {
            "gemma3-4b": (42.70, 35.0),
            "gemma4-26b": (35.04, 25.0),
            "llama3_1-70b": (44.95, 35.0),
            "qwen3-30b": (46.34, 35.0),
            "gpt-oss-20b": (42.88, 30.0),
            "qwen3_5-35b": (21.91, 15.0),
        },
    },
}

# Synthetic job timeline durations for Scenario 2 and Scenario 3.
_TEST_TPU_INIT_TIME = datetime.timedelta(seconds=4)
_TEST_TRAINING_PREPARATION_TIME = datetime.timedelta(seconds=6)
_TEST_SYNC_DATA_LOADING_TIME = datetime.timedelta(seconds=4)
_TEST_ASYNC_DATA_LOADING_TIME = datetime.timedelta(seconds=2)
_TEST_FIRST_STEP_EXTRA_TIME = datetime.timedelta(seconds=6)
_TEST_STEP_TIME = datetime.timedelta(seconds=10)
_TEST_ANOMALOUS_STEP_EXTRA_TIME = datetime.timedelta(seconds=20)
_TEST_CUSTOM_BADPUT_TIME = datetime.timedelta(seconds=5)
_TEST_CKPT_SAVE_TIME = datetime.timedelta(seconds=3)
_TEST_CKPT_RESTORE_TIME = datetime.timedelta(seconds=2)
_TEST_OTHER_GAP_TIME = datetime.timedelta(seconds=5)
_TEST_DISRUPTION_DOWNTIME = datetime.timedelta(seconds=10)
_TEST_TOTAL_STEPS = 4


def _flush_recorder(recorder: goodput.GoodputRecorder) -> None:
  """Flushes buffered async Cloud Logging entries if supported."""
  flush_fn = getattr(recorder, "flush", None)
  if callable(flush_fn):
    flush_fn()


def _calculate_workload_goodput(
    workload_id: str,
    using_pathways: bool = False,
    max_attempts: int = 12,
    retry_delay_s: float = 5.0,
    configured_ideal_step_time: float | None = None,
) -> tuple[float, dict[Any, Any], int, dict[int, float]]:
  """Instantiates GoodputCalculator in a temp cache dir and queries metrics."""
  with tempfile.TemporaryDirectory() as cache_dir:
    calculator = goodput.GoodputCalculator(
        job_name=workload_id,
        logger_name=f"goodput_{workload_id}",
        using_pathways=using_pathways,
        cache_dir=cache_dir,
    )
    last_err: Exception | None = None
    for attempt in range(max_attempts):
      try:
        job_goodput, badput_breakdown, last_step = calculator.get_job_goodput(
            include_badput_breakdown=True
        )
        # pylint: disable=protected-access
        if calculator._goodput_cache.get_job_end_time() is None:
          raise ValueError("job_end_time not yet indexed in Cloud Logging")
        # pylint: enable=protected-access
        break
      except ValueError as err:
        if max_attempts == 1:
          raise
        last_err = err
        # pylint: disable=protected-access
        calculator._goodput_cache.clear_cache()
        calculator._cloud_logger.job_start_time = None
        # pylint: enable=protected-access
        if attempt + 1 < max_attempts:
          time.sleep(retry_delay_s)
    else:
      raise RuntimeError(
          f"Goodput query failed after {max_attempts} attempts"
      ) from last_err

    step_deviations = (
        calculator.get_step_deviation(
            configured_ideal_step_time=configured_ideal_step_time
        )
        if configured_ideal_step_time is not None
        else {}
    )
    return job_goodput, badput_breakdown, last_step, step_deviations


@task
def resolve_e2e_run_name(
    project_id: str,
    target_e2e_run_name: str | None = None,
) -> str:
  """Resolves the latest fully completed E2E run_name in Cloud Logging."""
  if target_e2e_run_name and target_e2e_run_name.strip().lower() not in (
      "",
      "none",
  ):
    return target_e2e_run_name.strip()

  expected_prefixes = [
      f"{model}-{mode}"
      for cfg in GOODPUT_BASELINE_PROFILES.values()
      for mode in (cfg["mode"],)
      for model in cfg["models"]
  ]
  start_iso = (
      datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
  ).isoformat()
  entries = gcp_logging.Client(project=project_id).list_entries(
      filter_=(
          'resource.type="k8s_container" '
          f'AND logName:"projects/{project_id}/logs/goodput_" '
          f'AND jsonPayload.job_end_time:* AND timestamp>="{start_iso}"'
      ),
      order_by=gcp_logging.DESCENDING,
      page_size=50,
  )

  completed_by_run: dict[str, set[str]] = {}
  for entry in entries:
    job_name = (
        str(entry.payload.get("job_name", ""))
        if isinstance(entry.payload, dict)
        else ""
    )
    for prefix in expected_prefixes:
      if job_name.startswith(f"{prefix}-"):
        run_name = job_name.removeprefix(f"{prefix}-")
        completed_by_run.setdefault(run_name, set()).add(prefix)
        if len(completed_by_run[run_name]) == len(expected_prefixes):
          logging.info("Selected completed E2E run: %s", run_name)
          return run_name
        break

  raise RuntimeError(
      f"No completed E2E run ({len(expected_prefixes)} workloads) found in "
      f"{project_id} over the last 7 days."
  )


@task
def verify_steady_state_goodput(
    profile_name: str,
    run_name: str,
    project_id: str,
) -> None:
  """Scenario 1 verifier: checks all models in a profile against thresholds."""
  profile = GOODPUT_BASELINE_PROFILES[profile_name]
  mode = profile["mode"]
  using_pathways = profile["using_pathways"]
  expected_last_step = profile["total_steps"] - 1
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id

  failures: list[str] = []
  for model_name, (baseline_pct, threshold) in profile["models"].items():
    workload_id = f"{model_name}-{mode}-{run_name}"
    try:
      job_goodput, badput_breakdown, last_step, _ = _calculate_workload_goodput(
          workload_id=workload_id,
          using_pathways=using_pathways,
          max_attempts=1,
      )
    except Exception as err:  # pylint: disable=broad-exception-caught
      failures.append(f"[{workload_id}] Failed to query Goodput: {err}")
      continue

    log_goodput_summary(
        job_goodput=job_goodput,
        badput_breakdown=badput_breakdown,
        last_step=last_step,
        scenario_name=f"steady_state:{profile_name}:{model_name}",
    )

    if last_step < expected_last_step:
      failures.append(
          f"[{workload_id}] Expected last_step >= {expected_last_step}, "
          f"got {last_step}"
      )
    if job_goodput < threshold:
      failures.append(
          f"[{workload_id}] Goodput regression: computed {job_goodput:.2f}% "
          f"< threshold {threshold:.2f}% (calibrated baseline: "
          f"{baseline_pct:.2f}%)"
      )

  assert not failures, (
      f"[{profile_name}] Steady-state Goodput verification failed for "
      f"run_name={run_name}:\n" + "\n".join(failures)
  )


@task
def badput_attribution_breakdown(
    ts_nodash: str,
) -> None:
  """Scenario 2: Badput Attribution & Breakdown Functional Test.

  Executes a workload timeline containing known durations for every badput
  phase (TPU_INITIALIZATION, TRAINING_PREP, DATA_LOADING_SYNC,
  DATA_LOADING_ASYNC, PROGRAM_STARTUP, UNPRODUCTIVE_CHECKPOINT_SAVE_TIME,
  UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME, CUSTOM_BADPUT_EVENTS, and OTHER) and
  asserts that GoodputCalculator accurately attributes each BadputType bucket.
  """
  workload_id = f"goodput-func-badput-breakdown-{ts_nodash}"
  logger_name = f"goodput_{workload_id}"

  now_utc = datetime.datetime.now(datetime.timezone.utc)
  job_start = now_utc - datetime.timedelta(minutes=10)
  curr_time = job_start

  recorder = goodput.GoodputRecorder(
      job_name=workload_id,
      logger_name=logger_name,
      logging_enabled=True,
  )
  recorder.record_job_start_time(curr_time)

  # 1. TPU initialization (4s)
  recorder.record_tpu_init_start_time(curr_time)
  curr_time += _TEST_TPU_INIT_TIME
  recorder.record_tpu_init_end_time(curr_time)

  # 2. Training preparation (6s)
  recorder.record_training_preparation_start_time(curr_time)
  curr_time += _TEST_TRAINING_PREPARATION_TIME
  recorder.record_training_preparation_end_time(curr_time)

  # 3. Synchronous data loading before step 0 (4s)
  recorder.record_data_loading_start_time(curr_time)
  curr_time += _TEST_SYNC_DATA_LOADING_TIME
  recorder.record_data_loading_end_time(curr_time)

  # 4. Unattributed gap before step 0 (OTHER badput = 5s)
  curr_time += _TEST_OTHER_GAP_TIME

  # 5. Training steps 0..3 with startup compilation on step 0, async data
  #    loading on step 1, and a custom badput event (EVAL_STEP) on step 2.
  for step in range(_TEST_TOTAL_STEPS):
    recorder.record_step_start_time(step, curr_time)
    if step == 0:
      curr_time += _TEST_FIRST_STEP_EXTRA_TIME
    if step == 1:
      dl_start = curr_time + datetime.timedelta(seconds=1.0)
      recorder.record_data_loading_start_time(dl_start)
      recorder.record_data_loading_end_time(
          dl_start + _TEST_ASYNC_DATA_LOADING_TIME
      )
    if step == 2:
      eval_start = curr_time + datetime.timedelta(seconds=1.0)
      recorder.record_custom_badput_event_start_time(eval_start, "eval_step")
      eval_end = eval_start + _TEST_CUSTOM_BADPUT_TIME
      recorder.record_custom_badput_event_end_time(eval_end, "eval_step")
      curr_time += _TEST_CUSTOM_BADPUT_TIME
    curr_time += _TEST_STEP_TIME

  ckpt_save_s = _TEST_CKPT_SAVE_TIME.total_seconds()
  ckpt_restore_s = _TEST_CKPT_RESTORE_TIME.total_seconds()

  # 6. Checkpoint save and restore entries in the goodput logger stream.
  # pylint: disable=protected-access
  recorder._cloud_logger.write_cloud_logging_entry({
      "job_name": workload_id,
      "step": 2,
      "event_type": "save",
      "directory": f"gs://runner-maxtext-logs/{workload_id}",
      "reached_preemption": False,
      "synchronous": True,
      "wait_for_prev_start_time": curr_time.timestamp() - 10.0,
      "wait_for_prev_duration_secs": 0.5,
      "checkpointer_blocking_start_time": curr_time.timestamp() - 9.0,
      "checkpointer_blocking_duration_secs": 1.5,
      "get_old_steps_start_time": curr_time.timestamp() - 7.0,
      "get_old_steps_duration_secs": 1.0,
      "checkpoint_manager_blocking_start_time": curr_time.timestamp() - 10.0,
      "checkpoint_manager_blocking_duration_secs": ckpt_save_s,
  })
  recorder._cloud_logger.write_cloud_logging_entry({
      "job_name": workload_id,
      "step": 0,
      "event_type": "restore",
      "directory": f"gs://runner-maxtext-logs/{workload_id}",
      "checkpointer_start_time": job_start.timestamp() + 5.0,
      "checkpointer_duration_secs": ckpt_restore_s,
      "checkpoint_manager_start_time": job_start.timestamp() + 5.0,
      "checkpoint_manager_duration_secs": ckpt_restore_s,
  })
  # pylint: enable=protected-access

  # 7. Job end time.
  job_end = curr_time
  recorder.record_job_end_time(job_end)
  _flush_recorder(recorder)

  total_time_s = (job_end - job_start).total_seconds()

  job_goodput, badput_breakdown, last_step, _ = _calculate_workload_goodput(
      workload_id=workload_id,
  )

  log_goodput_summary(
      job_goodput=job_goodput,
      badput_breakdown=badput_breakdown,
      last_step=last_step,
      scenario_name="badput_attribution",
  )

  assert (
      last_step == _TEST_TOTAL_STEPS - 1
  ), f"Expected last_step={_TEST_TOTAL_STEPS - 1}, got {last_step}"

  expected_durations = {
      goodput.BadputType.TPU_INITIALIZATION: (
          _TEST_TPU_INIT_TIME.total_seconds()
      ),
      goodput.BadputType.TRAINING_PREP: (
          _TEST_TRAINING_PREPARATION_TIME.total_seconds()
      ),
      goodput.BadputType.DATA_LOADING_SYNC: (
          _TEST_SYNC_DATA_LOADING_TIME.total_seconds()
      ),
      goodput.BadputType.DATA_LOADING_ASYNC: (
          _TEST_ASYNC_DATA_LOADING_TIME.total_seconds()
      ),
      goodput.BadputType.PROGRAM_STARTUP: (
          _TEST_FIRST_STEP_EXTRA_TIME.total_seconds()
      ),
      goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_SAVE_TIME: ckpt_save_s,
      goodput.BadputType.UNPRODUCTIVE_CHECKPOINT_RESTORE_TIME: ckpt_restore_s,
      goodput.BadputType.OTHER: (
          _TEST_OTHER_GAP_TIME.total_seconds() - ckpt_restore_s
      ),
  }

  for badput_type, duration_s in expected_durations.items():
    assert (
        badput_type in badput_breakdown
    ), f"Missing {badput_type.name} in badput_breakdown"
    expected_pct = (duration_s / total_time_s) * 100
    actual_pct = float(badput_breakdown[badput_type])
    assert abs(actual_pct - expected_pct) <= 0.5, (
        f"{badput_type.name} mismatch: expected {expected_pct:.2f}%, "
        f"got {actual_pct:.2f}%"
    )

  custom_events = badput_breakdown.get(
      goodput.BadputType.CUSTOM_BADPUT_EVENTS, {}
  )
  assert (
      isinstance(custom_events, dict) and "EVAL_STEP" in custom_events
  ), f"Expected EVAL_STEP in CUSTOM_BADPUT_EVENTS, got {custom_events}"
  expected_eval_pct = (
      _TEST_CUSTOM_BADPUT_TIME.total_seconds() / total_time_s
  ) * 100
  actual_eval_pct = custom_events["EVAL_STEP"]
  assert abs(actual_eval_pct - expected_eval_pct) <= 0.5, (
      f"CUSTOM_BADPUT_EVENTS[EVAL_STEP] mismatch: "
      f"expected {expected_eval_pct:.2f}%, got {actual_eval_pct:.2f}%"
  )


@task
def disruption_recovery_and_step_deviation(
    ts_nodash: str,
) -> None:
  """Scenario 3: Disruption / Recovery & Step Deviation Functional Test.

  Emulates a mid-run disruption with rollback to an earlier checkpointed step
  plus a step-time anomaly, then queries GoodputCalculator.get_job_goodput() and
  GoodputCalculator.get_step_deviation() to verify wasted progress badput,
  infrastructure recovery badput, and step deviation calculation end-to-end.
  """
  workload_id = f"goodput-func-disruption-{ts_nodash}"
  logger_name = f"goodput_{workload_id}"

  now_utc = datetime.datetime.now(datetime.timezone.utc)
  job_start = now_utc - datetime.timedelta(minutes=10)
  curr_time = job_start

  recorder = goodput.GoodputRecorder(
      job_name=workload_id,
      logger_name=logger_name,
      logging_enabled=True,
  )

  # Segment 1: steps 0..4 (checkpoint saved at step 1, steps 2..4 lost on crash)
  recorder.record_job_start_time(curr_time)
  recorder.record_tpu_init_start_time(curr_time)
  curr_time += _TEST_TPU_INIT_TIME
  recorder.record_tpu_init_end_time(curr_time)

  recorder.record_training_preparation_start_time(curr_time)
  curr_time += _TEST_TRAINING_PREPARATION_TIME
  recorder.record_training_preparation_end_time(curr_time)

  recorder.record_data_loading_start_time(curr_time)
  curr_time += _TEST_SYNC_DATA_LOADING_TIME
  recorder.record_data_loading_end_time(curr_time)

  for step in range(5):
    recorder.record_step_start_time(step, curr_time)
    curr_time += _TEST_STEP_TIME

  # Simulate mid-run disruption and restart from step 2
  curr_time += _TEST_DISRUPTION_DOWNTIME
  recorder.record_job_start_time(curr_time)

  recorder.record_tpu_init_start_time(curr_time)
  curr_time += _TEST_TPU_INIT_TIME
  recorder.record_tpu_init_end_time(curr_time)

  recorder.record_training_preparation_start_time(curr_time)
  curr_time += _TEST_TRAINING_PREPARATION_TIME
  recorder.record_training_preparation_end_time(curr_time)

  recorder.record_data_loading_start_time(curr_time)
  curr_time += _TEST_SYNC_DATA_LOADING_TIME
  recorder.record_data_loading_end_time(curr_time)

  # Segment 2: resume from step 2..5, with a step-time anomaly on step 3
  for step in range(2, 6):
    recorder.record_step_start_time(step, curr_time)
    curr_time += _TEST_STEP_TIME
    if step == 3:
      curr_time += _TEST_ANOMALOUS_STEP_EXTRA_TIME

  recorder.record_job_end_time(curr_time)
  _flush_recorder(recorder)

  (
      job_goodput,
      badput_breakdown,
      last_step,
      step_deviations,
  ) = _calculate_workload_goodput(
      workload_id=workload_id,
      configured_ideal_step_time=_TEST_STEP_TIME.total_seconds(),
  )

  log_goodput_summary(
      job_goodput=job_goodput,
      badput_breakdown=badput_breakdown,
      last_step=last_step,
      scenario_name="disruption_recovery",
  )
  logging.info("[disruption_recovery] Step deviations: %s", step_deviations)

  assert last_step == 5, f"Expected last_step=5 after recovery, got {last_step}"
  assert 0.0 < job_goodput < 100.0, f"Unexpected goodput: {job_goodput:.2f}%"

  for disruption_type in (
      goodput.BadputType.WASTED_PROGRESS_FROM_DISRUPTION,
      goodput.BadputType.INFRASTRUCTURE_RECOVERY_FROM_DISRUPTION,
  ):
    disruption_pct = float(badput_breakdown.get(disruption_type, 0.0))
    assert disruption_pct > 0.0, (
        f"Expected positive {disruption_type.name} badput, "
        f"got {disruption_pct:.2f}%"
    )

  anomalous_extra_s = _TEST_ANOMALOUS_STEP_EXTRA_TIME.total_seconds()
  assert step_deviations, "Expected non-empty step_deviations dictionary"
  assert (
      3 in step_deviations
  ), f"Expected anomalous step 3 in step_deviations, got {step_deviations}"
  assert abs(step_deviations[3] - anomalous_extra_s) <= 0.5, (
      f"Expected step 3 deviation ~{anomalous_extra_s}s, "
      f"got {step_deviations[3]:.2f}s"
  )
  assert (
      abs(step_deviations[2]) <= 0.5
  ), f"Expected normal step 2 deviation ~0s, got {step_deviations[2]:.2f}s"


def _query_gcm_goodput_time_series_with_retry(
    project_id: str,
    workload_id: str,
    max_attempts: int = 6,
    retry_delay_s: float = 5.0,
) -> list[Any]:
  """Queries Cloud Monitoring for goodput_time series for workload_id."""
  metric_filter = (
      'metric.type = "compute.googleapis.com/workload/goodput_time" '
      f'AND resource.labels.workload_id = "{workload_id}"'
  )
  for attempt in range(max_attempts):
    now = time.time()
    series = gcp_util.list_time_series(
        project_id=project_id,
        filter_str=metric_filter,
        start_time=time_util.TimeUtil.from_unix_seconds(now - 3600),
        end_time=time_util.TimeUtil.from_unix_seconds(now + 60),
    )
    if series:
      return series
    if attempt + 1 < max_attempts:
      time.sleep(retry_delay_s)
  return []


@task
def goodput_monitor_and_final_flush(
    project_id: str,
    target_e2e_run_name: str,
) -> None:
  """Scenario 4: Goodput Monitor & Final Flush Functional Test.

  Connects to both Pre-Training (`gemma3-4b-pre-<run_name>`,
  pathway_enabled=False) and Post-Training (`gemma3-4b-sft-<run_name>`,
  pathway_enabled=True) E2E workload stream in Cloud Logging and verifies:
  1. GoodputMonitor uploader lifecycle (monitor_goodput=True,
     GCPOptions(enable_gcp_goodput_metrics=True)) and clean final flush on
     stop_goodput_uploader().
  2. Flushed TensorBoard scalar event files written by GoodputMonitor.
  3. Background uploads of compute.googleapis.com/workload/goodput_time time
     series to Google Cloud Monitoring.
  """
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
  for profile_label, prof_cfg in GOODPUT_BASELINE_PROFILES.items():
    mode = prof_cfg["mode"]
    using_pathways = prof_cfg["using_pathways"]
    target_workload_id = f"gemma3-4b-{mode}-{target_e2e_run_name}"
    logger_name = f"goodput_{target_workload_id}"
    logging.info(
        "[goodput_monitor:%s] Running GoodputMonitor uploader lifecycle test "
        "against workload=%s (pathway_enabled=%s)",
        profile_label,
        target_workload_id,
        using_pathways,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
      tb_dir = os.path.join(tmp_dir, "tensorboard")
      cache_dir = os.path.join(tmp_dir, "cache")
      os.makedirs(tb_dir, exist_ok=True)
      os.makedirs(cache_dir, exist_ok=True)

      gcp_options = monitoring.GCPOptions(
          enable_gcp_goodput_metrics=True,
          enable_gcp_step_deviation_metrics=True,
          acc_type="v5p-8",
          replica_id="0",
      )

      goodput_monitor = monitoring.GoodputMonitor(
          job_name=target_workload_id,
          logger_name=logger_name,
          tensorboard_dir=tb_dir,
          upload_interval=1,
          monitoring_enabled=True,
          pathway_enabled=using_pathways,
          include_badput_breakdown=True,
          include_step_deviation=True,
          configured_ideal_step_time=1.0,
          step_deviation_interval_seconds=1,
          gcp_options=gcp_options,
          final_flush_timeout_seconds=None,
          cache_dir=cache_dir,
      )

      # Run final flush synchronously in-process
      goodput_monitor.stop_goodput_uploader()
      goodput_monitor.stop_step_deviation_uploader()

      # Assert background processes exited cleanly on final flush.
      # pylint: disable=protected-access
      assert (
          goodput_monitor._goodput_process is None
      ), "Expected _goodput_process to be None after stop_goodput_uploader()"
      assert goodput_monitor._step_deviation_process is None, (
          "Expected _step_deviation_process to be None after "
          "stop_step_deviation_uploader()"
      )
      # pylint: enable=protected-access
      logging.info(
          "[goodput_monitor:%s] Verified stop_goodput_uploader() and "
          "stop_step_deviation_uploader() clean shutdown "
          "(_goodput_process=None, _step_deviation_process=None)",
          profile_label,
      )

      tb_goodput_dir = os.path.join(tb_dir, "goodput")
      tb_files = (
          os.listdir(tb_goodput_dir) if os.path.isdir(tb_goodput_dir) else []
      )
      tb_sizes = [
          os.path.getsize(os.path.join(tb_goodput_dir, fname))
          for fname in tb_files
      ]
      logging.info(
          "[goodput_monitor:%s] Flushed TensorBoard files in %s: %s "
          "(sizes: %s)",
          profile_label,
          tb_goodput_dir,
          tb_files,
          tb_sizes,
      )
      assert (
          tb_files
      ), f"Expected TensorBoard event files in {tb_goodput_dir}, found none"
      assert any(size > 40 for size in tb_sizes), (
          "Expected flushed TensorBoard scalar events (>40B header) in "
          f"{tb_goodput_dir}, got file sizes {tb_sizes}"
      )

      gcm_project_id = gcp_options.project_id or project_id

    # Verify background upload to Google Cloud Monitoring for the workload.
    gcm_series = _query_gcm_goodput_time_series_with_retry(
        project_id=gcm_project_id,
        workload_id=target_workload_id,
    )
    logging.info(
        "[goodput_monitor:%s] Found %d Cloud Monitoring goodput_time series "
        "for %s in project=%s",
        profile_label,
        len(gcm_series),
        target_workload_id,
        gcm_project_id,
    )
    assert gcm_series, (
        "Expected compute.googleapis.com/workload/goodput_time time series in "
        f"Cloud Monitoring (project={gcm_project_id}) for "
        f"workload_id={target_workload_id}, found none"
    )
    latest_goodput_time = gcm_series[0].points[0].value.double_value
    logging.info(
        "[goodput_monitor:%s] Cloud Monitoring goodput_time for %s: %.2fs",
        profile_label,
        target_workload_id,
        latest_goodput_time,
    )
    assert latest_goodput_time > 0.0, (
        f"Expected positive goodput_time in Cloud Monitoring for "
        f"{target_workload_id}, got {latest_goodput_time}"
    )


with models.DAG(
    dag_id=DAG_ID,
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "goodput",
        "e2e",
        "functional",
        "nightly",
        "mlscale_devx",
        "TPU",
        "v5p-8",
    ],
    start_date=datetime.datetime(2026, 6, 10),
    catchup=False,
    default_args={
        "owner": test_owner.KEVIN_Y,
        "retries": 0,
    },
    params={
        "target_e2e_run_name": Param(
            default=None,
            type=["null", "string"],
            description=(
                "(Optional) Target E2E run_name from "
                "maxtext_e2e_tpu_pre_training and "
                "maxtext_e2e_tpu_post_training to query in Cloud Logging for "
                "Goodput verification (e.g. e2e-39084001154). Leave empty to "
                "automatically discover the latest E2E run where all 19 model "
                "workloads have completed."
            ),
        ),
    },
) as dag:
  ts_nodash_param = "{{ ts_nodash }}"
  e2e_project_id = DEFAULT_CLUSTER.project

  resolved_run_name = resolve_e2e_run_name(
      project_id=e2e_project_id,
      target_e2e_run_name="{{ params.target_e2e_run_name or '' }}",
  )

  with TaskGroup(group_id="steady_state_goodput_regression") as steady_group:
    for prof_name in GOODPUT_BASELINE_PROFILES:
      verify_steady_state_goodput.override(
          task_id=f"verify_{prof_name}_goodput"
      )(
          profile_name=prof_name,
          run_name=resolved_run_name,
          project_id=e2e_project_id,
      )

  badput_breakdown_task = badput_attribution_breakdown(
      ts_nodash=ts_nodash_param,
  )
  disruption_recovery_task = disruption_recovery_and_step_deviation(
      ts_nodash=ts_nodash_param,
  )
  monitor_flush_task = goodput_monitor_and_final_flush(
      project_id=e2e_project_id,
      target_e2e_run_name=resolved_run_name,
  )

  chain(
      resolved_run_name,
      [
          steady_group,
          badput_breakdown_task,
          disruption_recovery_task,
          monitor_flush_task,
      ],
  )
