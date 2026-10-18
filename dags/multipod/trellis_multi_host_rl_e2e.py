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

"""Trellis Multi-Host Distributed RL E2E Tests DAG.

Lightweight, generic orchestration DAG for Trellis multi-host distributed RL
end-to-end validation on GKE (`bodaborg-v5p-nap` in `europe-west4`).
All workload, model, checkpoint, and TPU topology defaults live in the
`google/trellis` repository (`tests/multi_host/run_multi_host_gsm8k_e2e.sh`)
so that model/recipe changes do not require DAG updates:
  1. Validates the external trigger and required parameters via
     `validate_trigger` (no GitHub token is required to run this DAG).
  2. Clones `google/trellis` at `commit_sha` and executes `runner_script`
     (forwarding any optional overrides in `dag_run.conf` or `runner_env`).
  3. Cleans up all GKE JobSets for `job_prefix` with `TriggerRule.ALL_DONE`.
  4. Optionally fires a GitHub `repository_dispatch`
     (`airflow-trellis-multi-host-rl-callback`) event via
     `xlml.utils.github.trigger_github_repository_dispatch` with
     `TriggerRule.ALL_DONE` -- only when the caller explicitly supplies
     `github_token` in `dag_run.conf`. GitHub Actions callers are expected to
     poll the Airflow REST API (dagRuns state + the `return_value` XCom of
     `launch_and_verify_rl_jobsets`) instead of relying on the callback.

Credentials: the repository clone uses the `GITHUB_PAT_TRELLIS_CI` Airflow
Variable (read-only PAT), read at task runtime and never written to XCom or
exposed as a Param default. A caller-supplied `github_token` (callback) is
likewise read from `dag_run.conf` at runtime only.
"""

import base64
import datetime
import json
import logging
import os
import re
import subprocess
import tempfile
from typing import Any

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.operators.python import get_current_context
from airflow.utils.trigger_rule import TriggerRule
from dags.common.quarantined_tests import safe_get_from_variable
from xlml.utils.github import trigger_github_repository_dispatch

DEFAULT_CLUSTER_PROJECT = "cloud-tpu-shared-capacity"
DEFAULT_CLUSTER_ZONE = "europe-west4"
DEFAULT_CLUSTER_NAME = "bodaborg-v5p-nap"
DEFAULT_K8S_NAMESPACE = "trellis"
DEFAULT_ROLLOUT_REPLICAS = 2
DEFAULT_RUNNER_SCRIPT = "tests/multi_host/run_multi_host_gsm8k_e2e.sh"
# Name of the Airflow Variable holding a read-only PAT for the private
# google/trellis repository. Resolved inside tasks at runtime (see
# `_clone_token`), never at DAG-parse time, so the secret is not baked into
# Param defaults, rendered templates, or XCom.
GITHUB_PAT_TRELLIS_CI_VAR = "GITHUB_PAT_TRELLIS_CI"

# Optional dag_run.conf keys mapped to runner environment variables when
# explicitly supplied by the caller. When omitted, the runner script in
# google/trellis uses its own built-in defaults.
_OPTIONAL_CONF_TO_ENV = {
    "recipe": "RECIPE",
    "kueue_queue_name": "KUEUE_QUEUE_NAME",
    "gcs_scratch_location": "GCS_SCRATCH_LOCATION",
    "gcs_run_dir": "GCS_RUN_DIR",
    "trajectory_log_dir": "TRAJECTORY_LOG_DIR",
    "maxtext_ckpt": "MAXTEXT_CKPT",
    "maxtext_output_dir": "MAXTEXT_OUTPUT_DIR",
    "max_steps": "MAX_STEPS",
    "rollout_replicas": "ROLLOUT_REPLICAS",
    "rollout_tpu_slice": "ROLLOUT_TPU_SLICE",
    "trainer_tpu_slice": "TRAINER_TPU_SLICE",
    "wait_timeout_secs": "WAIT_TIMEOUT_SECS",
}


def _build_git_env(github_token: str) -> dict[str, str]:
  """Builds an env dict with GitHub HTTPS header auth if token is set."""
  env = os.environ.copy()
  env["GIT_TERMINAL_PROMPT"] = "0"
  if github_token:
    basic_auth = base64.b64encode(
        f"x-access-token:{github_token}".encode("utf-8")
    ).decode("ascii")
    env["GIT_CONFIG_COUNT"] = "1"
    env["GIT_CONFIG_KEY_0"] = "http.https://github.com/.extraheader"
    env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {basic_auth}"
  return env


def _conf_and_params(context: dict[str, Any]) -> tuple[dict, dict]:
  dag_run = context.get("dag_run")
  conf = (dag_run.conf or {}) if dag_run else {}
  params = context.get("params", {}) or {}
  return conf, params


def _callback_token(context: dict[str, Any]) -> str:
  """Caller-supplied GitHub token for the repository_dispatch callback.

  Read from `dag_run.conf` / Params at task runtime only. It is never copied
  into the run config returned by `prepare_run_config`, so it does not end up
  in XCom.
  """
  conf, params = _conf_and_params(context)
  return str(conf.get("github_token") or params.get("github_token") or "")


def _clone_token(context: dict[str, Any]) -> str:
  """Token used to clone the private google/trellis repository.

  Prefers an explicit caller-supplied `github_token`, otherwise the
  `GITHUB_PAT_TRELLIS_CI` Airflow Variable (read-only PAT), resolved at task
  runtime rather than at DAG-parse time.
  """
  return _callback_token(context) or safe_get_from_variable(
      GITHUB_PAT_TRELLIS_CI_VAR, ""
  )


@task
def validate_trigger() -> None:
  """Rejects manual UI runs and runs missing the required parameters.

  Unlike `xlml.utils.github.validate_git_trigger`, this does not require a
  GitHub token: callers that poll the Airflow REST API never send one.
  """
  context = get_current_context()
  dag_run = context.get("dag_run")
  if not dag_run or not dag_run.external_trigger:
    raise AirflowFailException(
        "This DAG should not be run manually from the Airflow UI."
    )
  conf, params = _conf_and_params(context)

  def _get(key: str) -> str:
    return str(conf.get(key) or params.get(key) or "")

  missing = [k for k in ("github_repo", "commit_sha") if not _get(k)]
  if missing:
    raise AirflowFailException(
        f"Missing required parameters: {', '.join(missing)}."
    )
  if not re.fullmatch(r"[0-9a-fA-F]{40}", _get("commit_sha")):
    raise AirflowFailException(
        "commit_sha must be a full 40-hex commit SHA so the run tests exactly"
        f" the commit it reports on; got {_get('commit_sha')!r}."
    )


@task
def prepare_run_config(**context: Any) -> dict[str, Any]:
  """Resolves DAG run parameters from `dag_run.conf` or Airflow Params.

  The returned dict is persisted as XCom and therefore must never contain a
  credential; tokens are resolved by `_clone_token` / `_callback_token` inside
  the tasks that need them.
  """
  conf, params = _conf_and_params(context)
  dag_run = context["dag_run"]

  def _get(key: str, default: Any) -> Any:
    val = conf.get(key, params.get(key, default))
    return default if val in (None, "") else val

  dag_run_id = dag_run.run_id if dag_run else "manual-run"
  commit_sha = str(_get("commit_sha", "HEAD"))
  short_sha = commit_sha[:7].lower()
  sanitized_run = re.sub(r"[^a-z0-9]+", "-", dag_run_id.lower()).strip("-")
  for prefix_head in (f"trellis-mh-{short_sha}-", f"tmh-{short_sha}-"):
    if sanitized_run.startswith(prefix_head):
      sanitized_run = sanitized_run[len(prefix_head) :]
  sanitized_run = sanitized_run[:8].strip("-")
  # Keep job_prefix <= 20 chars because JobSet's coordinator label on the
  # trainer is '<job_prefix>-train-proc-0-0.<job_prefix>-train'
  # (2 * len(job_prefix) + 22 <= 63 chars).
  default_prefix = f"tmh-{short_sha}-{sanitized_run}"[:20].strip("-")
  job_prefix = str(_get("job_prefix", default_prefix))[:20].strip("-")

  runner_env: dict[str, str] = {}
  image_uri = str(_get("image_uri", ""))
  if image_uri:
    runner_env["TUNIX_IMAGE"] = image_uri

  for conf_key, env_key in _OPTIONAL_CONF_TO_ENV.items():
    val = conf.get(conf_key, params.get(conf_key))
    if val not in (None, ""):
      runner_env[env_key] = str(val)

  if "verify_weights" in conf or "verify_weights" in params:
    verify_val = conf.get("verify_weights", params.get("verify_weights"))
    if verify_val is not None:
      runner_env["VERIFY_WEIGHTS"] = "true" if bool(verify_val) else "false"

  extra_env = _get("runner_env", {})
  if isinstance(extra_env, dict):
    for k, v in extra_env.items():
      if k and v is not None:
        runner_env[str(k)] = str(v)

  k8s_namespace = str(
      runner_env.get(
          "K8S_NAMESPACE", _get("k8s_namespace", DEFAULT_K8S_NAMESPACE)
      )
  )
  rollout_replicas = int(
      runner_env.get(
          "ROLLOUT_REPLICAS",
          _get("rollout_replicas", DEFAULT_ROLLOUT_REPLICAS),
      )
  )

  return {
      "dag_run_id": dag_run_id,
      "github_repo": str(_get("github_repo", "google/trellis")),
      "github_run_id": str(_get("github_run_id", dag_run_id)),
      # Whether a caller-supplied github_token exists; the token itself is
      # deliberately NOT stored here (XCom).
      "callback_enabled": bool(_callback_token(context)),
      "branch_ref": str(_get("branch_ref", "main")),
      "commit_sha": commit_sha,
      "image_uri": image_uri,
      "cluster_project": str(_get("cluster_project", DEFAULT_CLUSTER_PROJECT)),
      "cluster_zone": str(_get("cluster_zone", DEFAULT_CLUSTER_ZONE)),
      "cluster_name": str(_get("cluster_name", DEFAULT_CLUSTER_NAME)),
      "k8s_namespace": k8s_namespace,
      "rollout_replicas": rollout_replicas,
      "job_prefix": job_prefix,
      "runner_script": str(_get("runner_script", DEFAULT_RUNNER_SCRIPT)),
      "runner_env": runner_env,
  }


@task
def launch_and_verify_rl_jobsets(
    cfg: dict[str, Any], **context: Any
) -> dict[str, Any]:
  """Clones google/trellis, runs the multi-host E2E script, and verifies."""
  job_prefix = cfg["job_prefix"]
  github_repo = cfg["github_repo"]
  work_dir = tempfile.mkdtemp(prefix=f"{job_prefix}_")

  env = _build_git_env(_clone_token(context))
  repo_url = f"https://github.com/{github_repo}.git"
  subprocess.run(
      [
          "git",
          "clone",
          "--depth",
          "1",
          "-b",
          cfg["branch_ref"],
          repo_url,
          work_dir,
      ],
      env=env,
      check=True,
  )
  # Both must succeed: with check=False a failed fetch/checkout would silently
  # run the branch tip's runner and verifier while reporting results for
  # `commit_sha`, misattributing a pass or failure to the wrong commit.
  if cfg["commit_sha"] and cfg["commit_sha"] != "HEAD":
    subprocess.run(
        [
            "git",
            "-C",
            work_dir,
            "fetch",
            "--depth",
            "1",
            "origin",
            cfg["commit_sha"],
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        ["git", "-C", work_dir, "checkout", cfg["commit_sha"]],
        env=env,
        check=True,
    )

  log_dir = os.path.join(work_dir, "logs")
  os.makedirs(log_dir, exist_ok=True)
  env.update({
      "CLUSTER_PROJECT": cfg["cluster_project"],
      "CLUSTER_ZONE": cfg["cluster_zone"],
      "CLUSTER_NAME": cfg["cluster_name"],
      "K8S_NAMESPACE": cfg["k8s_namespace"],
      "COMMIT_SHA": cfg["commit_sha"],
      "JOB_PREFIX": cfg["job_prefix"],
      "LOG_OUTPUT_DIR": log_dir,
      "WAIT_FOR_COMPLETION": "true",
  })
  env.update(cfg.get("runner_env") or {})

  runner_script = os.path.join(
      work_dir, cfg.get("runner_script", DEFAULT_RUNNER_SCRIPT)
  )
  res = subprocess.run(["bash", runner_script], env=env, check=False)

  summary_path = os.path.join(log_dir, "verification_summary.json")
  summary: dict[str, Any] = {}
  if os.path.exists(summary_path):
    with open(summary_path, "r", encoding="utf-8") as f:
      summary = json.load(f)

  if res.returncode != 0 or not summary.get("passed", False):
    raise RuntimeError(
        f"Trellis Multi-Host RL E2E failed (exit={res.returncode}):"
        f" {json.dumps(summary)}"
    )
  return summary


@task(trigger_rule=TriggerRule.ALL_DONE)
def cleanup_rl_jobsets(cfg: dict[str, Any] | None) -> None:
  """Defense-in-depth cleanup of all GKE JobSets for this run."""
  if not cfg:
    logging.warning("Skipping GKE JobSet cleanup: no run config produced.")
    return
  cluster_zone = cfg["cluster_zone"]
  cluster_project = cfg["cluster_project"]
  job_prefix = cfg["job_prefix"]
  location_flag = (
      f"--zone={cluster_zone}"
      if re.search(r"-[a-z]$", cluster_zone)
      else f"--region={cluster_zone}"
  )
  subprocess.run(
      [
          "gcloud",
          "container",
          "clusters",
          "get-credentials",
          cfg["cluster_name"],
          location_flag,
          f"--project={cluster_project}",
      ],
      check=False,
  )
  replicas = max(
      int(cfg.get("rollout_replicas", DEFAULT_ROLLOUT_REPLICAS)),
      DEFAULT_ROLLOUT_REPLICAS,
  )
  jobsets = [
      f"{job_prefix}-orch",
      f"{job_prefix}-train",
      f"{job_prefix}-roll",
  ] + [f"{job_prefix}-roll-{i}" for i in range(replicas)]
  subprocess.run(
      [
          "kubectl",
          "delete",
          "jobset",
          "-n",
          cfg["k8s_namespace"],
          "--ignore-not-found=true",
          *jobsets,
      ],
      check=False,
  )


@task(trigger_rule=TriggerRule.ALL_DONE)
def fire_github_callback(
    cfg: dict[str, Any] | None,
    verification_summary: dict[str, Any] | None = None,
    **context: Any,
) -> None:
  """Optionally dispatches airflow-trellis-multi-host-rl-callback to GitHub.

  Opt-in: fires only when the caller supplied `github_token` in
  `dag_run.conf`. GitHub Actions callers poll the REST API instead, so for
  them this task is a no-op.
  """
  if not cfg or not cfg.get("callback_enabled"):
    logging.info(
        "Skipping GitHub callback: no caller-supplied github_token for run %s"
        " (callers are expected to poll the Airflow REST API).",
        cfg.get("dag_run_id") if cfg else "unknown",
    )
    return
  github_token = _callback_token(context)
  if not github_token:
    logging.warning("Skipping GitHub callback: github_token no longer present.")
    return

  dag_run = context.get("dag_run")
  task_instances = (
      dag_run.get_task_instances()
      if dag_run and hasattr(dag_run, "get_task_instances")
      else []
  )
  if task_instances:
    failed = any(
        ti.task_id == "launch_and_verify_rl_jobsets"
        and str(ti.state).lower()
        not in ("success", "taskinstancestate.success")
        for ti in task_instances
    )
  else:
    failed = not bool(
        verification_summary and verification_summary.get("passed", False)
    )
  overall_state = "failed" if failed else "success"

  webserver_base = safe_get_from_variable("COMPOSER_WEBSERVER_BASE_URL", "")
  dag_run_id = cfg["dag_run_id"]
  log_url = ""
  if webserver_base:
    log_url = (
        f"{webserver_base}/dags/trellis_multi_host_rl_e2e/grid"
        f"?dag_run_id={dag_run_id}"
    )

  # GitHub repository_dispatch enforces a maximum of 10 top-level properties in
  # client_payload; keep headroom below that limit.
  trigger_github_repository_dispatch.function(
      repo=cfg["github_repo"],
      token=github_token,
      event_type="airflow-trellis-multi-host-rl-callback",
      client_payload={
          "state": overall_state,
          "dag_run_id": cfg["dag_run_id"],
          "github_run_id": cfg["github_run_id"],
          "commit_sha": cfg["commit_sha"],
          "branch_ref": cfg["branch_ref"],
          "image_uri": cfg["image_uri"],
          "job_prefix": cfg["job_prefix"],
          "log_url": log_url,
          "verification_summary": verification_summary or {},
      },
  )


with models.DAG(
    dag_id="trellis_multi_host_rl_e2e",
    schedule=None,
    tags=[
        "trellis",
        "tunix",
        "maxtext",
        "raiden",
        "multi-host-rl",
        "tpu-v5p",
        "e2e",
    ],
    start_date=datetime.datetime(2026, 3, 1),
    catchup=False,
    max_active_runs=2,
    params={
        "github_repo": Param(
            default="google/trellis",
            type="string",
            description="GitHub repository in owner/repo format",
        ),
        "github_run_id": Param(
            default="manual",
            type="string",
            description="GitHub Actions run ID of the originating workflow",
        ),
        "github_token": Param(
            default="",
            type=["string", "null"],
            description=(
                "Optional GitHub token. When set, it is used to clone"
                " google/trellis and to fire the repository_dispatch callback;"
                " when empty (recommended), the clone uses the"
                f" {GITHUB_PAT_TRELLIS_CI_VAR} Airflow Variable and no callback"
                " is sent (callers poll the REST API instead)."
            ),
        ),
        "branch_ref": Param(
            default="main",
            type="string",
            description="Git branch or ref being tested",
        ),
        "commit_sha": Param(
            default="HEAD",
            type="string",
            description="Commit SHA being tested",
        ),
        "image_uri": Param(
            default="",
            type=["string", "null"],
            description=(
                "Optional candidate trellis-base image URI override (defaults"
                " to runner script's TUNIX_IMAGE in google/trellis)"
            ),
        ),
        "runner_script": Param(
            default=DEFAULT_RUNNER_SCRIPT,
            type="string",
            description="Relative path in google/trellis to E2E runner script",
        ),
        "runner_env": Param(
            default={},
            type="object",
            description=(
                "Optional dict of environment variable overrides passed to"
                " runner_script"
            ),
        ),
    },
) as dag:
  validate_task = validate_trigger()
  run_cfg = prepare_run_config()
  verified = launch_and_verify_rl_jobsets(run_cfg)
  cleaned = cleanup_rl_jobsets(run_cfg)
  callback = fire_github_callback(run_cfg, verified)

  chain(validate_task, run_cfg, verified, cleaned, callback)
