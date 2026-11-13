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
  2. Stages the `google/trellis` runner and recipe files into `work_dir` -- by
     extracting `/app` from the candidate `image_uri` in Artifact Registry when
     `image_uri` is provided (so no GitHub credential is needed in Composer),
     or by cloning `google/trellis` at `commit_sha` otherwise -- and executes
     `runner_script` (forwarding any optional overrides in `dag_run.conf` or
     `runner_env`).
  3. Cleans up all GKE JobSets for `job_prefix` with `TriggerRule.ALL_DONE`.
  4. Optionally fires a GitHub `repository_dispatch`
     (`airflow-trellis-multi-host-rl-callback`) event via
     `xlml.utils.github.trigger_github_repository_dispatch` with
     `TriggerRule.ALL_DONE` -- only when the caller explicitly supplies
     `github_token` in `dag_run.conf`. GitHub Actions callers are expected to
     poll the Airflow REST API instead of relying on the callback:
       * DAG run state (`GET /api/v1/dags/{dag_id}/dagRuns/{run_id}`): the
         `watcher` leaf fails the run whenever any task fails, so the run
         state is `failed` for a failed E2E (not just for a failed callback).
       * Verifier summary: XCom key `verification_summary` of
         `launch_and_verify_rl_jobsets`, pushed on both pass and fail (also
         the task's `return_value` on pass). Callers must require BOTH run
         state `success` AND `verification_summary.passed == true`.
  5. `watcher` (`TriggerRule.ONE_FAILED`) is the final leaf that turns any
     task failure into a failed DAG run.

Credentials: when `image_uri` is supplied (standard for `google/trellis` CI),
the runner scripts and `docker/base/lkg.env` are extracted directly from the
`/app` layer of `image_uri` in Artifact Registry using the Composer worker's
Google Cloud identity, requiring no GitHub credential. When `image_uri` is
omitted, the repository clone falls back to the `GITHUB_PAT_TRELLIS_CI` Airflow
Variable (read-only PAT), read at task runtime inside the git subprocess
environment only (never in the runner's environment, Param defaults, or XCom).
A caller-supplied `github_token` (opt-in callback) is read from `dag_run.conf`
at task runtime and is not copied into XCom; note that `dag_run.conf` itself is
stored in the Airflow metadata database and is visible to anyone with DAG-run
read access (REST API and UI), so callers that opt in expose that token to
them -- REST-polling callers should not send one at all.
"""

import base64
import datetime
import json
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

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
# google/trellis repository (used only when `image_uri` is not supplied).
# Resolved inside tasks at runtime (see `_clone_token`), never at DAG-parse
# time, so the secret is not baked into Param defaults, rendered templates, or
# XCom.
GITHUB_PAT_TRELLIS_CI_VAR = "GITHUB_PAT_TRELLIS_CI"

_AR_IMAGE_URI_RE = re.compile(
    r"^(?P<host>[a-z0-9-]+-docker\.pkg\.dev)/(?P<repo>[^:@]+?)"
    r"(?::(?P<tag>[^@:]+))?(?:@(?P<digest>sha256:[0-9a-fA-F]{64}))?$"
)
_OCI_MANIFEST_ACCEPT = (
    "application/vnd.oci.image.manifest.v1+json, "
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.oci.image.index.v1+json, "
    "application/vnd.docker.distribution.manifest.list.v2+json"
)
# Only these repository-relative prefixes inside `/app` are needed on the
# Composer worker to launch and verify the multi-host E2E test.
_STAGED_APP_PREFIXES = ("tests/multi_host", "docker/base/lkg.env")

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


def _ar_get(url: str, token: str, accept: str | None = None) -> bytes:
  """Performs an authenticated GET against Artifact Registry."""
  headers = {"Authorization": f"Bearer {token}"}
  if accept:
    headers["Accept"] = accept
  req = urllib_request.Request(url, headers=headers)
  try:
    with urllib_request.urlopen(req, timeout=60) as resp:
      return resp.read()
  except urllib_error.HTTPError as exc:
    raise AirflowFailException(
        f"Artifact Registry GET {url} failed with HTTP {exc.code}:"
        f" {exc.reason}."
    ) from exc


def _normalize_runner_script(runner_script: str) -> str:
  """Validates and normalizes `runner_script` as a relative path."""
  raw = (runner_script or "").strip()
  if not raw or os.path.isabs(raw):
    raise AirflowFailException(
        "runner_script must be a non-empty relative path inside the"
        f" repository; got {runner_script!r}."
    )
  norm = os.path.normpath(raw)
  if (
      norm in ("", ".")
      or os.path.isabs(norm)
      or norm == ".."
      or norm.startswith(f"..{os.sep}")
  ):
    raise AirflowFailException(
        "runner_script must stay inside the repository work_dir;"
        f" got {runner_script!r}."
    )
  return norm


def _stage_work_dir_from_image(
    image_uri: str,
    work_dir: str,
    runner_script: str = DEFAULT_RUNNER_SCRIPT,
) -> None:
  """Extracts test files from an Artifact Registry `trellis-base` image.

  `docker/base/Dockerfile` copies the repository checkout into `/app` (`COPY .
  /app`), including `tests/multi_host/*` and the resolved `docker/base/lkg.env`
  for the build under test. Downloading only that single layer (~4 MB) via the
  Artifact Registry HTTP v2 API avoids pulling the full multi-GB image or
  requiring a GitHub PAT on the Composer worker.

  Args:
    image_uri: Artifact Registry image URI (`<region>-docker.pkg.dev/...`).
    work_dir: Destination directory on the Composer worker.
    runner_script: Relative path to the runner script inside the repository.
  """
  norm_runner = _normalize_runner_script(runner_script)
  match = _AR_IMAGE_URI_RE.match(image_uri.strip())
  if not match:
    raise AirflowFailException(
        f"Unsupported Artifact Registry image_uri: {image_uri!r}."
    )
  host = match.group("host")
  repo = match.group("repo")
  ref = match.group("digest") or match.group("tag") or "latest"
  token = subprocess.run(
      ["gcloud", "auth", "print-access-token"],
      capture_output=True,
      text=True,
      check=True,
  ).stdout.strip()
  base_url = f"https://{host}/v2/{repo}"
  manifest = json.loads(
      _ar_get(f"{base_url}/manifests/{ref}", token, _OCI_MANIFEST_ACCEPT)
  )
  if "manifests" in manifest:
    manifests = manifest["manifests"]
    if not manifests:
      raise AirflowFailException(
          f"Manifest index for {image_uri} contains no image manifests."
      )
    amd64_entry = next(
        (
            entry
            for entry in manifests
            if (entry.get("platform") or {}).get("architecture") == "amd64"
        ),
        manifests[0],
    )
    amd64_digest = amd64_entry["digest"]
    manifest = json.loads(
        _ar_get(
            f"{base_url}/manifests/{amd64_digest}",
            token,
            _OCI_MANIFEST_ACCEPT,
        )
    )
  config_digest = manifest["config"]["digest"]
  config = json.loads(_ar_get(f"{base_url}/blobs/{config_digest}", token))
  non_empty_history = [
      entry
      for entry in config.get("history", [])
      if not entry.get("empty_layer")
  ]
  layers = manifest.get("layers", [])
  if len(non_empty_history) != len(layers):
    raise AirflowFailException(
        f"Image {image_uri} history/layer count mismatch:"
        f" {len(non_empty_history)} non-empty history entries vs"
        f" {len(layers)} layers."
    )
  app_layer_digest = None
  for idx in range(len(layers) - 1, -1, -1):
    created_by = non_empty_history[idx].get("created_by") or ""
    if "COPY . /app" in created_by:
      app_layer_digest = layers[idx]["digest"]
      break
  if not app_layer_digest:
    raise AirflowFailException(
        f"Could not locate the COPY . /app layer in {image_uri}."
    )
  allowed_prefixes = (*_STAGED_APP_PREFIXES, norm_runner)
  blob = _ar_get(f"{base_url}/blobs/{app_layer_digest}", token)
  with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
    tmp_tar.write(blob)
    tmp_tar.flush()
    with tarfile.open(tmp_tar.name, "r:*") as archive:
      for member in archive.getmembers():
        norm_path = os.path.normpath(member.name).lstrip("/")
        if not norm_path.startswith("app/"):
          continue
        rel_path = norm_path[4:]
        if not rel_path or ".." in rel_path.split("/"):
          continue
        if not any(
            rel_path == prefix or rel_path.startswith(f"{prefix}/")
            for prefix in allowed_prefixes
        ):
          continue
        dest_path = os.path.join(work_dir, rel_path)
        if member.isdir():
          os.makedirs(dest_path, exist_ok=True)
        elif member.isfile():
          os.makedirs(os.path.dirname(dest_path), exist_ok=True)
          extracted = archive.extractfile(member)
          if extracted is not None:
            with extracted, open(dest_path, "wb") as dst_file:
              shutil.copyfileobj(extracted, dst_file)
            os.chmod(dest_path, member.mode & 0o777)
  staged_runner = os.path.join(work_dir, norm_runner)
  if not os.path.isfile(staged_runner):
    raise AirflowFailException(
        f"Runner script {norm_runner} was not found in /app of {image_uri}."
    )


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
  runtime rather than at DAG-parse time. Fails fast with an actionable
  message when neither is set: an anonymous clone of the private repository
  only yields git's generic "could not read Username" error.
  """
  token = _callback_token(context) or safe_get_from_variable(
      GITHUB_PAT_TRELLIS_CI_VAR, ""
  )
  if not token:
    raise AirflowFailException(
        f"No GitHub token available to clone the private repository. Set the"
        f" {GITHUB_PAT_TRELLIS_CI_VAR} Airflow Variable (read-only PAT for"
        " google/trellis) or pass github_token in dag_run.conf."
    )
  return token


@task
def validate_trigger() -> None:
  """Rejects scheduler-created runs and runs missing required parameters.

  `external_trigger` is True for both REST API and Airflow UI runs, so the
  first check only rejects scheduler-created runs (this DAG has
  `schedule=None`). A default UI run is rejected by the `commit_sha` check:
  the DAG must be given the exact commit to test. Unlike
  `xlml.utils.github.validate_git_trigger`, no GitHub token is required:
  callers that poll the Airflow REST API never send one.
  """
  context = get_current_context()
  dag_run = context.get("dag_run")
  if not dag_run or not dag_run.external_trigger:
    raise AirflowFailException(
        "This DAG must be triggered externally (REST API,"
        " TriggerDagRunOperator, or the UI trigger form) with explicit"
        " parameters; scheduler-created runs are not supported."
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
  """Stages google/trellis, runs the multi-host E2E script, and verifies.

  When `cfg["image_uri"]` is set (standard for `google/trellis` CI), the runner
  scripts and `docker/base/lkg.env` are extracted directly from the `/app`
  layer of `image_uri` in Artifact Registry using the Composer worker's Google
  Cloud credentials, so no GitHub token is required. Otherwise, falls back to
  cloning `google/trellis` with `_clone_token`.

  The verifier summary is pushed to XCom key `verification_summary` on both
  pass and fail (with the runner's exit code added), so REST pollers can show
  what failed; the task itself fails unless the runner exited 0 AND the
  verifier reported `passed: true`.
  """
  job_prefix = cfg["job_prefix"]
  github_repo = cfg["github_repo"]
  repo_url = f"https://github.com/{github_repo}.git"
  image_uri = (cfg.get("image_uri") or "").strip()
  raw_runner_script = cfg.get("runner_script")
  rel_runner_script = _normalize_runner_script(
      DEFAULT_RUNNER_SCRIPT
      if raw_runner_script is None
      else str(raw_runner_script)
  )

  # TemporaryDirectory removes the staged files, logs/ and downloaded
  # trajectories from the Composer worker when the task ends; the summary is
  # read before the block exits.
  with tempfile.TemporaryDirectory(prefix=f"{job_prefix}_") as work_dir:
    if image_uri:
      logging.info(
          "Staging %s test files from Artifact Registry image %s.",
          github_repo,
          image_uri,
      )
      _stage_work_dir_from_image(image_uri, work_dir, rel_runner_script)
    else:
      # The clone token lives only in the environment of the three git
      # subprocesses. The runner (code from `commit_sha`) and everything it
      # spawns get a clean environment: the http.extraheader would otherwise be
      # inherited and sent on every https://github.com/ request the runner
      # makes.
      git_env = _build_git_env(_clone_token(context))
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
          env=git_env,
          check=True,
      )
      # Both must succeed: with check=False a failed fetch/checkout would
      # silently run the branch tip's runner and verifier while reporting
      # results for `commit_sha`, misattributing a pass or failure to the wrong
      # commit. (validate_trigger already requires a full SHA; the "HEAD" guard
      # is defense in depth.)
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
            env=git_env,
            check=True,
        )
        subprocess.run(
            ["git", "-C", work_dir, "checkout", cfg["commit_sha"]],
            env=git_env,
            check=True,
        )

    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    runner_env = os.environ.copy()
    runner_env.update({
        "CLUSTER_PROJECT": cfg["cluster_project"],
        "CLUSTER_ZONE": cfg["cluster_zone"],
        "CLUSTER_NAME": cfg["cluster_name"],
        "K8S_NAMESPACE": cfg["k8s_namespace"],
        "COMMIT_SHA": cfg["commit_sha"],
        "JOB_PREFIX": cfg["job_prefix"],
        "LOG_OUTPUT_DIR": log_dir,
        "WAIT_FOR_COMPLETION": "true",
    })
    runner_env.update(cfg.get("runner_env") or {})

    runner_script = os.path.join(work_dir, rel_runner_script)
    res = subprocess.run(["bash", runner_script], env=runner_env, check=False)

    summary_path = os.path.join(log_dir, "verification_summary.json")
    summary: dict[str, Any] = {}
    if os.path.exists(summary_path):
      with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

  summary["runner_exit_code"] = res.returncode
  ti = context.get("ti")
  if ti is not None:
    ti.xcom_push(key="verification_summary", value=summary)

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


@task(trigger_rule=TriggerRule.ONE_FAILED, retries=0)
def watcher() -> None:
  """Fails the DAG run when any upstream task failed (see module docstring)."""
  raise AirflowFailException("An upstream task failed.")


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
                " google/trellis (when image_uri is omitted) and to fire the"
                " repository_dispatch callback; when empty (recommended),"
                " image_uri is staged from Artifact Registry and no callback"
                " is sent (callers poll the REST API instead)."
            ),
        ),
        "branch_ref": Param(
            default="main",
            type="string",
            description="Git branch or ref being tested",
        ),
        "commit_sha": Param(
            default="",
            type=["string", "null"],
            description=(
                "Full 40-hex commit SHA of google/trellis to test (required;"
                " validate_trigger rejects anything else)"
            ),
        ),
        "image_uri": Param(
            default="",
            type=["string", "null"],
            description=(
                "Candidate trellis-base image URI in Artifact Registry. When"
                " set, the runner scripts and lkg.env are staged directly from"
                " the image's /app layer without needing a GitHub token."
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

  # Airflow derives the DAG run state from its leaf tasks only. Without this,
  # the ALL_DONE callback (which succeeds even when it no-ops) would be the
  # sole leaf and a failed E2E -- or a run rejected by validate_trigger --
  # would still end as a `success` DAG run. The watcher fails whenever any
  # upstream task failed and is skipped on green runs, so REST pollers and the
  # UI see the real outcome.
  others = list(dag.tasks)
  others >> watcher()
