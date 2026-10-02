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

Coordinates the multi-host Trellis end-to-end RL testing pipeline for GitHub CI
on Cloud TPU (`bodaborg-v5p-nap` in `europe-west4`) using the regional candidate
image `europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/trellis/trellis-base:<sha>`:
  1. Validates the external GitHub Actions trigger via `validate_git_trigger`.
  2. Launches and verifies the 3 disaggregated Kubernetes JobSets on GKE:
     - Orchestrator JobSet (`jobset.cpu.yaml`, CPU controller on :20000)
     - Trainer JobSet (`jobset.pathways.yaml`, MaxText on Pathways tpuv5:2x2x2)
     - Rollout JobSets (`jobset.tpu.yaml`, vLLM TPU inference replicas)
  3. Cleans up all GKE JobSets with `TriggerRule.ALL_DONE`.
  4. Fires a GitHub `repository_dispatch` (`airflow-trellis-multi-host-rl-callback`)
     event via `xlml.utils.github.trigger_github_repository_dispatch` with
     `TriggerRule.ALL_DONE` to gate `:latest` / `:lkg` promotion and the
     `deps/lkg-update` PR in `promote_or_alert_multi_host_rl.yml`.
"""

import datetime
import json
import os
import subprocess
import tempfile
from typing import Any

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.utils.trigger_rule import TriggerRule
from dags.common.quarantined_tests import safe_get_from_variable
from xlml.utils.github import (
    trigger_github_repository_dispatch,
    validate_git_trigger,
)

DEFAULT_CLUSTER_PROJECT = "cloud-tpu-shared-capacity"
DEFAULT_CLUSTER_ZONE = "europe-west4"
DEFAULT_CLUSTER_NAME = "bodaborg-v5p-nap"
DEFAULT_K8S_NAMESPACE = "default"
DEFAULT_GCS_SCRATCH = "gs://cloud-tpu-tunix-eu"
DEFAULT_MAXTEXT_CKPT = (
    "gs://niting-storage-europe-west4/qwen3.5-35b-a3b/scanned/0/items"
)
DEFAULT_IMAGE_REPO = (
    "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/trellis/trellis-base"
)


@task
def prepare_run_config(**context: Any) -> dict[str, Any]:
  """Resolves DAG run parameters from `dag_run.conf` or Airflow Params."""
  dag_run = context["dag_run"]
  conf = (dag_run.conf or {}) if dag_run else {}
  params = context.get("params", {})

  def _get(key: str, default: Any) -> Any:
    return conf.get(key, params.get(key, default))

  dag_run_id = dag_run.run_id if dag_run else "manual-run"
  commit_sha = str(_get("commit_sha", "HEAD"))
  short_sha = commit_sha[:7]
  sanitized_run = "".join(
      c if c.isalnum() else "-" for c in dag_run_id.lower()
  )[:22].strip("-")
  job_prefix = f"trellis-mh-{short_sha}-{sanitized_run}"[:42].strip("-")

  gcs_scratch = str(_get("gcs_scratch_location", DEFAULT_GCS_SCRATCH)).rstrip(
      "/"
  )
  gcs_run_dir = f"{gcs_scratch}/trellis_ci_runs/{job_prefix}"
  trajectory_log_dir = f"{gcs_run_dir}/trajectories"
  maxtext_output_dir = f"{gcs_run_dir}/maxtext"

  return {
      "dag_run_id": dag_run_id,
      "github_repo": str(_get("github_repo", "google/trellis")),
      "github_run_id": str(_get("github_run_id", "")),
      "github_token": str(
          _get(
              "github_token",
              safe_get_from_variable("GITHUB_PAT_TRELLIS_CI", ""),
          )
      ),
      "branch_ref": str(_get("branch_ref", "main")),
      "commit_sha": commit_sha,
      "image_uri": str(
          _get("image_uri", f"{DEFAULT_IMAGE_REPO}:{commit_sha}")
      ),
      "deployment_id": str(_get("deployment_id", "")),
      "is_lkg_sweep": bool(_get("is_lkg_sweep", True)),
      "candidate_lkg_pins": dict(_get("candidate_lkg_pins", {})),
      "max_steps": int(_get("max_steps", 10)),
      "rollout_replicas": int(_get("rollout_replicas", 2)),
      "rollout_tpu_slice": str(_get("rollout_tpu_slice", "tpuv5:2x2x2")),
      "trainer_tpu_slice": str(_get("trainer_tpu_slice", "tpuv5:2x2x2")),
      "maxtext_ckpt": str(
          _get("maxtext_ckpt", DEFAULT_MAXTEXT_CKPT) or DEFAULT_MAXTEXT_CKPT
      ),
      "maxtext_output_dir": str(
          _get("maxtext_output_dir", maxtext_output_dir) or maxtext_output_dir
      ),
      "verify_weights": bool(_get("verify_weights", True)),
      "wait_timeout_secs": int(_get("wait_timeout_secs", 5400)),
      "cluster_project": str(_get("cluster_project", DEFAULT_CLUSTER_PROJECT)),
      "cluster_zone": str(_get("cluster_zone", DEFAULT_CLUSTER_ZONE)),
      "cluster_name": str(_get("cluster_name", DEFAULT_CLUSTER_NAME)),
      "k8s_namespace": str(_get("k8s_namespace", DEFAULT_K8S_NAMESPACE)),
      "job_prefix": job_prefix,
      "gcs_scratch_location": gcs_scratch,
      "gcs_run_dir": gcs_run_dir,
      "trajectory_log_dir": trajectory_log_dir,
  }


@task
def launch_and_verify_rl_jobsets(cfg: dict[str, Any]) -> dict[str, Any]:
  """Clones google/trellis, executes tests/multi_host/run_multi_host_gsm8k_e2e.sh, and returns verification summary."""
  work_dir = tempfile.mkdtemp(prefix=f"{cfg['job_prefix']}_")
  log_dir = os.path.join(work_dir, "logs")
  os.makedirs(log_dir, exist_ok=True)

  repo_url = f"https://github.com/{cfg['github_repo']}.git"
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
      check=True,
  )
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
        check=False,
    )
    subprocess.run(
        ["git", "-C", work_dir, "checkout", cfg["commit_sha"]],
        check=False,
    )

  env = os.environ.copy()
  env.update({
      "CLUSTER_PROJECT": cfg["cluster_project"],
      "CLUSTER_ZONE": cfg["cluster_zone"],
      "CLUSTER_NAME": cfg["cluster_name"],
      "K8S_NAMESPACE": cfg["k8s_namespace"],
      "TUNIX_IMAGE": cfg["image_uri"],
      "COMMIT_SHA": cfg["commit_sha"],
      "JOB_PREFIX": cfg["job_prefix"],
      "GCS_SCRATCH_LOCATION": cfg["gcs_scratch_location"],
      "GCS_RUN_DIR": cfg["gcs_run_dir"],
      "TRAJECTORY_LOG_DIR": cfg["trajectory_log_dir"],
      "MAXTEXT_CKPT": cfg["maxtext_ckpt"],
      "MAXTEXT_OUTPUT_DIR": cfg["maxtext_output_dir"],
      "LOG_OUTPUT_DIR": log_dir,
      "MAX_STEPS": str(cfg["max_steps"]),
      "ROLLOUT_REPLICAS": str(cfg["rollout_replicas"]),
      "ROLLOUT_TPU_SLICE": cfg["rollout_tpu_slice"],
      "TRAINER_TPU_SLICE": cfg["trainer_tpu_slice"],
      "VERIFY_WEIGHTS": "true" if cfg["verify_weights"] else "false",
      "WAIT_FOR_COMPLETION": "true",
      "WAIT_TIMEOUT_SECS": str(cfg["wait_timeout_secs"]),
  })

  runner_script = os.path.join(
      work_dir, "tests", "multi_host", "run_multi_host_gsm8k_e2e.sh"
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
def cleanup_rl_jobsets(cfg: dict[str, Any]) -> None:
  """Defense-in-depth cleanup of all GKE JobSets for this run."""
  subprocess.run(
      [
          "gcloud",
          "container",
          "clusters",
          "get-credentials",
          cfg["cluster_name"],
          f"--zone={cfg['cluster_zone']}",
          f"--project={cfg['cluster_project']}",
      ],
      check=False,
  )
  jobsets = [
      f"{cfg['job_prefix']}-orch",
      f"{cfg['job_prefix']}-train",
      f"{cfg['job_prefix']}-roll",
  ] + [
      f"{cfg['job_prefix']}-roll-{i}"
      for i in range(int(cfg["rollout_replicas"]))
  ]
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
    cfg: dict[str, Any],
    verification_summary: dict[str, Any] | None = None,
    **context: Any,
) -> None:
  """Dispatches `airflow-trellis-multi-host-rl-callback` via xlml.utils.github."""
  dag_run = context["dag_run"]
  task_instances = dag_run.get_task_instances() if dag_run else []
  failed = any(
      ti.task_id == "launch_and_verify_rl_jobsets" and ti.state != "success"
      for ti in task_instances
  )
  overall_state = "failed" if failed else "success"

  webserver_base = safe_get_from_variable("COMPOSER_WEBSERVER_BASE_URL", "")
  log_url = (
      f"{webserver_base}/dags/trellis_multi_host_rl_e2e/grid?dag_run_id={cfg['dag_run_id']}"
      if webserver_base
      else ""
  )

  trigger_github_repository_dispatch.function(
      repo=cfg["github_repo"],
      token=cfg["github_token"],
      event_type="airflow-trellis-multi-host-rl-callback",
      client_payload={
          "state": overall_state,
          "dag_id": "trellis_multi_host_rl_e2e",
          "dag_run_id": cfg["dag_run_id"],
          "github_run_id": cfg["github_run_id"],
          "deployment_id": cfg["deployment_id"],
          "commit_sha": cfg["commit_sha"],
          "branch_ref": cfg["branch_ref"],
          "image_uri": cfg["image_uri"],
          "is_lkg_sweep": cfg["is_lkg_sweep"],
          "candidate_lkg_pins": cfg["candidate_lkg_pins"],
          "gcs_run_dir": cfg["gcs_run_dir"],
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
            default="",
            type="string",
            description="GitHub Actions run ID of the originating workflow",
        ),
        "github_token": Param(
            default="",
            type="string",
            description=(
                "GitHub PAT used to fire the repository_dispatch callback"
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
            default=f"{DEFAULT_IMAGE_REPO}:latest",
            type="string",
            description="Candidate trellis-base Docker image URI",
        ),
        "deployment_id": Param(
            default="",
            type="string",
            description="Optional GitHub Deployment ID for status reporting",
        ),
        "is_lkg_sweep": Param(
            default=True,
            type="boolean",
            description="Whether to promote :lkg and open deps/lkg-update PR",
        ),
        "candidate_lkg_pins": Param(
            default={},
            type="object",
            description="Candidate upstream commit pins built into image_uri",
        ),
        "max_steps": Param(
            default=10,
            type="integer",
            description="Number of distributed GRPO training steps to run",
        ),
        "rollout_replicas": Param(
            default=2,
            type="integer",
            description="Number of vLLM TPU rollout JobSet slices",
        ),
        "rollout_tpu_slice": Param(
            default="tpuv5:2x2x2",
            type="string",
            description="TPU slice topology per rollout replica",
        ),
        "trainer_tpu_slice": Param(
            default="tpuv5:2x2x2",
            type="string",
            description="TPU slice topology for Pathways MaxText trainer",
        ),
        "maxtext_ckpt": Param(
            default=DEFAULT_MAXTEXT_CKPT,
            type="string",
            description=(
                "GCS URI of pre-converted MaxText scanned Orbax checkpoint"
            ),
        ),
        "verify_weights": Param(
            default=True,
            type="boolean",
            description="Verify Raiden weight sync checksums across slices",
        ),
    },
) as dag:
  validate_task = validate_git_trigger(
      repo="{{ params.github_repo }}",
      token="{{ params.github_token }}",
      run_id="{{ params.github_run_id }}",
      commit_sha="{{ params.commit_sha }}",
  )
  run_cfg = prepare_run_config()
  verified = launch_and_verify_rl_jobsets(run_cfg)
  cleaned = cleanup_rl_jobsets(run_cfg)
  callback = fire_github_callback(run_cfg, verified)

  chain(validate_task, run_cfg, verified, cleaned, callback)
