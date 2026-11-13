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

"""Unit tests for the Trellis Multi-Host Distributed RL E2E DAG."""

import ast
import base64
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock
from urllib import error as urllib_error
from urllib import request as urllib_request

try:  # Behavioural tests need Airflow (available in CI via scripts/dag-check.sh).
  from airflow.exceptions import AirflowFailException
  from dags.multipod import trellis_multi_host_rl_e2e as dag_mod

  _AIRFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Airflow.
  AirflowFailException = None  # pylint: disable=invalid-name
  dag_mod = None
  _AIRFLOW_AVAILABLE = False


class _FallbackFailError(Exception):
  """Stand-in for AirflowFailException when Airflow is not installed."""


def _make_app_layer_blob(entries: dict[str, tuple[bytes, int]]) -> bytes:
  """Builds an in-memory gzipped tar layer for the given entries."""
  buf = io.BytesIO()
  with tarfile.open(fileobj=buf, mode="w:gz") as tf:
    for path, (data, mode) in entries.items():
      info = tarfile.TarInfo(name=path)
      info.size = len(data)
      info.mode = mode
      tf.addfile(info, io.BytesIO(data))
  return buf.getvalue()


class MultiHostRlE2eDagTest(unittest.TestCase):
  """Tests syntax, trigger rules, and staging helpers for the Trellis DAG."""

  def setUp(self):
    super().setUp()
    self.dags_dir = pathlib.Path(__file__).resolve().parent.parent
    self.trellis_dag = self.dags_dir / "trellis_multi_host_rl_e2e.py"

  def test_trellis_dag_syntax_and_all_done_trigger_rules(self):
    self.assertTrue(self.trellis_dag.exists())
    source = self.trellis_dag.read_text(encoding="utf-8")
    ast.parse(source)
    self.assertIn('dag_id="trellis_multi_host_rl_e2e"', source)
    self.assertIn("TriggerRule.ALL_DONE", source)
    self.assertIn("airflow-trellis-multi-host-rl-callback", source)
    self.assertIn("run_multi_host_gsm8k_e2e.sh", source)
    self.assertIn("bodaborg-v5p-nap", source)
    self.assertIn('DEFAULT_K8S_NAMESPACE = "trellis"', source)
    self.assertIn("DEFAULT_ROLLOUT_REPLICAS = 2", source)
    self.assertIn("_OPTIONAL_CONF_TO_ENV", source)
    self.assertIn('"recipe": "RECIPE"', source)
    self.assertIn('"runner_env"', source)
    self.assertIn('"runner_script"', source)
    self.assertNotIn("DEFAULT_MAXTEXT_CKPT", source)
    self.assertIn("--region={cluster_zone}", source)
    self.assertIn("--zone={cluster_zone}", source)
    self.assertIn("if not cfg:", source)
    self.assertIn("if not cfg or not cfg.get(", source)
    self.assertIn("_build_git_env", source)
    self.assertIn("_stage_work_dir_from_image", source)
    self.assertIn("http.https://github.com/.extraheader", source)
    self.assertIn("GITHUB_PAT_TRELLIS_CI", source)
    self.assertIn("[:20]", source)
    self.assertLess(
        source.index('"clone"'),
        source.index("os.makedirs(log_dir, exist_ok=True)"),
    )
    tree = ast.parse(source)
    payload_dicts = []
    for node in ast.walk(tree):
      if isinstance(node, ast.Call):
        for kw in node.keywords:
          if kw.arg == "client_payload" and isinstance(kw.value, ast.Dict):
            payload_dicts.append(kw.value)
    self.assertEqual(len(payload_dicts), 1)
    self.assertLessEqual(
        len(payload_dicts[0].keys),
        10,
        "GitHub repository_dispatch allows at most 10 top-level properties in"
        " client_payload",
    )

  def _function_node(self, name: str) -> ast.FunctionDef:
    tree = ast.parse(self.trellis_dag.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    self.assertEqual(len(nodes), 1, f"expected exactly one def {name}")
    return nodes[0]

  def test_git_fetch_and_checkout_of_commit_sha_must_succeed(self):
    """A failed fetch/checkout must fail the task, not silently test the tip."""
    func = self._function_node("launch_and_verify_rl_jobsets")
    git_calls = []
    for node in ast.walk(func):
      if not isinstance(node, ast.Call):
        continue
      if not (
          isinstance(node.func, ast.Attribute)
          and node.func.attr == "run"
          and isinstance(node.func.value, ast.Name)
          and node.func.value.id == "subprocess"
      ):
        continue
      if not node.args or not isinstance(node.args[0], ast.List):
        continue
      first = node.args[0].elts[0] if node.args[0].elts else None
      if isinstance(first, ast.Constant) and first.value == "git":
        git_calls.append(node)
    self.assertEqual(len(git_calls), 3, "expected clone, fetch, checkout")
    for call in git_calls:
      checks = [
          kw.value.value
          for kw in call.keywords
          if kw.arg == "check" and isinstance(kw.value, ast.Constant)
      ]
      self.assertEqual(checks, [True], ast.dump(call.args[0])[:80])

  def test_no_token_is_persisted_in_run_config_or_param_defaults(self):
    """Tokens are resolved inside tasks at runtime, never stored in XCom."""
    source = self.trellis_dag.read_text(encoding="utf-8")
    func = self._function_node("prepare_run_config")
    returned_keys = []
    for node in ast.walk(func):
      if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
        returned_keys.extend(
            k.value for k in node.value.keys if isinstance(k, ast.Constant)
        )
    self.assertIn("callback_enabled", returned_keys)
    self.assertNotIn("github_token", returned_keys)
    # The PAT must not be read at DAG-parse time into a Param default.
    self.assertNotIn('safe_get_from_variable("GITHUB_PAT_TRELLIS_CI"', source)
    self.assertIn('GITHUB_PAT_TRELLIS_CI_VAR = "GITHUB_PAT_TRELLIS_CI"', source)
    # Trigger validation no longer requires a token (REST-polling callers send none).
    self.assertNotIn("import validate_git_trigger", source)
    self.assertIn("def validate_trigger", source)
    self.assertIn("callback_enabled", source)

  def test_runner_env_never_contains_git_auth_header(self):
    """The clone PAT must be scoped to the git subprocesses only."""
    func = self._function_node("launch_and_verify_rl_jobsets")
    runner_calls = []
    for node in ast.walk(func):
      if not isinstance(node, ast.Call):
        continue
      if not (
          isinstance(node.func, ast.Attribute)
          and node.func.attr == "run"
          and node.args
          and isinstance(node.args[0], ast.List)
          and node.args[0].elts
          and isinstance(node.args[0].elts[0], ast.Constant)
          and node.args[0].elts[0].value == "bash"
      ):
        continue
      runner_calls.append(node)
    self.assertEqual(len(runner_calls), 1, "expected exactly one runner call")
    env_kw = [kw for kw in runner_calls[0].keywords if kw.arg == "env"]
    self.assertEqual(len(env_kw), 1)
    self.assertIsInstance(env_kw[0].value, ast.Name)
    self.assertEqual(env_kw[0].value.id, "runner_env")
    source = self.trellis_dag.read_text(encoding="utf-8")
    self.assertIn("runner_env = os.environ.copy()", source)
    self.assertIn("tempfile.TemporaryDirectory(", source)
    self.assertNotIn("tempfile.mkdtemp(", source)

  @unittest.skipUnless(_AIRFLOW_AVAILABLE, "requires apache-airflow")
  def test_validate_trigger_behaviour(self):
    def _validate(conf, external=True):
      ctx = {
          "dag_run": mock.Mock(external_trigger=external, conf=conf),
          "params": {"github_repo": "google/trellis", "commit_sha": ""},
      }
      with mock.patch.object(dag_mod, "get_current_context", return_value=ctx):
        return dag_mod.validate_trigger.function()

    full_sha = "a" * 40
    self.assertIsNone(_validate({"commit_sha": full_sha}))
    with self.assertRaisesRegex(AirflowFailException, "Missing required"):
      _validate({})
    with self.assertRaisesRegex(AirflowFailException, "40-hex"):
      _validate({"commit_sha": "abc1234"})
    with self.assertRaisesRegex(AirflowFailException, "triggered externally"):
      _validate({"commit_sha": full_sha}, external=False)

  @unittest.skipUnless(_AIRFLOW_AVAILABLE, "requires apache-airflow")
  def test_clone_token_fails_fast_without_any_token(self):
    ctx = {"dag_run": mock.Mock(conf={}), "params": {}}
    with mock.patch.object(dag_mod, "safe_get_from_variable", return_value=""):
      with self.assertRaisesRegex(
          AirflowFailException, "GITHUB_PAT_TRELLIS_CI"
      ):
        dag_mod._clone_token(ctx)  # pylint: disable=protected-access
    with mock.patch.object(
        dag_mod, "safe_get_from_variable", return_value="ghp_from_variable"
    ):
      self.assertEqual(
          dag_mod._clone_token(ctx),  # pylint: disable=protected-access
          "ghp_from_variable",
      )
    ctx_with_conf = {
        "dag_run": mock.Mock(conf={"github_token": "ghp_from_conf"}),
        "params": {},
    }
    self.assertEqual(
        dag_mod._clone_token(ctx_with_conf),  # pylint: disable=protected-access
        "ghp_from_conf",
    )

  @unittest.skipUnless(_AIRFLOW_AVAILABLE, "requires apache-airflow")
  def test_watcher_is_a_one_failed_leaf_downstream_of_every_other_task(self):
    """A failed task must fail the DAG run, not just the callback leaf."""
    dag = dag_mod.dag
    watcher = dag.get_task("watcher")
    self.assertEqual(watcher.trigger_rule, "one_failed")
    self.assertEqual(watcher.retries, 0)
    self.assertEqual(watcher.downstream_task_ids, set())
    others = {t.task_id for t in dag.tasks} - {"watcher"}
    self.assertEqual(watcher.upstream_task_ids, others)
    leaves = {t.task_id for t in dag.tasks if not t.downstream_task_ids}
    self.assertEqual(leaves, {"watcher"})

  def test_build_git_env_injects_auth_header_without_token_in_url(self):
    source = self.trellis_dag.read_text(encoding="utf-8")
    tree = ast.parse(source)
    func_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_git_env"
    ]
    self.assertEqual(len(func_nodes), 1)
    mod = ast.Module(body=func_nodes, type_ignores=[])
    namespace = {"base64": base64, "os": os}
    code = compile(mod, filename="<ast>", mode="exec")
    exec(code, namespace)  # pylint: disable=exec-used
    build_git_env = namespace["_build_git_env"]

    env_no_token = build_git_env("")
    self.assertEqual(env_no_token["GIT_TERMINAL_PROMPT"], "0")
    self.assertNotIn("GIT_CONFIG_VALUE_0", env_no_token)

    env_with_token = build_git_env("ghp_dummy_secret_123")
    expected_b64 = base64.b64encode(
        b"x-access-token:ghp_dummy_secret_123"
    ).decode("ascii")
    self.assertEqual(env_with_token["GIT_TERMINAL_PROMPT"], "0")
    self.assertEqual(env_with_token["GIT_CONFIG_COUNT"], "1")
    self.assertEqual(
        env_with_token["GIT_CONFIG_KEY_0"],
        "http.https://github.com/.extraheader",
    )
    self.assertEqual(
        env_with_token["GIT_CONFIG_VALUE_0"],
        f"AUTHORIZATION: basic {expected_b64}",
    )

  def _load_image_staging_helpers(self) -> dict[str, object]:
    source = self.trellis_dag.read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted_assigns = {
        "DEFAULT_RUNNER_SCRIPT",
        "_AR_IMAGE_URI_RE",
        "_OCI_MANIFEST_ACCEPT",
        "_STAGED_APP_PREFIXES",
    }
    wanted_funcs = {"_ar_get", "_stage_work_dir_from_image"}
    body = []
    for node in tree.body:
      if isinstance(node, ast.Assign):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if names & wanted_assigns:
          body.append(node)
      elif isinstance(node, ast.FunctionDef) and node.name in wanted_funcs:
        body.append(node)
    mod = ast.Module(body=body, type_ignores=[])
    namespace: dict[str, object] = {
        "AirflowFailException": AirflowFailException or _FallbackFailError,
        "json": json,
        "os": os,
        "re": re,
        "shutil": shutil,
        "subprocess": subprocess,
        "tarfile": tarfile,
        "tempfile": tempfile,
        "urllib_error": urllib_error,
        "urllib_request": urllib_request,
    }
    exec(compile(mod, filename="<ast>", mode="exec"), namespace)  # pylint: disable=exec-used
    return namespace

  def test_stage_work_dir_from_image_extracts_runner_and_lkg_env(self):
    ns = self._load_image_staging_helpers()
    stage_fn = ns["_stage_work_dir_from_image"]
    fail_exc = ns["AirflowFailException"]
    layer_blob = _make_app_layer_blob({
        "app/tests/multi_host/run_multi_host_gsm8k_e2e.sh": (
            b"#!/usr/bin/env bash\necho ok\n",
            0o755,
        ),
        "app/tests/multi_host/k8s_launcher.sh": (b"#!/bin/bash\n", 0o755),
        "app/docker/base/lkg.env": (b"TUNIX_COMMIT=deadbeef\n", 0o644),
        "app/experimental/unrelated.txt": (b"ignored\n", 0o644),
    })
    base = (
        "https://europe-west4-docker.pkg.dev/v2/"
        "cloud-tpu-multipod-dev/trellis/trellis-base"
    )
    index_doc = {
        "manifests": [{
            "digest": "sha256:" + "1" * 64,
            "platform": {"architecture": "amd64", "os": "linux"},
        }]
    }
    manifest_doc = {
        "config": {"digest": "sha256:" + "2" * 64},
        "layers": [
            {"digest": "sha256:" + "3" * 64},
            {"digest": "sha256:" + "4" * 64},
        ],
    }
    config_doc = {
        "history": [
            {"created_by": "RUN apt-get update"},
            {"created_by": "WORKDIR /app", "empty_layer": True},
            {"created_by": "COPY . /app # buildkit"},
        ]
    }
    zero_digest = "sha256:" + "1" * 64
    cfg_digest = "sha256:" + "2" * 64
    layer_digest = "sha256:" + "4" * 64
    responses = {
        f"{base}/manifests/mytag": json.dumps(index_doc).encode(),
        f"{base}/manifests/{zero_digest}": json.dumps(manifest_doc).encode(),
        f"{base}/blobs/{cfg_digest}": json.dumps(config_doc).encode(),
        f"{base}/blobs/{layer_digest}": layer_blob,
    }
    ns["_ar_get"] = lambda url, token, accept=None: responses[url]
    completed = subprocess.CompletedProcess(
        args=["gcloud"], returncode=0, stdout="mock_access_token\n", stderr=""
    )
    with tempfile.TemporaryDirectory() as work_dir:
      with mock.patch.object(subprocess, "run", return_value=completed):
        stage_fn(
            "europe-west4-docker.pkg.dev/cloud-tpu-multipod-dev/"
            "trellis/trellis-base:mytag",
            work_dir,
        )
      runner = os.path.join(
          work_dir, "tests", "multi_host", "run_multi_host_gsm8k_e2e.sh"
      )
      lkg = os.path.join(work_dir, "docker", "base", "lkg.env")
      unrelated = os.path.join(work_dir, "experimental", "unrelated.txt")
      self.assertTrue(os.path.isfile(runner))
      self.assertEqual(os.stat(runner).st_mode & 0o777, 0o755)
      self.assertTrue(os.path.isfile(lkg))
      self.assertFalse(os.path.exists(unrelated))

    with tempfile.TemporaryDirectory() as work_dir:
      with self.assertRaisesRegex(fail_exc, "Unsupported Artifact Registry"):
        stage_fn("gcr.io/not-artifact-registry/image:latest", work_dir)


if __name__ == "__main__":
  unittest.main()
