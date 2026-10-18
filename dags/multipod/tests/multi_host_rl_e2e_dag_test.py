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
import os
import pathlib
import unittest


class MultiHostRlE2eDagTest(unittest.TestCase):
  """Tests syntax, trigger rules, and git auth helpers for the Trellis DAG."""

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


if __name__ == "__main__":
  unittest.main()
