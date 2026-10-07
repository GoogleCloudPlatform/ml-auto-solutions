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
    self.assertIn("_build_git_env", source)
    self.assertIn("http.https://github.com/.extraheader", source)
    self.assertIn("GITHUB_PAT_TRELLIS_CI", source)
    self.assertLess(
        source.index('"clone"'),
        source.index("os.makedirs(log_dir, exist_ok=True)"),
    )

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
