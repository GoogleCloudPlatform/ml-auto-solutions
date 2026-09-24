#!/usr/bin/env python3
import ast
import pathlib
import unittest


class MultiHostRlE2eDagTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.dags_dir = pathlib.Path(__file__).resolve().parent.parent

  def test_trellis_dag_syntax_and_all_done_trigger_rules(self):
    trellis_dag = self.dags_dir / "trellis_multi_host_rl_e2e.py"
    self.assertTrue(trellis_dag.exists())
    source = trellis_dag.read_text(encoding="utf-8")
    ast.parse(source)
    self.assertIn('dag_id="trellis_multi_host_rl_e2e"', source)
    self.assertIn("TriggerRule.ALL_DONE", source)
    self.assertIn("airflow-trellis-multi-host-rl-callback", source)
    self.assertIn("run_multi_host_gsm8k_e2e.sh", source)
    self.assertIn("bodaborg-v5p-nap", source)


if __name__ == "__main__":
  unittest.main()
