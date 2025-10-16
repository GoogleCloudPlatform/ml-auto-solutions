# Copyright 2024 Google LLC
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

"""Lists all currently broken tests."""

import dataclasses
import fnmatch
import logging
import os
from typing import Set

from airflow.models import Variable

from dags.common.test_owner import Team


@dataclasses.dataclass
class TestInfo:
  """Description of a flaky test."""

  owner: Team
  date_added: str
  details: str = ""


def parse_quarantine_patterns(quarantine_patterns_str: str) -> Set[str]:
  pattern_set = set()
  if len(quarantine_patterns_str.strip()) == 0:
    return pattern_set
  for pattern in quarantine_patterns_str.split("\n"):
    if len(pattern.strip()) > 0:
      pattern_set.add(pattern.strip().lower())  # Check capital?
  return pattern_set


def match_quarantine_patterns(
    test_name: str, quarantine_patterns_set: Set[str]
) -> bool:
  test_name_lower = test_name.lower()
  for pattern in quarantine_patterns_set:
    if fnmatch.fnmatch(test_name_lower, pattern):
      return True
  return False


def safe_get_from_variable(key: str, default_var: str):
  """
  Check whether the current runtime is GitHub Actions. Skip retrieving variables in GitHub Actions to avoid excessive log output.
  """
  value = default_var
  is_ci_env = os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
  if is_ci_env:
    logging.info("In GitHub Actions, skip getting variables")
  else:
    value = Variable.get(key, default_var=default_var)
  return value


"""
The quarantine list is defined by a set of UNIX Shell Glob Patterns. 
These patterns are used to match and quarantine tests. 

The patterns are stored in the Airflow Variable named 'quarantine_patterns'.

Sample values:
- maxtext-profiling-* (Matches all 'maxtext-profiling-' tests)
- maxd-sdxl-* (Matches all 'maxd-sdxl-' tests)
"""
quarantine_patterns = parse_quarantine_patterns(
    safe_get_from_variable("quarantine_patterns", "")
)


class QuarantineTests:
  """A list of currently-flaky tests."""

  tests = {
      # DAG: maxtext_gpu_end_to_end
      "maxtext-pinned-checkpoint-compatibility-h100-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-stable-checkpoint-compatibility-h100-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-pinned-checkpoint-compatibility-h100-mega-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-stable-checkpoint-compatibility-h100-mega-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-pinned-llama2-7b-train-1node-h100-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-pinned-llama2-7b-train-2node-h100-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-stable-llama2-7b-train-2node-h100-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-stable-llama2-7b-train-2node-h100-mega-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      "maxtext-pinned-llama2-7b-h100-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-01-17",
          "b/390738384",
      ),
      # DAG: maxtext_end_to_end
      "chained_tests_gemma-7b_stable": TestInfo(Team.LLM_DEVX, "2024-11-12"),
      "chained_tests_gemma-7b_nightly": TestInfo(Team.LLM_DEVX, "2024-11-12"),
      "chained_tests_llama2-70b_stable": TestInfo(Team.LLM_DEVX, "2024-11-12"),
      "chained_tests_llama2-70b_nightly": TestInfo(Team.LLM_DEVX, "2024-11-12"),
      # DAG: jax_stable_stack_gpu_e2e
      "maxtext-stable-stack-train-c4-data-h100-80gb-8": TestInfo(
          Team.JAX_MODELS_AND_PERFORMANCE, "2024-11-12"
      ),
      "maxtext-stable-stack-train-c4-data-h100-mega-80gb-8": TestInfo(
          Team.JAX_MODELS_AND_PERFORMANCE, "2024-11-12"
      ),
      "axlearn-jax-stable-stack-v4-16-2x-2xv4-16": TestInfo(
          Team.JAX_MODELS_AND_PERFORMANCE, "2024-11-12"
      ),
      # DAG: maxtext_configs_aot
      "maxtext-aot-v5e-stable-v4-8": TestInfo(Team.PERFORMANCE, "2024-11-12"),
      "maxtext-aot-v5e-nightly-v4-8": TestInfo(Team.PERFORMANCE, "2024-11-12"),
      # DAG: maxtext_configs_aot_hybridsim
      "16b-1xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "16b-2xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "16b-4xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "16b-8xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "32b-1xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "32b-2xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "32b-4xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "32b-8xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "64b-1xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "64b-2xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "64b-4xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "64b-8xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "128b-1xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "128b-2xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "128b-4xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "128b-8xv5litepod-256-aot-hybridsim": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      # DAG: mxla_gpt_6b_nightly_gke
      "mxla-gpt3-6b-nightly-gke-v5p-8": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "mxla-gpt3-6b-nightly-gke-2xv5p-8": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "mxla-gpt3-6b-nightly-gke-4xv5p-8": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "mxla-gpt3-6b-nightly-gke-8xv5p-8": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      # DAG: maxtext_trillium_configs_perf
      "maxtext-llama2_70b_4096-stable-3-2xv6e-256": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      "maxtext-llama2_70b_4096-nightly-3-2xv6e-256": TestInfo(
          Team.PERFORMANCE, "2024-11-12"
      ),
      # DAG: maxtext_v5e_configs_perf
      "maxtext-16b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-16b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-16b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-16b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-stable-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-stable-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-stable-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-stable-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-16b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-16b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-16b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-16b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-32b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-64b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-128b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-gpt3_175b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_7b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_13b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-nightly-0-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-nightly-2-v5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      "maxtext-llama2_70b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PERFORMANCE, "2024-11-13"
      ),
      # DAG: pathways_maxtext_v5e_configs_perf
      "p-maxtext-16b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-16b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-16b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-16b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-stable-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-stable-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-stable-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-stable-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-16b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-16b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-16b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-16b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-32b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-64b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-128b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-gpt3_175b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_7b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_13b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-nightly-0-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-nightly-1-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-nightly-2-v5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      "p-maxtext-llama2_70b-nightly-3-2xv5litepod-256": TestInfo(
          Team.PRODUCTIVITY, "2024-11-13"
      ),
      # DAG: maxtext_gpu_end_to_end
      "maxtext-pinned-mixtral-8x7b-1node-h100-mega-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-05-07",
          "b/407807678",
      ),
      "maxtext-pinned-mixtral-8x7b-2node-h100-mega-80gb-8": TestInfo(
          Team.LLM_DEVX,
          "2025-05-07",
          "b/407807678",
      ),
  }

  @staticmethod
  def is_quarantined(test_name) -> bool:
    """
    Checks if a test is quarantined using both legacy and current methods.

    The legacy method checks if `test_name` is present in `QuarantineTests.tests`.
    The current method checks against a runtime quarantine list fetched from Airflow Variables
    (key: 'quarantine_list') using `is_in_runtime_quarantine_list()`.

    The test is considered quarantined if it's found by either method.
    """
    return test_name in QuarantineTests.tests or match_quarantine_patterns(
        test_name, quarantine_patterns
    )
