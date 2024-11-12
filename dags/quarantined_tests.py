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

from dags.test_owner import Team as team


class QuarantineTests:

  @staticmethod
  def test_info(owner, date_added, details=None):
    return {
        "owner": owner,
        "date_added": date_added,
        "details": details,
    }

  tests = {
      # DAG: maxtext_gpu_end_to_end
      "maxtext-pinned-train-c4-data-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-c4-data-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-train-c4-data-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-synthetic-data-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-synthetic-data-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-train-synthetic-data-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-flash-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-flash-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-train-flash-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-quarter-batch-size-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-quarter-batch-size-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-train-quarter-batch-size-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-int8-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-int8-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-train-int8-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-fp8-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-train-fp8-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-train-fp8-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-decode-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-decode-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-decode-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-decode-quarter-batch-size-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-decode-quarter-batch-size-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-decode-quarter-batch-size-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-generate-param-only-checkpoint-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-generate-param-only-checkpoint-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-generate-param-only-checkpoint-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-generate-param-only-checkpoint-int8-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-generate-param-only-checkpoint-int8-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-generate-param-only-checkpoint-int8-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-grain-checkpoint-determinism-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-grain-checkpoint-determinism-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-grain-checkpoint-determinism-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-checkpoint-compatibility-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-checkpoint-compatibility-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-checkpoint-compatibility-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-checkpoint-compatibility-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-llama2-7b-train-1node-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-llama2-7b-train-1node-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-llama2-7b-train-1node-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-llama2-7b-train-1node-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-llama2-7b-train-2node-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-llama2-7b-train-2node-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-llama2-7b-train-2node-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-llama2-7b-train-2node-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-llama2-7b-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-llama2-7b-h100-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-pinned-llama2-7b-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
      "maxtext-stable-llama2-7b-h100-mega-80gb-8": test_info(
          team.LLM_DEVX, "2024-11-11"
      ),
  }

  @staticmethod
  def is_quarantined(test_name) -> bool:
    return test_name in QuarantineTests.tests
