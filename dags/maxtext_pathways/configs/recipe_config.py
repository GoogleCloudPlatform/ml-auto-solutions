# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common recipe configs"""

import enum


class Recipe(enum.Enum):
  """
  An enumeration of known recipe names used for configuring training workload.

  These recipe names correspond to defined execution flows or scripts,
  often found in the referenced MaxText repository:
  https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/recipes
  """

  PW_MCJAX_BENCHMARK_RECIPE = "pw_mcjax_benchmark_recipe"

  @property
  def run_command(self) -> str:
    """
    Generates the complete command string to execute this recipe as a Python module.
    """
    return f"python3 -m benchmarks.recipes.{self.value}"


RECIPE_FLAG = [
    "user",
    "cluster_name",
    "project",
    "zone",
    "benchmark_steps",
    "num_slices_list",
    "server_image",
    "proxy_image",
    "runner",
    "selected_model_framework",
    "selected_model_names",
    "priority",
    "max_restarts",
    "bq_enable",
    "bq_db_project",
    "bq_db_dataset",
    "temp_key",
    "device_type",
]
