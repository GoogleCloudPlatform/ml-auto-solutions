# Copyright 2025 Google LLC
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

"""Sample job to run Aotc reproducibility benchmarks."""
import sys
import os

# Skip execution when being run as part of the DAG check
# Checking if the file doesn't exist is a reliable way to detect this context
base_recipe_repo_root = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "internal-gpu-recipes",
    )
)

if not os.path.exists(base_recipe_repo_root):
  print(
      f"Skipping sample_a3ultra_maxtext_single_run.py - required directory not found: {base_recipe_repo_root}"
  )
  sys.exit(0)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")

if project_root not in sys.path:
  sys.path.insert(0, project_root)

import datetime
from dags.map_reproducibility.utils.constants import Image
from dags.map_reproducibility.internal_runs.dag_configs import DAG_CONFIGS_ULTRA
from dags.map_reproducibility.utils.sample_workload_utils import run_internal_sample_aotc_workload


def main():
  utc_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
  NIGHTLY_IMAGE = f"{Image.MAXTEXT_JAX_STABLE_NIGHTLY}:{utc_date}"
  RELEASE_IMAGE = f"{Image.MAXTEXT_JAX_STABLE_RELEASE}:{utc_date}"
  RELEASE_IMAGE = f"{Image.MAXTEXT_JAX_STABLE_RELEASE}:2025-04-17"
  SAMPLE_RUN_BUCKET_NAME = "yujunzou-dev-supercomputer-testing"

  # Setup configuration
  relative_config_yaml_path = (
      "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_bf16_maxtext.yaml"
  )
  config_name = relative_config_yaml_path.replace(".yaml", "")
  timeout = DAG_CONFIGS_ULTRA[relative_config_yaml_path]["timeout_minutes"]

  run_internal_sample_aotc_workload(
      relative_config_yaml_path=relative_config_yaml_path,
      base_recipe_repo_root=base_recipe_repo_root,
      timeout=timeout,
      image_version=RELEASE_IMAGE,
      sample_run_bucket_name=SAMPLE_RUN_BUCKET_NAME,
  )


if __name__ == "__main__":
  main()
