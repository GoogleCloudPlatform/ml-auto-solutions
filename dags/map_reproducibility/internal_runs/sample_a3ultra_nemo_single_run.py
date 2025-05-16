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
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")

if project_root not in sys.path:
  sys.path.insert(0, project_root)

from dags.map_reproducibility.utils.constants import Image, WorkloadLauncher
from dags.map_reproducibility.internal_runs.dag_configs import DAG_CONFIGS_ULTRA_NEMO
from dags.map_reproducibility.utils.internal_aotc_workload import run_internal_united_workload


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Resolve base paths
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

base_helm_repo_root = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "..",
        "reproducible-benchmark-recipes/projects/gpu-recipes",
    )
)

# Log the resolved paths
logging.info(f"Resolved base_recipe_repo_root: {base_recipe_repo_root}")
logging.info(f"Resolved base_helm_repo_root: {base_helm_repo_root}")

# Check and warn if either path does not exist
if not os.path.exists(base_recipe_repo_root):
  logging.warning(
      f"Required directory not found: {base_recipe_repo_root}. Skipping sample_a3ultra_nemo_single_run.py."
  )
  sys.exit(0)

if not os.path.exists(base_helm_repo_root):
  logging.warning(
      f"Helm directory not found: {base_helm_repo_root}. Continuing execution, but helm operations may fail."
  )


def main():
  RELEASE_IMAGE = Image.NEMO_STABLE_RELEASE_A3U
  SAMPLE_RUN_BUCKET_NAME = "yujunzou-dev-supercomputer-testing"

  # Setup configuration
  relative_config_yaml_path = (
      "recipes/a3ultra/nemo/a3ultra_llama3.1-8b_8gpus_bf16_nemo.yaml"
  )
  timeout = DAG_CONFIGS_ULTRA_NEMO[relative_config_yaml_path]["timeout_minutes"]

  run_internal_united_workload(
      relative_config_yaml_path=relative_config_yaml_path,
      base_recipe_repo_root=base_recipe_repo_root,
      base_helm_repo_root=base_helm_repo_root,
      timeout=timeout,
      image_version=RELEASE_IMAGE,
      sample_run_bucket_name=SAMPLE_RUN_BUCKET_NAME,
      workload_launcher=WorkloadLauncher.NEMO_TEN_LAUNCHER,
  )


if __name__ == "__main__":
  main()
