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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

print(f"Script directory: {script_dir}")
print(f"Project root: {project_root}")

if project_root not in sys.path:
  sys.path.insert(0, project_root)


from dags.map_reproducibility.utils.constants import Image
from dags.map_reproducibility.internal_runs.dag_configs import DAG_CONFIGS_A4_NEMO
from dags.map_reproducibility.utils.sample_workload_utils import run_internal_sample_aotc_workload_nemo

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
      f"Skipping sample_a4_nemo_single_run.py - required directory not found: {base_recipe_repo_root}"
  )
  sys.exit(0)


def main():
  RELEASE_IMAGE = Image.NEMO_STABLE_RELEASE_A4
  SAMPLE_RUN_BUCKET_NAME = "yujunzou-dev-supercomputer-testing"

  # Setup configuration
  relative_config_yaml_path = (
      "recipes/a4/nemo/a4_llama3.1-70b_256gpus_fp8_nemo.yaml"
  )
  timeout = DAG_CONFIGS_A4_NEMO[relative_config_yaml_path]["timeout_minutes"]

  run_internal_sample_aotc_workload_nemo(
      relative_config_yaml_path=relative_config_yaml_path,
      base_recipe_repo_root=base_recipe_repo_root,
      timeout=timeout,
      image_version=RELEASE_IMAGE,
      sample_run_bucket_name=SAMPLE_RUN_BUCKET_NAME,
  )


if __name__ == "__main__":
  main()
