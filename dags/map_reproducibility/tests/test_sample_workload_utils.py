import sys
import os
import unittest

# --- Setup sys.path and repo check (Keep as is) ---
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
      f"Skipping test_sample_workload_utils.py - required directory not found: {base_recipe_repo_root}"
  )

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))

print(f"Test Script directory: {script_dir}")
print(f"Test Project root: {project_root}")

if project_root not in sys.path:
  sys.path.insert(0, project_root)
# --- End Setup sys.path ---

# Assuming the functions are in this module or imported
from dags.map_reproducibility.utils.sample_workload_utils import (
    sample_workload_gcs_to_cns_cmds,
    execute_workload_commands,
)
from dags.map_reproducibility.utils.common_utils import (
    find_xprof_gcs_path,
)


class TestSampleWorkloadUtils(unittest.TestCase):

  def test_execute_workload_commands_real_success(self):
    """
    Test execute_workload_commands with a real subprocess that succeeds.
    """
    # Use simple commands guaranteed to succeed in most environments
    gcs_path = "gs://yujunzou-dev-supercomputer-testing/maxtext/yujunzou-coreml-llama-3-1-8b-1745453263-maxtext-xpbx-1745453272-xppn/tensorboard/plugins/profile/2025_04_24_00_13_31/yujunzou-coreml-llama-3-1-8b-1745453263-maxtext-xpbx-0.xplane.pb"
    commands = sample_workload_gcs_to_cns_cmds(gcs_path)

    # --- Act ---
    # Execute the commands using the real subprocess mechanism
    success, results = execute_workload_commands(commands, "/tmp")
    print(f"Real execution success flag: {success}")
    print(f"Real execution results: {results}")

  def test_find_xprof_gcs_path_real_success(self):
    """
    Test find_xprof_gcs_path with a real subprocess that succeeds.
    """
    gcs_run_bucket_folder = "gs://yujunzou-dev-supercomputer-testing/maxtext/yujunzou-coreml-llama-3-1-8b-1745363352-maxtext-okrp-1745363360-h593/"
    xprof_path = find_xprof_gcs_path(gcs_run_bucket_folder)
    print(f"xprof_path is {xprof_path}")


if __name__ == "__main__":
  # Run only the tests in the TestSampleWorkloadUtils class
  suite = unittest.TestSuite()
  suite.addTest(
      TestSampleWorkloadUtils("test_execute_workload_commands_real_success")
  )
  runner = unittest.TextTestRunner()
  runner.run(suite)
