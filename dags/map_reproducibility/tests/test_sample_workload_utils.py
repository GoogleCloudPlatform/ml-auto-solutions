# This test file should be run from the project root directory using:
# python -m unittest discover dags/map_reproducibility/tests -p "test_sample_workload_utils.py"
#
# Other methods that might work:
# 1. Specific test: python -m unittest dags.map_reproducibility.tests.test_sample_workload_utils.TestSampleWorkloadUtils.test_execute_workload_commands_real_success


import unittest

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
    self.assertTrue(success)

  def test_find_xprof_gcs_path_real_success(self):
    """
    Test find_xprof_gcs_path with a real subprocess that succeeds.
    """
    gcs_run_bucket_folder = "gs://yujunzou-dev-supercomputer-testing/maxtext/yujunzou-coreml-llama-3-1-8b-1745363352-maxtext-okrp-1745363360-h593/"
    xprof_path = find_xprof_gcs_path(gcs_run_bucket_folder)
    self.assertIsNotNone(xprof_path, "xprof_path should not be None")
    self.assertTrue(
        xprof_path.startswith("gs://"), "xprof_path should be a GCS path"
    )
