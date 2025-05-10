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

import unittest
import tempfile
import os
import yaml
from dags.map_reproducibility.utils.common_utils import helm_apply_cmds_workload, BUCKET_NAME, copy_bucket_cmds_workload, extract_value_from_yaml


class TestHelmApplyCmdsWorkload(unittest.TestCase):

  def setUp(self):
    self.framework_maxtext = "maxtext"
    self.framework_nemo = "nemo"
    self.hypercomputer = "a3ultra"
    self.config_file = "/path/to/config.yaml"
    self.recipe_repo_root = "/path/to/recipes"
    self.workload_launcher = "my-launcher"
    self.base_expected_start = (
        "helm install -f values.yaml"
        " --namespace default"
        " --set namespace=default"
        f' --set-file workload_config="{self.config_file}"'
        f' --set-file workload_launcher="{self.workload_launcher}"'
    )
    self.base_expected_end = f" $JOB_NAME {self.recipe_repo_root}/src/helm-charts/{self.hypercomputer}/jobset"

  def test_maxtext_framework_with_num_steps(self):
    num_steps = 100
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        num_steps=num_steps,
    )
    self.assertEqual(len(cmds), 1)
    expected_workload_args = f'--set workload.arguments[0]="base_output_directory=gs://{BUCKET_NAME}/maxtext-experiments jax_distributed_initialization_timeout=600 steps={num_steps}"'
    self.assertIn(self.base_expected_start, cmds[0])
    self.assertIn(expected_workload_args, cmds[0])
    self.assertNotIn(
        f"--set volumes.gcsMounts[0].bucketName={BUCKET_NAME}", cmds[0]
    )  # gcs_cmd is empty for maxtext
    self.assertIn(self.base_expected_end, cmds[0])

  def test_maxtext_framework_without_num_steps(self):
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
    )
    self.assertEqual(len(cmds), 1)
    expected_workload_args = f'--set workload.arguments[0]="base_output_directory=gs://{BUCKET_NAME}/maxtext-experiments jax_distributed_initialization_timeout=600"'
    self.assertIn(self.base_expected_start, cmds[0])
    self.assertIn(expected_workload_args, cmds[0])
    self.assertNotIn(
        f"--set volumes.gcsMounts[0].bucketName={BUCKET_NAME}", cmds[0]
    )
    self.assertIn(self.base_expected_end, cmds[0])

  def test_other_framework_with_num_steps(self):
    num_steps = 200
    cmds = helm_apply_cmds_workload(
        framework=self.framework_nemo,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        num_steps=num_steps,
    )
    self.assertEqual(len(cmds), 1)
    expected_workload_args = (
        f'--set workload.arguments[0]="trainer.max_steps={num_steps}"'
    )
    expected_gcs_cmd = f"--set volumes.gcsMounts[0].bucketName={BUCKET_NAME}"
    self.assertIn(self.base_expected_start, cmds[0])
    self.assertIn(expected_workload_args, cmds[0])
    self.assertIn(expected_gcs_cmd, cmds[0])
    self.assertIn(self.base_expected_end, cmds[0])

  def test_other_framework_without_num_steps(self):
    cmds = helm_apply_cmds_workload(
        framework=self.framework_nemo,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
    )
    self.assertEqual(len(cmds), 1)
    # workload_arguments_cmd is empty in this case
    self.assertNotIn("--set workload.arguments[0]=", cmds[0])
    expected_gcs_cmd = f"--set volumes.gcsMounts[0].bucketName={BUCKET_NAME}"
    self.assertIn(self.base_expected_start, cmds[0])
    self.assertIn(expected_gcs_cmd, cmds[0])
    self.assertIn(self.base_expected_end, cmds[0])

  def test_with_kueue_name(self):
    kueue_name = "my-custom-queue"
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        kueue_name=kueue_name,
    )
    self.assertEqual(len(cmds), 1)
    self.assertIn(f" --set queue={kueue_name}", cmds[0])

  def test_without_kueue_name(self):
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        kueue_name=None,  # Explicitly None
    )
    self.assertEqual(len(cmds), 1)
    self.assertNotIn(" --set queue=", cmds[0])

  def test_with_aotc_true(self):
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        aotc=True,
    )
    self.assertEqual(len(cmds), 1)
    self.assertIn(" --set-string workload.aotc=true ", cmds[0])

  def test_with_aotc_false(self):
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        aotc=False,  # Explicitly False (default)
    )
    self.assertEqual(len(cmds), 1)
    self.assertNotIn(" --set-string workload.aotc=true ", cmds[0])

  def test_with_additional_cmds(self):
    additional_cmds = "--set foo=bar --set baz=qux"
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        additional_cmds=additional_cmds,
    )
    self.assertEqual(len(cmds), 1)
    self.assertIn(additional_cmds, cmds[0])

  def test_all_options_enabled_maxtext(self):
    num_steps = 50
    kueue_name = "test-queue"
    additional_cmds = "--set custom.value=true"
    cmds = helm_apply_cmds_workload(
        framework=self.framework_maxtext,
        hypercomputer=self.hypercomputer,
        config_file=self.config_file,
        recipe_repo_root=self.recipe_repo_root,
        workload_launcher=self.workload_launcher,
        aotc=True,
        kueue_name=kueue_name,
        additional_cmds=additional_cmds,
        num_steps=num_steps,
    )
    self.assertEqual(len(cmds), 1)
    expected_command = (
        f"{self.base_expected_start}"
        f" --set queue={kueue_name}"
        f' --set workload.arguments[0]="base_output_directory=gs://{BUCKET_NAME}/maxtext-experiments jax_distributed_initialization_timeout=600 steps={num_steps}"'
        # No gcs_cmd for maxtext
        f" --set-string workload.aotc=true "
        f"{additional_cmds}"
        f"{self.base_expected_end}"
    )
    # Normalize spaces for comparison
    self.assertEqual(
        " ".join(cmds[0].split()), " ".join(expected_command.split())
    )


class TestExtractValueFromYaml(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.TemporaryDirectory()
    self.test_yaml_file_name = "test_config.yaml"
    self.test_yaml_content = {
        "workload": {
            "image": "test_image:latest",
            "gpus": 8,
            "settings": {"timeout": 3600, "retries": 3},
        },
        "metadata": {"version": "1.0", "author": "test_user"},
        "simple_key": "simple_value",
    }
    with open(
        os.path.join(self.temp_dir.name, self.test_yaml_file_name), "w"
    ) as f:
      yaml.dump(self.test_yaml_content, f)

  def tearDown(self):
    self.temp_dir.cleanup()

  def test_extract_top_level_key(self):
    value = extract_value_from_yaml(
        self.temp_dir.name, self.test_yaml_file_name, "simple_key"
    )
    self.assertEqual(value, "simple_value")

  def test_extract_nested_key(self):
    value = extract_value_from_yaml(
        self.temp_dir.name, self.test_yaml_file_name, "workload.image"
    )
    self.assertEqual(value, "test_image:latest")

  def test_extract_deeply_nested_key(self):
    value = extract_value_from_yaml(
        self.temp_dir.name,
        self.test_yaml_file_name,
        "workload.settings.timeout",
    )
    self.assertEqual(value, 3600)

  def test_key_not_found_top_level(self):
    value = extract_value_from_yaml(
        self.temp_dir.name, self.test_yaml_file_name, "non_existent_key"
    )
    self.assertIsNone(value)

  def test_key_not_found_nested(self):
    value = extract_value_from_yaml(
        self.temp_dir.name,
        self.test_yaml_file_name,
        "workload.non_existent_setting",
    )
    self.assertIsNone(value)

  def test_file_not_found(self):
    value = extract_value_from_yaml(
        self.temp_dir.name, "non_existent_file.yaml", "workload.image"
    )
    self.assertIsNone(value)

  def test_invalid_yaml_content(self):
    invalid_yaml_file_name = "invalid.yaml"
    with open(
        os.path.join(self.temp_dir.name, invalid_yaml_file_name), "w"
    ) as f:
      f.write("workload: image: test_image:latest\n  gpus: 8")  # Malformed YAML
    value = extract_value_from_yaml(
        self.temp_dir.name, invalid_yaml_file_name, "workload.image"
    )
    self.assertIsNone(value)


class TestCopyBucketCmdsWorkload(unittest.TestCase):

  def setUp(self):
    self.recipe_repo_root = "/path/to/recipes"
    self.tmpdir = "/test/tmp"
    # BUCKET_NAME is imported from common_utils

  def test_maxtext_framework(self):
    framework = "maxtext"
    cmds = copy_bucket_cmds_workload(
        recipe_repo_root=self.recipe_repo_root,
        tmpdir=self.tmpdir,
        framework=framework,
    )

    gcs_location = f"gs://{BUCKET_NAME}/maxtext-experiments/"
    expected_cmds = (
        f"METRICS_FILE={self.tmpdir}/tflog/metrics",
        f"export BUCKET_FOLDER=$(gcloud storage ls {gcs_location} | grep $JOB_NAME)",
        'echo "BUCKET_FOLDER ${BUCKET_FOLDER}"',
        "export COMPLETE_JOB_NAME=$(gcloud storage ls ${BUCKET_FOLDER}tensorboard/ | grep $JOB_NAME)",
        'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
        "export LOG_FILE=$(gcloud storage ls ${COMPLETE_JOB_NAME} | grep events)",
        'echo "LOG_FILE ${LOG_FILE}"',
        f"gcloud storage cp $LOG_FILE $METRICS_FILE",
    )
    self.assertEqual(cmds, expected_cmds)

  def test_nemo_framework(self):
    framework = "nemo"
    cmds = copy_bucket_cmds_workload(
        recipe_repo_root=self.recipe_repo_root,
        tmpdir=self.tmpdir,  # tmpdir is not used by nemo branch, but it's a required arg
        framework=framework,
    )

    gcs_location = f"gs://{BUCKET_NAME}/nemo-experiments/"
    expected_cmds = (
        f"export COMPLETE_JOB_NAME=$(gcloud storage ls {gcs_location} | grep $JOB_NAME)",
        'echo "COMPLETE_JOB_NAME ${COMPLETE_JOB_NAME}"',
        f"cd {self.recipe_repo_root}/src/utils/training_metrics",
        "gcloud storage cp ${COMPLETE_JOB_NAME}dllogger/rank-0/dllogger.json .",
    )
    self.assertEqual(cmds, expected_cmds)

  def test_other_framework_behaves_like_nemo(self):
    framework = "some_other_framework"  # Should fall into the 'else' block
    # For a framework not explicitly "maxtext", it should default to the "nemo" command structure.
    # This test verifies that behavior.
    # Reusing the expected commands from the "nemo" test.
    self.test_nemo_framework()


if __name__ == "__main__":
  unittest.main()
