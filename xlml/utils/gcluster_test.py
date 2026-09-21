# Copyright 2026 Google LLC
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

"""Unit tests for gcluster.py and shared GKE sensor helpers."""

import unittest
from unittest import mock

from airflow.exceptions import AirflowFailException
from dags.common.vm_resource import GpuVersion
import kubernetes
import urllib3
from xlml.utils import gcluster, gke


class GclusterTest(unittest.TestCase):

  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_command_success(self, mock_hook_cls):
    """Executes command successfully when exit code is 0."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster._run_command(["echo", "hello"], {"VAR": "value"})
    mock_hook.run_command.assert_called_once_with(
        ["echo", "hello"], env={"VAR": "value"}
    )

  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_command_failure_raises_runtime_error(self, mock_hook_cls):
    """Raises RuntimeError when subprocess command exits non-zero."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 1
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    with self.assertRaises(RuntimeError):
      gcluster._run_command(["false"], {})

  @mock.patch("os.path.exists", return_value=True)
  @mock.patch("os.chmod")
  @mock.patch("os.makedirs")
  @mock.patch("xlml.utils.gcluster._run_command")
  def test_setup_gcluster(
      self, mock_run_cmd, mock_makedirs, mock_chmod, mock_exists
  ):
    """Downloads and unpacks gcluster bundle binary into tmpdir."""
    bin_path = gcluster._setup_gcluster("/tmp/test_dir", "v1.102.0", {})
    self.assertEqual(bin_path, "/tmp/test_dir/bin/gcluster")
    mock_makedirs.assert_called_once_with("/tmp/test_dir/bin", exist_ok=True)
    self.assertEqual(mock_run_cmd.call_count, 2)
    curl_call = mock_run_cmd.call_args_list[0][0][0]
    self.assertIn("curl", curl_call)
    self.assertIn(
        "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/"
        "download/v1.102.0/gcluster_bundle_linux_amd64.tgz",
        curl_call,
    )
    tar_call = mock_run_cmd.call_args_list[1][0][0]
    self.assertEqual(
        tar_call,
        [
            "tar",
            "-xzf",
            "/tmp/test_dir/gcluster_bundle.tgz",
            "-C",
            "/tmp/test_dir/bin",
        ],
    )
    mock_chmod.assert_called_once_with("/tmp/test_dir/bin/gcluster", 0o755)

  def test_get_credentials_command(self):
    """Constructs gcloud get-credentials command."""
    cmd = gcluster._get_credentials_command(
        cluster_name="test-cluster",
        location="us-central1-a",
        project_id="test-project",
    )
    self.assertEqual(
        cmd,
        [
            "gcloud",
            "container",
            "clusters",
            "get-credentials",
            "test-cluster",
            "--location=us-central1-a",
            "--project=test-project",
        ],
    )

  def test_set_namespace_command(self):
    """Constructs kubectl set-context command with namespace."""
    cmd = gcluster._set_namespace_command("automation-testing")
    self.assertEqual(
        cmd,
        [
            "kubectl",
            "config",
            "set-context",
            "--current",
            "--namespace=automation-testing",
        ],
    )

  def test_is_valid_gpu_version(self):
    """Validates GPU accelerator types against supported models."""
    self.assertTrue(gcluster.is_valid_gpu_version(GpuVersion.H100.value))
    self.assertTrue(
        gcluster.is_valid_gpu_version(GpuVersion.XPK_H100_MEGA.value)
    )
    self.assertTrue(gcluster.is_valid_gpu_version(GpuVersion.L4.value))
    self.assertFalse(gcluster.is_valid_gpu_version("v4-8"))
    self.assertFalse(gcluster.is_valid_gpu_version("v5e-16"))

  @mock.patch("uuid.uuid4")
  def test_generate_workload_id_deterministic(self, mock_uuid):
    """Generates sanitized, lowercase, RFC 1123-compliant workload IDs."""
    fake_uuid = mock.MagicMock()
    fake_uuid.hex = "abcdef123456"
    mock_uuid.return_value = fake_uuid

    res = gcluster.generate_workload_id.function("MaxText-TPU-Pre-Train-1")
    self.assertEqual(res, "maxtext-tpu-pre-tra-abcde")
    self.assertLessEqual(len(res), 28)

    res = gcluster.generate_workload_id.function("test@#$_bench!*")
    self.assertEqual(res, "test-bench-abcde")

    res_empty = gcluster.generate_workload_id.function("")
    self.assertEqual(res_empty, "job-abcde")

    res_symbols = gcluster.generate_workload_id.function("!@#$%^&*")
    self.assertEqual(res_symbols, "job-abcde")

  @mock.patch("uuid.uuid4")
  def test_generate_workload_id_trailing_hyphen_truncation(self, mock_uuid):
    """Removes trailing hyphens produced at the 19-char truncation boundary."""
    fake_uuid = mock.MagicMock()
    fake_uuid.hex = "1234567890ab"
    mock_uuid.return_value = fake_uuid

    res = gcluster.generate_workload_id.function("my-long-test-names-extra")
    self.assertEqual(res, "my-long-test-names-12345")
    self.assertFalse(res.startswith("my-long-test-names--"))

  @mock.patch("tempfile.TemporaryDirectory")
  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_tpu(self, mock_hook_cls, mock_log_meta, mock_tempdir):
    """Executes 'gcluster job submit' command with TPU slice parameters."""
    mock_tempdir.return_value.__enter__.return_value = "/tmp/mock_gcluster_dir"
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type="v4-8",
        run_cmds="bash run.sh",
        num_slices=1,
        priority="very-high",
        namespace="automation-testing",
        mounts=["/dev/shm;/dev/shm;rw", "/local/path;/container/path;ro"],
    )

    mock_log_meta.assert_called_once()
    self.assertEqual(mock_hook.run_command.call_count, 5)
    submit_cmd = mock_hook.run_command.call_args_list[4][0][0]
    kwargs = mock_hook.run_command.call_args_list[4][1]

    self.assertIn("job", submit_cmd)
    self.assertIn("submit", submit_cmd)
    self.assertIn("--cluster=test-cluster", submit_cmd)
    self.assertIn("--location=us-central1-a", submit_cmd)
    self.assertIn("--project=test-project", submit_cmd)
    self.assertIn("--name=test-workload-12345", submit_cmd)
    self.assertIn("--command=bash run.sh", submit_cmd)
    self.assertIn("--compute-type=v4-8", submit_cmd)
    self.assertIn("--image=us-docker.pkg.dev/img:latest", submit_cmd)
    self.assertIn("--num-slices=1", submit_cmd)
    self.assertNotIn("--num-nodes=1", submit_cmd)
    self.assertIn("--priority=very-high", submit_cmd)
    self.assertIn("--env=GCS_OUTPUT=gs://test-bucket/output", submit_cmd)
    self.assertIn("--skip-prereqs", submit_cmd)
    self.assertIn("--queue=default", submit_cmd)
    self.assertIn("--gke-namespace=automation-testing", submit_cmd)
    self.assertIn("--mount=/dev/shm;/dev/shm;rw", submit_cmd)
    self.assertIn("--mount=/local/path;/container/path;ro", submit_cmd)
    self.assertEqual(
        kwargs["env"]["KUBECONFIG"],
        "/tmp/mock_gcluster_dir/gcluster_kube.conf",
    )

  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_gpu_and_mtc_and_restarts(
      self, mock_hook_cls, mock_log_meta
  ):
    """Formats GPU parameters, MTC ramdisk flags, and restart policy."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type=GpuVersion.XPK_H100_MEGA.value,
        run_cmds="python3 -c \"print('hello')\"",
        num_slices=2,
        ramdisk_directory="/dev/shm/ramdisk",
        mtc_enabled=True,
        max_restart=5,
        mounts="/dev/shm;/dev/shm;rw",
    )

    mock_log_meta.assert_called_once()
    submit_cmd = mock_hook.run_command.call_args_list[4][0][0]
    self.assertIn("--num-nodes=2", submit_cmd)
    self.assertNotIn("--num-slices=2", submit_cmd)
    self.assertIn("--gke-mtc-ramdisk-dir=/dev/shm/ramdisk", submit_cmd)
    self.assertIn("--gke-mtc-enabled", submit_cmd)
    self.assertIn("--restarts=5", submit_cmd)
    self.assertIn("--gke-scheduler=gke.io/topology-aware-auto", submit_cmd)
    self.assertIn("--mount=/dev/shm;/dev/shm;rw", submit_cmd)

  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_pathways_and_tensorboard(
      self, mock_hook_cls, mock_log_meta
  ):
    """Formats Pathways and Vertex TensorBoard CLI flags."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type="v4-8",
        run_cmds="bash run.sh",
        use_pathways=True,
        use_vertex_tensorboard=True,
    )

    mock_log_meta.assert_called_once()
    submit_cmd = mock_hook.run_command.call_args_list[4][0][0]
    self.assertIn("--pathways", submit_cmd)
    self.assertIn(
        "--pathways-gcs-location=gs://test-bucket/output/pathways", submit_cmd
    )
    self.assertNotIn("--use-pathways", submit_cmd)
    self.assertIn("--use-vertex-tensorboard", submit_cmd)

  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_custom_pathways_gcs_location(
      self, mock_hook_cls, mock_log_meta
  ):
    """Passes user-specified custom pathways GCS location."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type="v4-8",
        run_cmds="bash run.sh",
        use_pathways=True,
        pathways_gcs_location="gs://custom-bucket/scratch",
    )

    mock_log_meta.assert_called_once()
    submit_cmd = mock_hook.run_command.call_args_list[4][0][0]
    self.assertIn("--pathways", submit_cmd)
    self.assertIn(
        "--pathways-gcs-location=gs://custom-bucket/scratch", submit_cmd
    )

  @mock.patch("xlml.utils.composer.log_metadata_for_xlml_dashboard")
  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_omits_optional_arguments(
      self, mock_hook_cls, mock_log_meta
  ):
    """Omits --queue and --mount when not specified."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.run_workload.function(
        task_id="run_workload",
        cluster_project="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        benchmark_id="test-bench",
        workload_id="test-workload-12345",
        gcs_path="gs://test-bucket/output",
        docker_image="us-docker.pkg.dev/img:latest",
        accelerator_type="v4-8",
        run_cmds="bash run.sh",
        queue="",
        namespace="default",
        mounts=None,
    )
    mock_log_meta.assert_called_once()
    submit_cmd = mock_hook.run_command.call_args_list[4][0][0]
    self.assertNotIn("--queue", submit_cmd)
    self.assertNotIn("--gke-namespace", submit_cmd)
    for arg in submit_cmd:
      self.assertFalse(arg.startswith("--mount"))

  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_run_workload_failure_raises_runtime_error(self, mock_hook_cls):
    """Raises RuntimeError when gcluster job submit command exits non-zero."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 1
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    with self.assertRaises(RuntimeError):
      gcluster.run_workload.function(
          task_id="run_workload",
          cluster_project="test-project",
          zone="us-central1-a",
          cluster_name="test-cluster",
          benchmark_id="test-bench",
          workload_id="test-workload-12345",
          gcs_path="gs://test-bucket/output",
          docker_image="us-docker.pkg.dev/img:latest",
          accelerator_type="v4-8",
          run_cmds="bash run.sh",
      )

  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_clean_up_workload_success(self, mock_hook_cls):
    """Issues 'gcluster job cancel' with cluster and namespace flags."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 0
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    gcluster.clean_up_workload.function(
        workload_id="test-workload",
        project_id="test-project",
        zone="us-central1-a",
        cluster_name="test-cluster",
        gcluster_version="v1.102.0",
        namespace="automation-testing",
    )

    self.assertEqual(mock_hook.run_command.call_count, 5)
    cancel_cmd = mock_hook.run_command.call_args_list[4][0][0]
    self.assertIn("job", cancel_cmd)
    self.assertIn("cancel", cancel_cmd)
    self.assertIn("test-workload", cancel_cmd)
    self.assertIn("--cluster=test-cluster", cancel_cmd)
    self.assertIn("--location=us-central1-a", cancel_cmd)
    self.assertIn("--project=test-project", cancel_cmd)
    self.assertIn("--skip-prereqs", cancel_cmd)
    self.assertIn("--gke-namespace=automation-testing", cancel_cmd)

  @mock.patch("xlml.utils.gcluster.SubprocessHook")
  def test_clean_up_workload_failure_raises_runtime_error(self, mock_hook_cls):
    """Raises RuntimeError when gcluster job cancel exits non-zero."""
    mock_hook = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.exit_code = 1
    mock_hook.run_command.return_value = mock_result
    mock_hook_cls.return_value = mock_hook

    with self.assertRaises(RuntimeError):
      gcluster.clean_up_workload.function(
          workload_id="test-workload",
          project_id="test-project",
          zone="us-central1-a",
          cluster_name="test-cluster",
      )

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_start_all_running(
      self, mock_get_client, mock_list_pods
  ):
    """Returns True when all workload pods are in Running phase."""
    mock_pod1 = mock.MagicMock()
    mock_pod1.metadata.name = "test-workload-0"
    mock_pod1.status.phase = "Running"
    mock_pod2 = mock.MagicMock()
    mock_pod2.metadata.name = "test-workload-1"
    mock_pod2.status.phase = "Running"
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod1, mock_pod2]
    mock_list_pods.return_value = mock_pod_list

    started = gke.wait_for_workload_start.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    self.assertTrue(started)

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_start_mixed_running_and_pending(
      self, mock_get_client, mock_list_pods
  ):
    """Returns False when at least one pod remains in Pending phase."""
    mock_pod1 = mock.MagicMock()
    mock_pod1.metadata.name = "test-workload-0"
    mock_pod1.status.phase = "Running"

    mock_pod2 = mock.MagicMock()
    mock_pod2.metadata.name = "test-workload-1"
    mock_pod2.status.phase = "Pending"

    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod1, mock_pod2]
    mock_list_pods.return_value = mock_pod_list

    started = gke.wait_for_workload_start.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    self.assertFalse(started)

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_start_no_pods(
      self, mock_get_client, mock_list_pods
  ):
    """Returns False when no workload pods have been created yet."""
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list

    started = gke.wait_for_workload_start.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    self.assertFalse(started)

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_start_failed_pod(
      self, mock_get_client, mock_list_pods
  ):
    """Fails fast with AirflowFailException when a pod is in Failed phase."""
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "test-workload-0"
    mock_pod.status.phase = "Failed"
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_list_pods.return_value = mock_pod_list

    with self.assertRaises(AirflowFailException):
      gke.wait_for_workload_start.function(
          workload_id="test-workload",
          project_id="test-project",
          region="us-central1",
          cluster_name="test-cluster",
      )
    mock_get_client.assert_called_once()

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_succeeded(
      self, mock_get_client, mock_list_pods
  ):
    """Returns True when all workload pods reach Succeeded phase."""
    mock_pod1 = mock.MagicMock()
    mock_pod1.metadata.name = "test-workload-0"
    mock_pod1.status.phase = "Succeeded"
    mock_pod1.status.container_statuses = []
    mock_pod2 = mock.MagicMock()
    mock_pod2.metadata.name = "test-workload-1"
    mock_pod2.status.phase = "Succeeded"
    mock_pod2.status.container_statuses = []
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod1, mock_pod2]
    mock_list_pods.return_value = mock_pod_list

    completed = gke.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    self.assertTrue(completed)

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_failed_container_exit_code(
      self, mock_get_client, mock_list_pods
  ):
    """Raises AirflowFailException when container has non-zero exit code."""
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "test-workload-0"
    mock_pod.status.phase = "Succeeded"

    mock_container = mock.MagicMock()
    mock_container.name = "main"
    mock_container.state.terminated.exit_code = 1
    mock_pod.status.container_statuses = [mock_container]

    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_list_pods.return_value = mock_pod_list

    mock_core_api = mock.MagicMock()
    # `_preload_content=False` returns a raw urllib3 response that yields
    # `bytes` lines.
    mock_log_response = mock.MagicMock()
    mock_log_response.__iter__.return_value = iter([b"Error traceback\n"])
    mock_core_api.read_namespaced_pod_log.return_value = mock_log_response
    mock_get_client.return_value = mock_core_api

    with self.assertRaises(AirflowFailException):
      gke.wait_for_workload_completion.function(
          workload_id="test-workload",
          project_id="test-project",
          region="us-central1",
          cluster_name="test-cluster",
      )
    mock_get_client.assert_called_once()
    mock_core_api.read_namespaced_pod_log.assert_called_once()

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_still_running(
      self, mock_get_client, mock_list_pods
  ):
    """Returns False while workload pods are still active in Running phase."""
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "test-workload-0"
    mock_pod.status.phase = "Running"
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_list_pods.return_value = mock_pod_list

    completed = gke.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    self.assertFalse(completed)

  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_failed(
      self, mock_get_client, mock_list_pods
  ):
    """Raises AirflowFailException when a pod enters Failed phase."""
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "test-workload-0"
    mock_pod.status.phase = "Failed"
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = [mock_pod]
    mock_list_pods.return_value = mock_pod_list

    with self.assertRaises(AirflowFailException):
      gke.wait_for_workload_completion.function(
          workload_id="test-workload",
          project_id="test-project",
          region="us-central1",
          cluster_name="test-cluster",
      )
    mock_get_client.assert_called_once()

  @mock.patch("xlml.utils.gke.get_workload_job")
  @mock.patch("xlml.utils.gke.get_batch_api_client")
  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_no_pods_job_completed(
      self, mock_get_client, mock_list_pods, mock_get_batch, mock_get_job
  ):
    """Falls back to batch Job status Complete when pod list is empty."""
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list

    mock_condition = mock.MagicMock()
    mock_condition.type = "Complete"
    mock_condition.status = "True"
    mock_job = mock.MagicMock()
    mock_job.status.conditions = [mock_condition]
    mock_get_job.return_value = mock_job

    completed = gke.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    mock_get_batch.assert_called_once()
    self.assertTrue(completed)

  @mock.patch("xlml.utils.gke.get_workload_job")
  @mock.patch("xlml.utils.gke.get_batch_api_client")
  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_no_pods_job_failed(
      self, mock_get_client, mock_list_pods, mock_get_batch, mock_get_job
  ):
    """Raises AirflowFailException on batch Job Failed status fallback."""
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list

    mock_condition = mock.MagicMock()
    mock_condition.type = "Failed"
    mock_condition.status = "True"
    mock_job = mock.MagicMock()
    mock_job.status.conditions = [mock_condition]
    mock_get_job.return_value = mock_job

    with self.assertRaises(AirflowFailException):
      gke.wait_for_workload_completion.function(
          workload_id="test-workload",
          project_id="test-project",
          region="us-central1",
          cluster_name="test-cluster",
      )
    mock_get_client.assert_called_once()
    mock_get_batch.assert_called_once()

  @mock.patch("xlml.utils.gke.get_custom_objects_api_client")
  @mock.patch("xlml.utils.gke.get_workload_job")
  @mock.patch("xlml.utils.gke.get_batch_api_client")
  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_no_pods_job_condition_false(
      self,
      mock_get_client,
      mock_list_pods,
      mock_get_batch,
      mock_get_job,
      mock_get_custom,
  ):
    """Returns False when Job condition Complete has status False."""
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list

    mock_condition = mock.MagicMock()
    mock_condition.type = "Complete"
    mock_condition.status = "False"
    mock_job = mock.MagicMock()
    mock_job.status.conditions = [mock_condition]
    mock_get_job.return_value = mock_job
    mock_custom_api = mock.MagicMock()
    mock_custom_api.get_namespaced_custom_object.return_value = None
    mock_get_custom.return_value = mock_custom_api

    completed = gke.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    mock_get_batch.assert_called_once()
    self.assertFalse(completed)

  @mock.patch("xlml.utils.gke.get_workload_jobset")
  @mock.patch("xlml.utils.gke.get_custom_objects_api_client")
  @mock.patch("xlml.utils.gke.get_workload_job")
  @mock.patch("xlml.utils.gke.get_batch_api_client")
  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_no_pods_jobset_completed(
      self,
      mock_get_client,
      mock_list_pods,
      mock_get_batch,
      mock_get_job,
      mock_get_custom,
      mock_get_jobset,
  ):
    """Falls back to JobSet CRD status Completed when pod list is empty."""
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list
    mock_get_job.return_value = None

    mock_jobset = {
        "status": {"conditions": [{"type": "Completed", "status": "True"}]}
    }
    mock_get_jobset.return_value = mock_jobset

    completed = gke.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    mock_get_batch.assert_called_once()
    mock_get_custom.assert_called_once()
    self.assertTrue(completed)

  @mock.patch("xlml.utils.gke.get_workload_jobset")
  @mock.patch("xlml.utils.gke.get_custom_objects_api_client")
  @mock.patch("xlml.utils.gke.get_workload_job")
  @mock.patch("xlml.utils.gke.get_batch_api_client")
  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_no_pods_jobset_failed(
      self,
      mock_get_client,
      mock_list_pods,
      mock_get_batch,
      mock_get_job,
      mock_get_custom,
      mock_get_jobset,
  ):
    """Raises AirflowFailException on JobSet CRD status Failed fallback."""
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list
    mock_get_job.return_value = None

    mock_jobset = {
        "status": {"conditions": [{"type": "Failed", "status": "True"}]}
    }
    mock_get_jobset.return_value = mock_jobset

    with self.assertRaises(AirflowFailException):
      gke.wait_for_workload_completion.function(
          workload_id="test-workload",
          project_id="test-project",
          region="us-central1",
          cluster_name="test-cluster",
      )
    mock_get_client.assert_called_once()
    mock_get_batch.assert_called_once()
    mock_get_custom.assert_called_once()

  @mock.patch("xlml.utils.gke.get_workload_jobset")
  @mock.patch("xlml.utils.gke.get_custom_objects_api_client")
  @mock.patch("xlml.utils.gke.get_workload_job")
  @mock.patch("xlml.utils.gke.get_batch_api_client")
  @mock.patch("xlml.utils.gke.list_workload_pods")
  @mock.patch("xlml.utils.gke.get_core_api_client")
  def test_wait_for_workload_completion_no_pods_jobset_status_false(
      self,
      mock_get_client,
      mock_list_pods,
      mock_get_batch,
      mock_get_job,
      mock_get_custom,
      mock_get_jobset,
  ):
    """Returns False when JobSet conditions have status False."""
    mock_pod_list = mock.MagicMock()
    mock_pod_list.items = []
    mock_list_pods.return_value = mock_pod_list
    mock_get_job.return_value = None

    mock_jobset = {
        "status": {
            "conditions": [
                {"type": "Completed", "status": "False"},
                {"type": "Failed", "status": "False"},
            ]
        }
    }
    mock_get_jobset.return_value = mock_jobset

    completed = gke.wait_for_workload_completion.function(
        workload_id="test-workload",
        project_id="test-project",
        region="us-central1",
        cluster_name="test-cluster",
    )
    mock_get_client.assert_called_once()
    mock_get_batch.assert_called_once()
    mock_get_custom.assert_called_once()
    self.assertFalse(completed)

  def test_gke_get_workload_jobset_uses_crd_api_group(self):
    """Reads the CRD group `jobset.x-k8s.io`, not the label prefix."""
    mock_custom_api = mock.MagicMock()

    gke.get_workload_jobset(mock_custom_api, "test-workload")

    mock_custom_api.get_namespaced_custom_object.assert_called_once_with(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace="default",
        plural="jobsets",
        name="test-workload",
    )

  def test_gke_log_workload_pod_statuses_clean_exit_is_not_an_error(self):
    """A container that exited 0 is logged at INFO, a failure at ERROR."""

    def container(name, exit_code):
      status = mock.MagicMock()
      status.name = name
      status.state.waiting = None
      status.state.terminated.reason = "Completed"
      status.state.terminated.exit_code = exit_code
      return status

    pod = mock.MagicMock()
    pod.metadata.name = "test-workload-pathways-head-0-0-abcde"
    pod.status.phase = "Succeeded"
    pod.status.container_statuses = [
        container("workload-container", 0),
        container("pathways-worker", 1),
    ]
    pods = mock.MagicMock(items=[pod])

    with self.assertLogs(level="INFO") as logs:
      gke.log_workload_pod_statuses("test-workload", pods)

    records = {
        r.levelname
        for r in logs.records
        if "workload-container' TERMINATED" in r.getMessage()
    }
    self.assertEqual(records, {"INFO"})
    records = {
        r.levelname
        for r in logs.records
        if "pathways-worker' TERMINATED" in r.getMessage()
    }
    self.assertEqual(records, {"ERROR"})

  @mock.patch("xlml.utils.gke.get_authenticated_client")
  @mock.patch("kubernetes.client.CustomObjectsApi")
  @mock.patch("kubernetes.client.BatchV1Api")
  @mock.patch("kubernetes.client.CoreV1Api")
  def test_gke_get_api_clients(
      self,
      mock_core_cls,
      mock_batch_cls,
      mock_custom_cls,
      mock_get_auth_client,
  ):
    """Initializes and verifies authenticated Kubernetes API clients."""
    mock_auth_client = mock.MagicMock()
    mock_get_auth_client.return_value = mock_auth_client

    core_api = gke.get_core_api_client("test-proj", "us-central1", "cluster-1")
    mock_get_auth_client.assert_called_with(
        "test-proj", "us-central1", "cluster-1"
    )
    mock_core_cls.assert_called_once_with(mock_auth_client)
    self.assertEqual(core_api, mock_core_cls.return_value)

    batch_api = gke.get_batch_api_client(
        "test-proj", "us-central1", "cluster-1"
    )
    mock_batch_cls.assert_called_once_with(mock_auth_client)
    self.assertEqual(batch_api, mock_batch_cls.return_value)

    custom_api = gke.get_custom_objects_api_client(
        "test-proj", "us-central1", "cluster-1"
    )
    mock_custom_cls.assert_called_once_with(mock_auth_client)
    self.assertEqual(custom_api, mock_custom_cls.return_value)

    self.assertEqual(mock_get_auth_client.call_count, 3)

  def test_gke_list_workload_pods_fallback(self):
    """Lists pods by job-name and falls back to jobset-name selector."""
    mock_core_api = mock.MagicMock()
    empty_pods = mock.MagicMock()
    empty_pods.items = []

    matched_pods = mock.MagicMock()
    matched_pods.items = [mock.MagicMock()]

    # First call with job-name returns empty, second call returns matched
    mock_core_api.list_namespaced_pod.side_effect = [empty_pods, matched_pods]

    result = gke.list_workload_pods(
        mock_core_api, "test-workload", namespace="test-ns"
    )
    self.assertEqual(result, matched_pods)
    self.assertEqual(mock_core_api.list_namespaced_pod.call_count, 2)
    mock_core_api.list_namespaced_pod.assert_any_call(
        namespace="test-ns", label_selector="job-name=test-workload"
    )
    mock_core_api.list_namespaced_pod.assert_any_call(
        namespace="test-ns",
        label_selector="jobset.sigs.k8s.io/jobset-name=test-workload",
    )

  def test_gke_get_workload_job_fallback(self):
    """Reads Job by name and falls back to label selector on ApiException."""
    mock_batch_api = mock.MagicMock()
    mock_batch_api.read_namespaced_job.side_effect = (
        kubernetes.client.exceptions.ApiException(status=404)
    )

    mock_job = mock.MagicMock()
    mock_job_list = mock.MagicMock()
    mock_job_list.items = [mock_job]
    mock_batch_api.list_namespaced_job.return_value = mock_job_list

    result = gke.get_workload_job(
        mock_batch_api, "test-workload", namespace="test-ns"
    )
    self.assertEqual(result, mock_job)
    mock_batch_api.read_namespaced_job.assert_called_once_with(
        name="test-workload", namespace="test-ns"
    )
    mock_batch_api.list_namespaced_job.assert_called_once_with(
        label_selector="jobset.sigs.k8s.io/jobset-name=test-workload",
        namespace="test-ns",
    )

  def test_gke_print_pod_logs_all_containers(self):
    """Streams and decodes logs of every container of the pod."""
    mock_core_api = mock.MagicMock()
    responses = []
    for _ in range(2):
      response = mock.MagicMock()
      response.__iter__.return_value = iter([b"line 1\n", b"line 2\n"])
      responses.append(response)
    mock_core_api.read_namespaced_pod_log.side_effect = responses

    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "test-workload-0"
    mock_pod.metadata.namespace = "test-ns"
    main_container = mock.MagicMock()
    main_container.name = "main"
    sidecar_container = mock.MagicMock()
    sidecar_container.name = "sidecar"
    mock_pod.spec.containers = [main_container, sidecar_container]

    with self.assertLogs(level="INFO") as logs:
      gke.print_pod_logs(mock_core_api, mock_pod)

    self.assertEqual(mock_core_api.read_namespaced_pod_log.call_count, 2)
    mock_core_api.read_namespaced_pod_log.assert_any_call(
        name="test-workload-0",
        namespace="test-ns",
        container="main",
        _preload_content=False,
    )
    mock_core_api.read_namespaced_pod_log.assert_any_call(
        name="test-workload-0",
        namespace="test-ns",
        container="sidecar",
        _preload_content=False,
    )
    for response in responses:
      response.release_conn.assert_called_once()
    self.assertIn("line 1", "\n".join(logs.output))

  def test_gke_print_pod_logs_without_container_spec(self):
    """Skips log fetching when the pod has no container spec."""
    mock_core_api = mock.MagicMock()
    mock_pod = mock.MagicMock()
    mock_pod.metadata.name = "test-workload-0"
    mock_pod.spec = None

    with self.assertLogs(level="WARNING"):
      gke.print_pod_logs(mock_core_api, mock_pod)

    mock_core_api.read_namespaced_pod_log.assert_not_called()

  def test_gke_print_container_logs_api_exception(self):
    """Logs a warning instead of raising when the log API call fails."""
    mock_core_api = mock.MagicMock()
    mock_core_api.read_namespaced_pod_log.side_effect = (
        kubernetes.client.exceptions.ApiException(status=404)
    )

    with self.assertLogs(level="WARNING") as logs:
      gke.print_container_logs(
          mock_core_api,
          namespace="test-ns",
          pod_name="test-workload-0",
          container_name="main",
      )

    self.assertIn("Could not retrieve logs", "\n".join(logs.output))

  def test_gke_print_container_logs_interrupted_stream(self):
    """Releases the connection when the log stream breaks midway."""
    mock_core_api = mock.MagicMock()
    mock_response = mock.MagicMock()
    mock_response.__iter__.side_effect = urllib3.exceptions.ProtocolError(
        "connection broken"
    )
    mock_core_api.read_namespaced_pod_log.return_value = mock_response

    with self.assertLogs(level="WARNING") as logs:
      gke.print_container_logs(
          mock_core_api,
          namespace="test-ns",
          pod_name="test-workload-0",
          container_name="main",
      )

    self.assertIn("was interrupted", "\n".join(logs.output))
    mock_response.release_conn.assert_called_once()


if __name__ == "__main__":
  unittest.main()
