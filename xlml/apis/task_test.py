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

"""Tests for task.py, gke_cluster_config.py, and GKE config builders."""

import copy
import datetime
import unittest
from unittest import mock

from airflow import models
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import QuarantineTests
from dags.common.vm_resource import GkeClusters, Project, TpuVersion
from dags.multipod.configs import gke_config
from xlml.apis import gcp_config, metric_config, task, test_config


class GclusterTaskTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.test_dag = models.DAG(
        dag_id="test_gcluster_dag",
        start_date=datetime.datetime(2026, 8, 1),
        schedule=None,
    )
    self.test_cfg = test_config.TpuGkeTest(
        test_config.Tpu(version=TpuVersion.V4, cores=8),
        test_name="test-gcluster-workload",
        run_model_cmds=("echo 'hello'",),
        set_up_cmds=None,
        timeout=datetime.timedelta(minutes=30),
        task_owner=test_owner.SURBHI_J,
        cluster_name="test-cluster",
        docker_image="gcr.io/test/image:latest",
        namespace="automation-testing",
    )
    self.gcp_cfg = gcp_config.GCPConfig(
        project_name=Project.CLOUD_TPU_MULTIPOD_DEV.value,
        zone="us-central1-a",
        dataset_name=metric_config.DatasetOption.XLML_DATASET,
    )
    self.runner_cfg = task.GclusterRunnerConfig(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
    )
    self.gcluster_task = task.GclusterTask(
        runner_config=self.runner_cfg,
    )

  def test_gcluster_config_override(self):
    """Overrides GkeClusterConfig and verifies propagation in GKE factory."""
    cfg = GkeClusters.TPU_V5P_MLPERF_CLUSTER
    self.assertEqual(cfg.core_count, 8)
    self.assertEqual(cfg.namespace, "automation-testing")

    overridden = cfg.override(
        core_count=64, namespace="custom-ns", queue="custom-queue"
    )
    self.assertEqual(overridden.core_count, 64)
    self.assertEqual(overridden.namespace, "custom-ns")
    self.assertEqual(overridden.queue, "custom-queue")
    self.assertEqual(overridden.name, cfg.name)

    gcluster_task_obj = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-override",
        run_model_cmds=("echo 'override'",),
        docker_image="gcr.io/test/image:latest",
        cluster=overridden,
        test_owner=test_owner.SURBHI_J,
        use_gcluster=True,
    )
    test_cfg = gcluster_task_obj.runner_config.task_test_config
    self.assertEqual(test_cfg.accelerator.cores, 64)
    self.assertEqual(test_cfg.namespace, "custom-ns")
    self.assertEqual(gcluster_task_obj.runner_config.queue, "custom-queue")

  def test_get_gke_config_gcluster(self):
    """Constructs a GclusterTask forwarding all configuration parameters."""
    gcluster_task_obj = gke_config.get_gke_config(
        time_out_in_min=45,
        test_name="test-gcluster",
        run_model_cmds=("echo 'gcluster'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V5P_MLPERF_CLUSTER,
        test_owner=test_owner.SURBHI_J,
        num_slices=2,
        priority="very-high",
        max_restart=3,
        ramdisk_directory="/ramdisk",
        mtc_enabled=True,
        use_pathways=True,
        pathways_gcs_location="gs://custom-bucket/scratch",
        gcluster_version="v1.102.0",
        mounts=["/dev/shm;/dev/shm;rw"],
        use_gcluster=True,
    )
    self.assertIsInstance(gcluster_task_obj, task.GclusterTask)
    self.assertIsInstance(gcluster_task_obj, task.BaseRunnerTask)
    runner_cfg = gcluster_task_obj.runner_config
    self.assertIsInstance(runner_cfg, task.GclusterRunnerConfig)
    self.assertEqual(runner_cfg.task_test_config.num_slices, 2)
    self.assertEqual(runner_cfg.priority, "very-high")
    self.assertEqual(runner_cfg.max_restart, 3)
    self.assertEqual(runner_cfg.ramdisk_directory, "/ramdisk")
    self.assertTrue(runner_cfg.mtc_enabled)
    self.assertTrue(runner_cfg.use_pathways)
    self.assertEqual(
        runner_cfg.pathways_gcs_location, "gs://custom-bucket/scratch"
    )
    self.assertEqual(runner_cfg.gcluster_version, "v1.102.0")
    self.assertEqual(runner_cfg.mounts, ["/dev/shm;/dev/shm;rw"])
    self.assertEqual(
        runner_cfg.task_gcp_config.project_name,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.project,
    )
    self.assertEqual(
        runner_cfg.task_gcp_config.zone,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.zone,
    )
    self.assertEqual(
        runner_cfg.task_test_config.accelerator.version,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.device_version,
    )
    self.assertEqual(
        runner_cfg.task_test_config.accelerator.cores,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.core_count,
    )
    self.assertEqual(
        runner_cfg.task_test_config.cluster_name,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.name,
    )
    self.assertEqual(
        runner_cfg.task_test_config.namespace,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.namespace,
    )
    self.assertEqual(
        runner_cfg.queue,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.queue,
    )
    self.assertEqual(
        runner_cfg.task_test_config.timeout,
        datetime.timedelta(minutes=45),
    )

  def test_get_gke_config_gcluster_cpu(self):
    """Constructs a GclusterTask with CpuGkeTest and regional zone for CPU."""
    gcluster_task_obj = gke_config.get_gke_config(
        time_out_in_min=90,
        test_name="to-mt",
        run_model_cmds=("echo 'cpu conversion'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.CPU_N2_STANDARD_64_CLUSTER,
        test_owner=test_owner.JACKY_F,
        priority="medium",
        mounts="/dev/shm;/mnt/shm;rw",
        use_gcluster=True,
    )
    self.assertIsInstance(gcluster_task_obj, task.GclusterTask)
    runner_cfg = gcluster_task_obj.runner_config
    self.assertIsInstance(runner_cfg.task_test_config, test_config.CpuGkeTest)
    self.assertEqual(
        runner_cfg.task_test_config.accelerator.name, "n2-standard-64-1"
    )
    self.assertEqual(
        runner_cfg.task_test_config.benchmark_id, "to-mt-n2-standard-64-1"
    )
    self.assertEqual(runner_cfg.task_gcp_config.zone, "us-central1")
    self.assertEqual(runner_cfg.mounts, "/dev/shm;/mnt/shm;rw")
    self.assertIsNone(runner_cfg.queue)

  def test_get_gke_config_with_name_gen_and_quarantine_gcluster(self):
    """Constructs GclusterNameGenAndQuarantineTask forwarding all parameters."""
    gcluster_task_obj = gke_config.get_gke_config_with_name_gen_and_quarantine(
        time_out_in_min=35,
        test_name="test-gcluster-namegen",
        run_model_cmds=("echo 'gcluster'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V5P_MLPERF_CLUSTER,
        test_owner=test_owner.SURBHI_J,
        num_slices=2,
        priority="very-high",
        max_restart=2,
        ramdisk_directory="/ramdisk",
        mtc_enabled=True,
        use_pathways=True,
        pathways_gcs_location="gs://custom-bucket/scratch",
        gcluster_version="v1.102.0",
        mounts=["/dev/shm;/dev/shm;rw"],
        use_gcluster=True,
    )
    self.assertIsInstance(
        gcluster_task_obj, task.GclusterNameGenAndQuarantineTask
    )
    self.assertIsInstance(gcluster_task_obj, task.BaseRunnerTask)
    runner_cfg = gcluster_task_obj.runner_config
    self.assertIsInstance(runner_cfg, task.GclusterRunnerConfig)
    self.assertEqual(
        runner_cfg.task_test_config.test_name, "test-gcluster-namegen"
    )
    self.assertEqual(runner_cfg.task_test_config.num_slices, 2)
    self.assertEqual(runner_cfg.priority, "very-high")
    self.assertEqual(runner_cfg.max_restart, 2)
    self.assertEqual(runner_cfg.ramdisk_directory, "/ramdisk")
    self.assertTrue(runner_cfg.mtc_enabled)
    self.assertTrue(runner_cfg.use_pathways)
    self.assertEqual(
        runner_cfg.pathways_gcs_location, "gs://custom-bucket/scratch"
    )
    self.assertEqual(runner_cfg.gcluster_version, "v1.102.0")
    self.assertEqual(runner_cfg.mounts, ["/dev/shm;/dev/shm;rw"])
    self.assertEqual(
        runner_cfg.task_gcp_config.project_name,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.project,
    )
    self.assertEqual(
        runner_cfg.task_gcp_config.zone,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.zone,
    )
    self.assertEqual(
        runner_cfg.task_test_config.accelerator.version,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.device_version,
    )
    self.assertEqual(
        runner_cfg.task_test_config.accelerator.cores,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.core_count,
    )
    self.assertEqual(
        runner_cfg.task_test_config.cluster_name,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.name,
    )
    self.assertEqual(
        runner_cfg.task_test_config.namespace,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.namespace,
    )
    self.assertEqual(
        runner_cfg.queue,
        GkeClusters.TPU_V5P_MLPERF_CLUSTER.queue,
    )
    self.assertEqual(
        runner_cfg.task_test_config.timeout,
        datetime.timedelta(minutes=35),
    )

  def test_get_gke_config_xpk(self):
    """Constructs an XpkTask when use_gcluster=False is defaulted."""
    xpk_task_obj = gke_config.get_gke_config(
        time_out_in_min=30,
        test_name="test-xpk",
        run_model_cmds=("echo 'xpk'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V4_8_MAXTEXT_CLUSTER,
        test_owner=test_owner.SURBHI_J,
    )
    self.assertIsInstance(xpk_task_obj, task.XpkTask)
    self.assertIsInstance(xpk_task_obj, task.BaseRunnerTask)
    self.assertIsInstance(xpk_task_obj.runner_config, task.XpkRunnerConfig)

  def test_get_gke_config_with_name_gen_and_quarantine_xpk(self):
    """Constructs XpkNameGenAndQuarantineTask when use_gcluster=False."""
    xpk_task_obj = gke_config.get_gke_config_with_name_gen_and_quarantine(
        time_out_in_min=30,
        test_name="test-xpk-namegen",
        run_model_cmds=("echo 'xpk'",),
        docker_image="gcr.io/test/image:latest",
        cluster=GkeClusters.TPU_V4_8_MAXTEXT_CLUSTER,
        test_owner=test_owner.SURBHI_J,
    )
    self.assertIsInstance(xpk_task_obj, task.XpkNameGenAndQuarantineTask)
    self.assertIsInstance(xpk_task_obj, task.BaseRunnerTask)
    self.assertIsInstance(xpk_task_obj.runner_config, task.XpkRunnerConfig)

  def test_gcluster_task_run_structure(self):
    """Builds gcluster lifecycle tasks and verifies operator dependencies."""
    runner_cfg = task.GclusterRunnerConfig(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        priority="high",
        gcluster_version="v1.102.0",
    )
    gcluster_task = task.GclusterTask(runner_config=runner_cfg)
    with self.test_dag:
      tg = gcluster_task.run(
          skip_post_process=True,
      )

    self.assertIsNotNone(tg)
    self.assertEqual(tg.group_id, self.test_cfg.benchmark_id)

    b_id = self.test_cfg.benchmark_id
    prep_op = self.test_dag.get_task(f"{b_id}.pre_process.generate_workload_id")
    dummy_op = self.test_dag.get_task(f"{b_id}.run_model.dummy_op_for_teardown")
    run_op = self.test_dag.get_task(
        f"{b_id}.run_model.launch_workload.run_workload"
    )
    wait_start_op = self.test_dag.get_task(
        f"{b_id}.run_model.launch_workload.wait_for_workload_start"
    )
    wait_comp_op = self.test_dag.get_task(
        f"{b_id}.run_model.wait_for_workload_completion"
    )
    clean_op = self.test_dag.get_task(f"{b_id}.run_model.clean_up_workload")

    self.assertIn(dummy_op, prep_op.downstream_list)
    self.assertIn(run_op, dummy_op.downstream_list)
    self.assertIn(wait_start_op, run_op.downstream_list)
    self.assertIn(wait_comp_op, wait_start_op.downstream_list)
    self.assertIn(clean_op, wait_comp_op.downstream_list)
    self.assertTrue(clean_op.is_teardown)
    self.assertEqual(clean_op.trigger_rule, "all_done_setup_success")

  @mock.patch("xlml.utils.gcluster.run_workload")
  def test_gcluster_task_with_mounts_precedence(self, mock_run_workload):
    """Passes runner_config mounts with precedence over test config mounts."""
    mock_op = mock.MagicMock()
    mock_run_workload.override.return_value = mock_op

    test_cfg_with_mounts = copy.copy(self.test_cfg)
    test_cfg_with_mounts.mounts = ["/fallback/path;/fallback/path;ro"]
    runner_cfg = task.GclusterRunnerConfig(
        task_test_config=test_cfg_with_mounts,
        task_gcp_config=self.gcp_cfg,
        mounts=["/dev/shm;/dev/shm;rw"],
    )
    runner = task.GclusterRunner(
        configs=runner_cfg,
        workload_id="test-wl-1",
        gcs_path="gs://test-bucket/out",
    )
    with self.test_dag:
      runner.launch_workload()
    self.assertEqual(runner_cfg.mounts, ["/dev/shm;/dev/shm;rw"])
    self.assertEqual(
        mock_op.call_args.kwargs.get("mounts"),
        ["/dev/shm;/dev/shm;rw"],
    )

  @mock.patch("xlml.utils.gcluster.run_workload")
  def test_gcluster_task_with_mounts_fallback(self, mock_run_workload):
    """Falls back to test config mounts when runner_config mounts is None."""
    mock_op = mock.MagicMock()
    mock_run_workload.override.return_value = mock_op

    test_cfg_with_mounts = copy.copy(self.test_cfg)
    test_cfg_with_mounts.mounts = ["/fallback/path;/fallback/path;ro"]
    runner_cfg = task.GclusterRunnerConfig(
        task_test_config=test_cfg_with_mounts,
        task_gcp_config=self.gcp_cfg,
        mounts=None,
    )
    runner = task.GclusterRunner(
        configs=runner_cfg,
        workload_id="test-wl-1",
        gcs_path="gs://test-bucket/out",
    )
    with self.test_dag:
      runner.launch_workload()
    self.assertIsNone(runner_cfg.mounts)
    self.assertEqual(
        mock_op.call_args.kwargs.get("mounts"),
        ["/fallback/path;/fallback/path;ro"],
    )

  def test_gcluster_name_gen_and_quarantine_task(self):
    """Injects dynamic run name generator without mutating input configs."""
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    runner_cfg = task.GclusterRunnerConfig(
        task_test_config=self.test_cfg,
        task_gcp_config=self.gcp_cfg,
        task_metric_config=metric_cfg,
        priority="very-high",
        mounts="/dev/shm;/dev/shm;rw",
    )
    original_cmds = list(self.test_cfg.run_model_cmds)
    namegen_task = task.GclusterNameGenAndQuarantineTask(
        runner_config=runner_cfg,
    )
    with self.test_dag:
      tg = namegen_task.run(
          skip_post_process=False,
      )

    self.assertIsNotNone(tg)
    task_ids = [t.task_id for t in self.test_dag.tasks]
    self.assertTrue(any("generate_run_name" in tid for tid in task_ids))
    self.assertTrue(any("generate_tb_file_location" in tid for tid in task_ids))
    self.assertTrue(any("post_process" in tid for tid in task_ids))

    # Validate configuration propagation in _pre_process
    with self.test_dag:
      _, runner = namegen_task._pre_process()
    self.assertIsInstance(runner, task.GclusterRunner)
    self.assertTrue(
        str(runner.configs.task_test_config.run_model_cmds[0]).startswith(
            "export M_RUN_NAME="
        )
    )
    self.assertEqual(
        tuple(runner.configs.task_test_config.run_model_cmds[1:]),
        self.test_cfg.run_model_cmds,
    )
    self.assertIsNotNone(
        runner.configs.task_metric_config.tensorboard_summary.file_location
    )
    self.assertNotEqual(
        runner.configs.task_metric_config.tensorboard_summary.file_location,
        "gs://test-bucket/tb",
    )

    # Assert original configs remain unmutated
    self.assertEqual(self.test_cfg.run_model_cmds, tuple(original_cmds))
    self.assertEqual(
        metric_cfg.tensorboard_summary.file_location, "gs://test-bucket/tb"
    )
    self.assertEqual(runner_cfg.priority, "very-high")

  @mock.patch.object(QuarantineTests, "is_quarantined", return_value=True)
  def test_gcluster_task_with_quarantine_true(self, mock_quarantine):
    """Links quarantined test execution to the quarantine TaskGroup."""
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    with self.test_dag:
      quarantine_group = TaskGroup(group_id="quarantine_group")
      runner_cfg = task.GclusterRunnerConfig(
          task_test_config=self.test_cfg,
          task_gcp_config=self.gcp_cfg,
          task_metric_config=metric_cfg,
          priority="very-high",
      )
      namegen_task = task.GclusterNameGenAndQuarantineTask(
          runner_config=runner_cfg,
          quarantine_task_group=quarantine_group,
      )
      tg = namegen_task.run(
          skip_post_process=True,
      )
    self.assertIsNotNone(tg)
    mock_quarantine.assert_called_once_with(self.test_cfg.benchmark_id)
    self.assertEqual(tg.parent_group, quarantine_group)

  @mock.patch.object(QuarantineTests, "is_quarantined", return_value=False)
  def test_gcluster_task_with_quarantine_false(self, mock_quarantine):
    """Does not attach TaskGroup to quarantine group when not quarantined."""
    metric_cfg = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location="gs://test-bucket/tb",
            aggregation_strategy=metric_config.AggregationStrategy.LAST,
            use_regex_file_location=True,
        ),
    )
    with self.test_dag:
      quarantine_group = TaskGroup(group_id="quarantine_group")
      runner_cfg = task.GclusterRunnerConfig(
          task_test_config=self.test_cfg,
          task_gcp_config=self.gcp_cfg,
          task_metric_config=metric_cfg,
          priority="very-high",
      )
      namegen_task = task.GclusterNameGenAndQuarantineTask(
          runner_config=runner_cfg,
          quarantine_task_group=quarantine_group,
      )
      tg = namegen_task.run(
          skip_post_process=True,
      )
    self.assertIsNotNone(tg)
    mock_quarantine.assert_called_once_with(self.test_cfg.benchmark_id)
    self.assertNotEqual(tg.parent_group, quarantine_group)


if __name__ == "__main__":
  unittest.main()
