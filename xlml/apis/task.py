# Copyright 2023 Google LLC
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

"""Base task file for a test job."""

import abc
import contextlib
import copy
import dataclasses
import datetime
import shlex
from typing import Any, Tuple, Union

import airflow
from airflow.models import BaseOperator
from airflow.models.baseoperator import chain
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from airflow.decorators import task
from airflow.operators.empty import EmptyOperator

from dags.common.quarantined_tests import QuarantineTests
from xlml.utils import gpu, metric, name_format, ssh, tpu, xpk, axlearn, gke, kpo
from xlml.apis import gcp_config, metric_config, test_config, gcs


class BaseTask(abc.ABC):
  """This is a class to set up base tasks."""

  @abc.abstractmethod
  def run(self) -> DAGNode:
    """Run a test job.

    Returns:
      A DAG node that executes this test.
    """
    pass

  def run_with_quarantine(self, quarantine_task_group):
    """Run a test job.

    If the test job is flaky, wrap it in a special task group.

    Returns:
      A DAG node that executes this test.
    """
    if hasattr(self, "runner_config"):
      task_test_config = self.runner_config.task_test_config
    elif hasattr(self, "task_test_config"):
      task_test_config = self.task_test_config
    elif hasattr(self, "test_cfg"):
      task_test_config = self.test_cfg
    else:
      raise AttributeError(
          f"{self.__class__.__name__} does not have a test configuration"
          " attribute."
      )
    test_name = task_test_config.benchmark_id
    if QuarantineTests.is_quarantined(test_name):
      with quarantine_task_group:
        return self.run()
    else:
      return self.run()


def run_queued_resource_test(
    # TODO(wcromar): make these args less verbose
    task_test_config: test_config.TestConfig[test_config.Tpu],
    task_gcp_config: gcp_config.GCPConfig,
    task_metric_config: metric_config.MetricConfig | None = None,
    tpu_create_timeout: datetime.timedelta = datetime.timedelta(minutes=60),
    tpu_name_env_var: bool = False,
    all_workers: bool = True,
    skip_post_process: bool = False,
    custom_env: dict[str, str] | None = None,
):
  """This is a class to set up tasks for TPU provisioned by Queued Resource.

  Test steps:
  1. Generates a random TPU name and SSH keys, creates a Queued Resource, and
     runs the test config's setup script on the TPU when it is ready.
  2. Run the TPU test in `task_test_config` via SSH.
  3. Process metrics and metadata, then insert them into BigQuery tables.
  4. Clean up TPU resources created by for this test

  Attributes:
    task_test_config: Test configs to run on this TPU.
    task_gcp_config: Runtime TPU creation parameters.
    task_metric_config: Metric configs to process metrics.
    tpu_create_timeout: Time to provision the machine.
    tpu_name_env_var: The flag to define if set up env variable for tpu name.
    all_workers: The flag to define if run commands on all workers or worker 0
      only.
    skip_post_process: If True, the post processing step will be skipped.
    custom_env: Extra enviroment variables.

  Returns:
      A task group with the following tasks chained: provision, run_model,
      post_process and clean_up.
  """

  if custom_env is None:
    custom_env = {}

  with TaskGroup(
      group_id=task_test_config.benchmark_id, prefix_group_id=True
  ) as test:
    with TaskGroup(group_id="provision") as provision:
      with TaskGroup(group_id="initialize"):
        tpu_name = tpu.generate_tpu_name(
            task_test_config.benchmark_id, tpu_name_env_var
        )
        ssh_keys = ssh.obtain_persist_ssh_keys()
        output_location = name_format.generate_gcs_folder_location(
            task_test_config.gcs_subfolder,
            task_test_config.benchmark_id,
        )

      queued_resource_op, queued_resource_name = tpu.create_queued_resource(
          tpu_name,
          task_gcp_config,
          ssh_keys,
          tpu_create_timeout,
          task_test_config,
      )

      setup_task = tpu.ssh_tpu.override(
          task_id="setup",
          # Setup/install retries don’t need a long cooldown.
          # 30s is enough for network connection problem; longer delays do not
          # make sense.
          retry_delay=datetime.timedelta(seconds=30),
      )(
          queued_resource_name,
          task_test_config.setup_script,
          ssh_keys,
          True if task_test_config.test_name.startswith("tf_") else all_workers,
      )
      _ = queued_resource_op >> setup_task

    run_model = tpu.ssh_tpu.override(
        task_id="run_model",
        execution_timeout=task_test_config.timeout,
        owner=task_test_config.task_owner,
    )(
        queued_resource_name,
        task_test_config.test_script,
        ssh_keys,
        all_workers,
        # We purposely put `custom_env` last to allow overriding values.
        # For example, `GCS_OUTPUT` can be overridden if needed.
        env={
            metric_config.SshEnvVars.GCS_OUTPUT.name: output_location,
            **custom_env,
        },
    )

    clean_up = tpu.delete_queued_resource.override(group_id="clean_up")(
        queued_resource_name
    )

    if skip_post_process:
      _ = provision >> run_model >> clean_up
    else:
      with TaskGroup(group_id="post_process") as post_process:
        process_id = metric.generate_process_id.override(retries=0)()
        metric.process_metrics.override(retries=0)(
            process_id,
            task_test_config,
            task_metric_config,
            task_gcp_config,
            folder_location=output_location,
        )
      _ = provision >> run_model >> post_process >> clean_up

  return test


@dataclasses.dataclass
class AXLearnTask(BaseTask):
  """
  This is a class to set up tasks for TPU/GPU AXLearn.

  Attributes:
    test_cfg: Test configs to run on this TPU/GPU.
    gcp_cfg: Runtime TPU/GPU creation parameters.
    workload_provision_timeout: Timedelta object allowed for
      provisioning a workload.
    workload_run_timeout: Timedelta object allowed for the actual
      workload execution.
    workload_post_test_timeout: Timedelta object allowed for cleanup tasks
      after execution.
    image_name: The name of the Docker image.
    image_repo: The repository path of the Docker image.
    image_full_url: The full URL of the Docker image.
    module: The specific AXLearn module being tested.
    model_name: The configuration file or string for the model.
    trainer_dir: The base directory for trainer output.
    trace_steps: A list of steps where XLA compiler will trace it.
    label: A string used to categorize the workload, often indicating the
      accelerator type (e.g., 'tpu-v5p') or test category.
  """

  test_cfg: Union[
      test_config.TpuGkeTest, test_config.GpuXpkTest, test_config.CpuGkeTest
  ]
  gcp_cfg: gcp_config.GCPConfig

  workload_provision_timeout: datetime.timedelta
  workload_run_timeout: datetime.timedelta
  workload_post_test_timeout: datetime.timedelta

  image_name: str
  image_repo: str
  image_full_url: str

  module: str
  model_name: str
  trainer_dir: str
  trace_steps: str
  label: str

  def run(
      self,
      workload_id: airflow.XComArg,
  ) -> DAGNode:
    """
    Run a test job within a docker image.

    Attributes:
      workload_id: A descriptive name for the test run, which is used to
        generate the unique workload ID.

    Returns:
      A task group with the following task : run_model.
    """
    with TaskGroup(group_id=self.test_cfg.benchmark_id) as group:
      dummy_op_for_teardown = EmptyOperator(
          task_id="dummy_op_for_teardown"
      ).as_setup()

      update_image_tag_cmd = axlearn.update_image_tag_cmd.override(
          owner=self.test_cfg.task_owner
      )(
          image_name=self.image_full_url,
          workload_id=workload_id,
      )

      gen_cmds = axlearn.generate_axlearn_cli_command.override(
          owner=self.test_cfg.task_owner
      )(
          task_id="run_workload",
          project_id=self.gcp_cfg.project_name,
          zone=self.gcp_cfg.zone,
          cluster_name=self.test_cfg.cluster_name,
          workload_id=workload_id,
          docker_image_name=self.image_name,
          docker_image_repo=self.image_repo,
          docker_image_full_url=self.image_full_url,
          accelerator_type=f"tpu-{self.test_cfg.accelerator.name}",
          module=self.module,
          model_config=self.model_name,
          trainer_dir=self.trainer_dir,
          num_slices=self.test_cfg.num_slices,
          trace_steps=self.trace_steps,
          label=self.label,
      )

      run_workload = kpo.run_command_in_kpo(
          start_cli_command=gen_cmds,
          workload_id=workload_id,
          task_owner=self.test_cfg.task_owner,
          provisioning_timeout=self.workload_provision_timeout,
          workload_run_timeout=self.workload_run_timeout,
          image_full_url=self.image_full_url,
      )

      wait_for_workload_start = xpk.wait_for_workload_start.override(
          timeout=self.workload_provision_timeout.total_seconds(),
          owner=self.test_cfg.task_owner,
      )(
          workload_id=workload_id,
          project_id=self.gcp_cfg.project_name,
          region=gke.zone_to_region(self.gcp_cfg.zone),
          cluster_name=self.test_cfg.cluster_name,
      )

      wait_for_workload_completion = xpk.wait_for_workload_completion.override(
          timeout=int(self.workload_run_timeout.total_seconds()),
          owner=self.test_cfg.task_owner,
      )(
          workload_id=workload_id,
          project_id=self.gcp_cfg.project_name,
          region=gke.zone_to_region(self.gcp_cfg.zone),
          cluster_name=self.test_cfg.cluster_name,
      )

      cleanup = xpk.clean_up_workload.override(
          trigger_rule=TriggerRule.ALL_DONE,
          execution_timeout=self.workload_post_test_timeout,
          owner=self.test_cfg.task_owner,
      )(
          workload_id=workload_id,
          project_id=self.gcp_cfg.project_name,
          zone=self.gcp_cfg.zone,
          cluster_name=self.test_cfg.cluster_name,
          xpk_branch=xpk.MAIN_BRANCH,
      ).as_teardown(
          setups=dummy_op_for_teardown, on_failure_fail_dagrun=True
      )

      chain(wait_for_workload_start, wait_for_workload_completion, cleanup)
      flow1 = run_workload
      flow2 = wait_for_workload_start

      chain(
          update_image_tag_cmd,
          gen_cmds,
          dummy_op_for_teardown,
          # We run them in parallel because flow1 blocks until completion and
          # flow2 must run concurrently to monitor progress and ensure cleanup.
          [flow1, flow2],
      )

    return group


@dataclasses.dataclass
class RunnerConfig:
  """Base static configuration for all workload runners."""

  task_test_config: Union[
      test_config.TpuGkeTest, test_config.GpuXpkTest, test_config.CpuGkeTest
  ]
  task_gcp_config: gcp_config.GCPConfig
  task_metric_config: metric_config.MetricConfig | None = None
  workload_provision_timeout: datetime.timedelta = datetime.timedelta(
      minutes=60
  )


@dataclasses.dataclass
class XpkRunnerConfig(RunnerConfig):
  """Static configuration specific to XPK workloads on GKE."""

  xpk_branch: str = xpk.MAIN_BRANCH
  priority: str = "high"
  max_restart: int = 0
  ramdisk_directory: str = ""
  mtc_enabled: bool = False
  use_pathways: bool = False


@dataclasses.dataclass
class Runner(abc.ABC):
  """Base runner managing runtime workload lifecycle and dynamic XComArgs."""

  configs: RunnerConfig
  workload_id: airflow.XComArg
  gcs_path: airflow.XComArg

  @abc.abstractmethod
  def launch_workload(self) -> DAGNode:
    pass

  @abc.abstractmethod
  def wait_workload_complete(self) -> DAGNode:
    pass

  @abc.abstractmethod
  def cleanup_workload(self, tear_down_of: BaseOperator) -> DAGNode:
    pass


@dataclasses.dataclass
class XpkRunner(Runner):
  """XpkRunner executes XPK workloads and manages their lifecycle."""

  configs: XpkRunnerConfig

  def launch_workload(self) -> DAGNode:
    """Create the workload and wait for it to provision."""
    use_vertex_tensorboard = (
        self.configs.task_metric_config.use_vertex_tensorboard
        if self.configs.task_metric_config
        else False
    )

    with TaskGroup(group_id="launch_workload") as group:
      run_workload = xpk.run_workload.override(
          owner=self.configs.task_test_config.task_owner
      )(
          task_id="run_workload",
          cluster_project=self.configs.task_gcp_config.project_name,
          zone=self.configs.task_gcp_config.zone,
          cluster_name=self.configs.task_test_config.cluster_name,
          benchmark_id=self.configs.task_test_config.benchmark_id,
          workload_id=self.workload_id,
          gcs_path=self.gcs_path,
          docker_image=self.configs.task_test_config.docker_image,
          accelerator_type=self.configs.task_test_config.accelerator.name,
          run_cmds=self.configs.task_test_config.test_script,
          num_slices=self.configs.task_test_config.num_slices,
          use_vertex_tensorboard=use_vertex_tensorboard,
          use_pathways=self.configs.use_pathways,
          ramdisk_directory=self.configs.ramdisk_directory,
          mtc_enabled=self.configs.mtc_enabled,
          xpk_branch=self.configs.xpk_branch,
          max_restart=self.configs.max_restart,
          priority=self.configs.priority,
          namespace=self.configs.task_test_config.namespace,
      )
      wait_for_workload_start = xpk.wait_for_workload_start.override(
          timeout=self.configs.workload_provision_timeout.total_seconds()
      )(
          workload_id=self.workload_id,
          project_id=self.configs.task_gcp_config.project_name,
          region=gke.zone_to_region(self.configs.task_gcp_config.zone),
          cluster_name=self.configs.task_test_config.cluster_name,
          namespace=self.configs.task_test_config.namespace,
      )
      chain(run_workload, wait_for_workload_start)
      return group

  def wait_workload_complete(self) -> DAGNode:
    op = xpk.wait_for_workload_completion
    if self.configs.task_test_config.timeout:
      op = op.override(
          timeout=int(self.configs.task_test_config.timeout.total_seconds())
      )
    return op(
        workload_id=self.workload_id,
        project_id=self.configs.task_gcp_config.project_name,
        region=gke.zone_to_region(self.configs.task_gcp_config.zone),
        cluster_name=self.configs.task_test_config.cluster_name,
        namespace=self.configs.task_test_config.namespace,
    )

  def cleanup_workload(self, tear_down_of: BaseOperator) -> DAGNode:
    return xpk.clean_up_workload(
        workload_id=self.workload_id,
        project_id=self.configs.task_gcp_config.project_name,
        zone=self.configs.task_gcp_config.zone,
        cluster_name=self.configs.task_test_config.cluster_name,
        xpk_branch=self.configs.xpk_branch,
        namespace=self.configs.task_test_config.namespace,
    ).as_teardown(
        setups=tear_down_of,
        on_failure_fail_dagrun=True,
    )

  def wait_workload_reach_to_step(
      self,
      expect_reach_to_step: int,
      check_file_exists: bool,
  ) -> DAGNode:
    with TaskGroup(group_id="wait_workload") as group:
      wait_reach_to_step = xpk.wait_for_workload_reach_step.override(
          task_id="wait_for_workload_reach_step"
      )(
          workload_id=self.workload_id,
          project_id=self.configs.task_gcp_config.project_name,
          region=gke.zone_to_region(self.configs.task_gcp_config.zone),
          cluster_name=self.configs.task_test_config.cluster_name,
          expect_reach_to_step=str(expect_reach_to_step),
          namespace=self.configs.task_test_config.namespace,
      )

      task_id_wait_file_exist = "wait_for_file_to_exist"
      wait_for_file_to_exist = gcs.wait_for_file_to_exist.override(
          task_id=task_id_wait_file_exist
      )(
          file_path=(
              f"{self.gcs_path}/{str(expect_reach_to_step)}/commit_success.txt"
          ),
      )
      task_id_do_nothing = "do_nothing"
      do_nothing = EmptyOperator(task_id=task_id_do_nothing)

      @task.branch
      def task_path_decider(check_file_exists: bool = False) -> str:
        """Dynamically route the workflow depending on check_file_exists."""
        if check_file_exists:
          return f"{group.group_id}.{task_id_wait_file_exist}"
        return f"{group.group_id}.{task_id_do_nothing}"

      # Conditional checks: depending on the `check_file_exists` argument
      # specified by the upper-level caller.
      maybe_check_file_exists = task_path_decider(check_file_exists)

      chain(
          wait_reach_to_step,
          maybe_check_file_exists,
          [wait_for_file_to_exist, do_nothing],
      )

      return group

  def interrupt_workload(self, is_targeting_on_last_node: bool) -> DAGNode:
    return xpk.delete_node.override(
        owner=self.configs.task_test_config.task_owner,
        trigger_rule="none_failed",
    )(
        project=self.configs.task_gcp_config.project_name,
        zone=self.configs.task_gcp_config.zone,
        cluster_name=self.configs.task_test_config.cluster_name,
        workload_id=self.workload_id,
        dry_run=False,
        last_node=is_targeting_on_last_node,
        namespace=self.configs.task_test_config.namespace,
    )


@dataclasses.dataclass
class XpkTask(BaseTask):
  """This is a class to set up tasks for TPU/GPU provisioned by XPK tool.

  Attributes:
    runner_config: Static configuration for the XPK runner.
  """

  runner_config: XpkRunnerConfig

  def run(
      self,
      gcs_location: airflow.XComArg | None = None,
      skip_post_process: bool = False,
  ) -> DAGNode:
    """Run a test job within a docker image.

    Attributes:
      gcs_location: GCS path for all artifacts of the test.
      skip_post_process: If True, skip the post-processing step.

    Returns:
      A task group with the following tasks chained: run_model and
      post_process.
    """
    with TaskGroup(
        group_id=self.runner_config.task_test_config.benchmark_id
    ) as group:
      pre_process, xpk_runner = self._pre_process(
          gcs_location=gcs_location,
      )

      run_model = self._run_model(xpk_runner)

      nodes = [pre_process, run_model]
      if not skip_post_process:
        nodes.append(self._post_process(xpk_runner.gcs_path))

      chain(*nodes)

    return group

  def _run_model(self, xpk_runner: XpkRunner) -> DAGNode:
    with TaskGroup(group_id="run_model") as group:
      dummy_op = self._dummy_op_for_teardown()

      chain(
          dummy_op,
          xpk_runner.launch_workload(),
          xpk_runner.wait_workload_complete(),
          xpk_runner.cleanup_workload(tear_down_of=dummy_op),
      )
      return group

  def _maybe_generate_gcs_location(
      self,
      gcs_location: airflow.XComArg | None = None,
  ) -> airflow.XComArg:
    if gcs_location:
      return gcs_location

    return name_format.generate_gcs_folder_location(
        self.runner_config.task_test_config.gcs_subfolder,
        self.runner_config.task_test_config.benchmark_id,
    )

  def _pre_process(
      self,
      gcs_location: airflow.XComArg | None = None,
  ) -> tuple[DAGNode, XpkRunner]:
    with TaskGroup(group_id="pre_process") as group:
      workload_id = xpk.generate_workload_id(
          self.runner_config.task_test_config.benchmark_id
      )

      gcs_path = self._maybe_generate_gcs_location(gcs_location)

      xpk_runner = XpkRunner(
          configs=self.runner_config,
          workload_id=workload_id,
          gcs_path=gcs_path,
      )

    return group, xpk_runner

  def _post_process(self, result_location: str | None = None) -> DAGNode:
    """Process metrics and metadata, and insert them into BigQuery tables.

    Returns:
      A DAG node that executes the post process.
    """
    with TaskGroup(group_id="post_process") as group:
      process_id = metric.generate_process_id.override(retries=0)()
      task_metric_config = self.runner_config.task_metric_config

      if task_metric_config and task_metric_config.profile:
        task_metric_config = copy.copy(task_metric_config)
        profile = copy.copy(task_metric_config.profile)
        profile.metrics = metric.xplane_to_metrics.override(retries=0)(
            profile.file_location
        )
        task_metric_config.profile = profile

        post_process_metrics = metric.process_metrics.override(retries=0)(
            process_id,
            self.runner_config.task_test_config,
            task_metric_config,
            self.runner_config.task_gcp_config,
            folder_location=result_location,
        )
        chain(
            process_id,
            profile.metrics,
            post_process_metrics,
        )
      else:
        post_process_metrics = metric.process_metrics.override(retries=0)(
            process_id,
            self.runner_config.task_test_config,
            task_metric_config,
            self.runner_config.task_gcp_config,
            folder_location=result_location,
        )
        chain(process_id, post_process_metrics)

      return group

  def _dummy_op_for_teardown(self) -> BaseOperator:
    return EmptyOperator(task_id="dummy_op_for_teardown").as_setup()


@dataclasses.dataclass
class XpkNodeInterruptionTask(XpkTask):
  """Task for running XPK workloads with node interruption."""

  expect_reach_to_step: int = 0
  last_node: bool = False
  check_file_exists: bool = False

  def _run_model(self, xpk_runner: XpkRunner) -> DAGNode:
    with TaskGroup(group_id="run_model") as group:
      dummy_op = self._dummy_op_for_teardown()

      chain(
          dummy_op,
          xpk_runner.launch_workload(),
          xpk_runner.wait_workload_reach_to_step(
              self.expect_reach_to_step,
              self.check_file_exists,
          ),
          xpk_runner.interrupt_workload(self.last_node),
          xpk_runner.wait_workload_complete(),
          xpk_runner.cleanup_workload(tear_down_of=dummy_op),
      )

      return group


@dataclasses.dataclass
class XpkNameGenAndQuarantineTask(XpkTask):
  """Task for running XPK workloads with name generation and quarantine."""

  quarantine_task_group: Any = None
  run_name_env: str = "M_RUN_NAME"
  nested_run_name_in_tb_file_location: bool = True

  def run(
      self,
      gcs_location: airflow.XComArg | None = None,
      skip_post_process: bool = False,
  ) -> DAGNode:
    """Generate a unique run name, tensorboard file location,
    and profile file location (if metric config has profile),
    then run a test job within a docker image.

    Returns:
      A task group with the following tasks chained: generate_run_name,
      generate_tb_file_location, generate_profile_file_location (optional),
      run provision, run_model, post_process.
    """

    test_name = self.runner_config.task_test_config.benchmark_id
    cm = (
        self.quarantine_task_group
        if QuarantineTests.is_quarantined(test_name)
        and self.quarantine_task_group
        else contextlib.nullcontext()
    )

    with cm:
      return super().run(
          gcs_location=gcs_location,
          skip_post_process=skip_post_process,
      )

  def _pre_process(
      self,
      gcs_location: airflow.XComArg | None = None,
  ) -> tuple[DAGNode, XpkRunner]:
    with TaskGroup(group_id="pre_process") as group:
      run_name = name_format.generate_run_name(
          self.runner_config.task_test_config.benchmark_id
      )
      workload_id = xpk.generate_workload_id(
          self.runner_config.task_test_config.benchmark_id
      )

      gcs_path = self._maybe_generate_gcs_location(gcs_location)

      nodes = [run_name]

      # Shallow-copy runner_config and task_test_config to prevent mutating
      # shared configs.
      runner_config = copy.copy(self.runner_config)
      task_test_config = copy.copy(self.runner_config.task_test_config)
      task_test_config.run_model_cmds = [
          f"export {self.run_name_env}={run_name}",
          *self.runner_config.task_test_config.run_model_cmds,
      ]
      runner_config.task_test_config = task_test_config

      # Update tensorboard and profile file locations
      if (
          self.runner_config.task_metric_config
          and self.runner_config.task_metric_config.tensorboard_summary
      ):
        task_metric_config = copy.copy(self.runner_config.task_metric_config)
        runner_config.task_metric_config = task_metric_config

        tensorboard_summary = copy.copy(task_metric_config.tensorboard_summary)
        tb_file_location = name_format.generate_tb_file_location(
            run_name,
            tensorboard_summary.file_location,
            self.nested_run_name_in_tb_file_location,
        )
        tensorboard_summary.file_location = tb_file_location
        task_metric_config.tensorboard_summary = tensorboard_summary

        if task_metric_config.profile:
          profile = copy.copy(task_metric_config.profile)
          profile_file_location = name_format.generate_profile_file_location(
              run_name,
              profile.file_location,
          )
          profile.file_location = profile_file_location
          task_metric_config.profile = profile
          nodes.append([tb_file_location, profile_file_location])
        else:
          nodes.append(tb_file_location)

      xpk_runner = XpkRunner(
          configs=runner_config,
          workload_id=workload_id,
          gcs_path=gcs_path,
      )

      chain(*nodes)

    return group, xpk_runner


@dataclasses.dataclass
class GpuCreateResourceTask(BaseTask):
  """This is a class to set up tasks for GPU.

  Attributes:
    image_project: the project that an image belongs to.
    image_family: the family group that an image belongs to.
    task_test_config: task configutation.
    task_gcp_config: gcp related config (e.g., zone, project) for the task.
    task_metric_config: metric configuration (e.g., result gcs path).
    gpu_create_timeout: timeout when waiting for the GPU vm creation.
    install_nvidia_drivers: whether to install Nvidia drivers.
    existing_instance_name: whether an existing GPU instance shall be used.
    reservation: use a specific reservation for the VM instance, if available
  """

  image_project: str
  image_family: str
  task_test_config: test_config.TestConfig[test_config.Gpu]
  task_gcp_config: gcp_config.GCPConfig
  task_metric_config: metric_config.MetricConfig | None = None
  gpu_create_timeout: datetime.timedelta = datetime.timedelta(minutes=60)
  install_nvidia_drivers: bool = False
  existing_instance_name: str = None
  reservation: bool = False

  def run(self) -> DAGNode:
    """Run a test job.

    Returns:
      A task group with the following tasks chained: provision, run_model,
      post_process, clean_up.
    """
    # piz: We skip the queued resource for GPU for now since there is no queued
    # resource command for GPU.
    if self.existing_instance_name is not None:
      return self.run_with_existing_instance()

    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      (
          provision,
          ip_address,
          instance_name,
          ssh_keys,
          gcs_location,
      ) = self.provision()
      # If you already specify `task_metric_config.json_lines` value in the
      # test config script, then `gcs_location` will take no effect.
      if (
          self.task_metric_config
          and self.task_metric_config.use_runtime_generated_gcs_folder
      ):
        env_variable = {
            f"{metric_config.SshEnvVars.GCS_OUTPUT.name}": gcs_location
        }
      else:
        env_variable = None
      run_model = self.run_model(ip_address, ssh_keys, env_variable)
      post_process = self.post_process(gcs_location)
      clean_up = self.clean_up(
          instance_name,
          self.task_gcp_config.project_name,
          self.task_gcp_config.zone,
      )
      _ = provision >> run_model >> post_process >> clean_up
    return group

  def run_with_existing_instance(self) -> DAGNode:
    """Run a test job via existing instance.

    Returns:
      A task group with the following tasks chained:
      provision, run_model and post_process, clean_up.
    """
    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      (
          provision,
          ip_address,
          ssh_keys,
          gcs_location,
      ) = self.provision_via_existing_instance()
      if (
          self.task_metric_config
          and self.task_metric_config.use_runtime_generated_gcs_folder
      ):
        env_variable = {
            f"{metric_config.SshEnvVars.GCS_OUTPUT.name}": gcs_location
        }
      else:
        env_variable = None
      post_process = self.post_process(gcs_location)
      run_model = self.run_model(ip_address, ssh_keys, env_variable)
      clean_up = self.clean_up_existing_instance(ssh_keys)
      _ = provision >> run_model >> post_process >> clean_up
    return group

  def provision_via_existing_instance(
      self,
  ) -> Tuple[DAGNode, airflow.XComArg, airflow.XComArg, airflow.XComArg,]:
    """Provision an existing GPU accelerator.

    Returns:
      A DAG node that will provision a GPU, an XCome value of the ip address
      for the host,an XCom value for the SSH keys.
    """
    with TaskGroup(group_id="provision") as group:
      ssh_keys = ssh.generate_ssh_keys()
      ip_address = gpu.get_existing_resource(
          instance_name=self.existing_instance_name,
          ssh_keys=ssh_keys,
          gcp=self.task_gcp_config,
      )
      gcs_location = name_format.generate_gcs_folder_location(
          self.task_test_config.gcs_subfolder,
          self.task_test_config.benchmark_id,
      )
      return group, ip_address, ssh_keys, gcs_location

  def provision(
      self,
  ) -> Tuple[
      DAGNode,
      airflow.XComArg,
      airflow.XComArg,
      airflow.XComArg,
      airflow.XComArg,
  ]:
    """Provision a GPU accelerator via a resource creation.

    Generates a random GPU name and SSH keys, creates a VM Resource, and
    runs the test config's setup script on the GPU when it is ready.

    Returns:
      A DAG node that will provision a GPU, an XCome value of the ip address
      for the host, an XCom value for the GPU name, and an XCom value for
      the SSH keys.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    with TaskGroup(group_id="provision") as group:
      with TaskGroup(group_id="initialize"):
        gpu_name = gpu.generate_gpu_name()
        ssh_keys = ssh.generate_ssh_keys()
        gcs_location = name_format.generate_gcs_folder_location(
            self.task_test_config.gcs_subfolder,
            self.task_test_config.benchmark_id,
        )

      ip_address = gpu.create_resource(
          gpu_name,
          self.image_project,
          self.image_family,
          self.task_test_config.accelerator,
          self.task_gcp_config,
          ssh_keys,
          timeout=self.gpu_create_timeout,
          install_nvidia_drivers=self.install_nvidia_drivers,
          reservation=self.reservation,
      )

      _ = ip_address >> gpu.ssh_host.override(task_id="setup")(
          ip_address,
          self.task_test_config.setup_script,
          ssh_keys,
      )

    return group, ip_address, gpu_name, ssh_keys, gcs_location

  def run_model(
      self,
      resource: airflow.XComArg,
      ssh_keys: airflow.XComArg,
      env: airflow.XComArg | None = None,
  ) -> DAGNode:
    """Run the GPU test in `task_test_config`.

    Args:
      gpu_name: XCom value for the GPU name (string).
      ssh_keys: And XCom value for the GPU's SSH keys (SshKeys).

    Returns:
      A DAG node that executes the model test.
    """
    return gpu.ssh_host.override(
        task_id="run_model",
        execution_timeout=self.task_test_config.timeout,
        owner=self.task_test_config.task_owner,
    )(
        resource,
        self.task_test_config.test_script,
        ssh_keys,
        env,
    )

  def post_process(
      self,
      result_location: airflow.XComArg | None = None,
  ) -> DAGNode:
    """Process metrics and metadata, and insert them into BigQuery tables.

    Returns:
      A DAG node that executes the post process.
    """
    with TaskGroup(group_id="post_process") as group:
      process_id = metric.generate_process_id.override(retries=0)()
      metric.process_metrics.override(retries=0)(
          process_id,
          self.task_test_config,
          self.task_metric_config,
          self.task_gcp_config,
          folder_location=result_location,
      )
      return group

  def clean_up(
      self, resource: airflow.XComArg, project_id: str, zone: str
  ) -> DAGNode:
    """Clean up GPU resources created by `provision`.

    Args:
      resource: an XCom value for the qualified instance name.
      project_id: project of the instance.
      zone: zone of the instance.
    Returns:
      A DAG node that deletes the resource and its owned nodes.

    Raises:
      AirflowTaskTimeout: An error occurs when execution_timeout is breached.
    """
    return gpu.delete_resource.override(group_id="clean_up")(
        resource, project_id, zone
    )

  def clean_up_existing_instance(self, ssh_keys: airflow.XComArg) -> DAGNode:
    """Clean up existing GPU resources
      - remove the one-time use generated ssh_keys.

    Args:
      ssh_keys: generated GPU's one-time use SSH keys to be removed.
    Returns:
      A DAG node that cleaned up the ssh_keys.
    """
    return gpu.clean_up_ssh_keys(
        instance_name=self.existing_instance_name,
        ssh_keys=ssh_keys,
        gcp=self.task_gcp_config,
    )


# TODO(ranran): This class is big. Let's move it to a new file.
@dataclasses.dataclass
class GpuGkeTask(BaseTask):
  """This is a class to set up tasks for GPU on a GKE cluster.

  Attributes:
    task_test_config: task configutation.
    task_gcp_config: gcp related config (e.g., zone, project) for the task.
    cluster_name: Name of the GCP cluster.
    job_create_timeout: Amount of time to wait for all pods to become active.
    task_metric_config: metric configuration (e.g., result gcs path).
  """

  task_test_config: test_config.GpuGkeTest
  task_gcp_config: gcp_config.GCPConfig
  cluster_name: str
  job_create_timeout: datetime.timedelta = datetime.timedelta(minutes=10)
  task_metric_config: metric_config.MetricConfig | None = None

  def run(self) -> DAGNode:
    """Run a test job and do post data process.

    Returns:
      A task group that runs the given test config on a GKE cluster.
    """
    with TaskGroup(
        group_id=self.task_test_config.benchmark_id, prefix_group_id=True
    ) as group:
      gcs_location = name_format.generate_gcs_folder_location(
          self.task_test_config.gcs_subfolder,
          self.task_test_config.benchmark_id,
      )

      job_body = self._get_job_manifest()

      gke_run = gke.run_job.override(group_id="run_model")(
          job_body,
          self.task_test_config,
          self.task_gcp_config,
          self.cluster_name,
          self.job_create_timeout,
          self.task_test_config.task_owner,
          gcs_location,
      )
      post_process = self.post_process(gcs_location)
      _ = gcs_location >> gke_run >> post_process
    return group

  def post_process(
      self, result_location: airflow.XComArg | None = None
  ) -> DAGNode:
    """Process metrics and metadata, and insert them into BigQuery tables.

    Returns:
      A DAG node that executes the post process.
    """
    with TaskGroup(group_id="post_process") as group:
      process_id = metric.generate_process_id.override(retries=0)()
      metric.process_metrics.override(retries=0)(
          process_id,
          self.task_test_config,
          self.task_metric_config,
          self.task_gcp_config,
          folder_location=result_location,
      )
      return group

  def _get_job_manifest(self):
    # pylint: disable=line-too-long
    accelerator = self.task_test_config.accelerator
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "generateName": f"{self.task_test_config.test_name}",
            "labels": {
                "accelerator": accelerator.name,
                "benchmarkId": self.task_test_config.benchmark_id,
            },
        },
        "spec": {
            "activeDeadlineSeconds": int(
                self.task_test_config.timeout.total_seconds()
            )
            or 3600,
            "backoffLimit": 0,
            "completionMode": "Indexed",
            "completions": self.task_test_config.num_hosts,
            "parallelism": self.task_test_config.num_hosts,
            "template": {
                "metadata": {
                    # Matches `headless-svc` in GKE cluster.
                    # See deployments directory.
                    "labels": {"headless-svc": "true"},
                },
                "spec": {
                    "subdomain": "headless-svc",
                    "nodeSelector": {
                        "cloud.google.com/gke-accelerator": (
                            accelerator.accelerator_type
                        ),
                    },
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "main",
                            "image": self.task_test_config.docker_image,
                            "imagePullPolicy": "Always",
                            "command": shlex.split(
                                self.task_test_config.setup_script
                            ),
                            "args": shlex.split(
                                self.task_test_config.test_script
                            ),
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": accelerator.count,
                                }
                            },
                            "env": [
                                {
                                    "name": "POD_NAME",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.name"
                                        }
                                    },
                                },
                                {
                                    "name": "POD_NAMESPACE",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.namespace"
                                        }
                                    },
                                },
                                {
                                    "name": "JOB_NAME",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": (
                                                "metadata.labels['job-name']"
                                            )
                                        }
                                    },
                                },
                            ],
                            "volumeMounts": [
                                {
                                    "mountPath": "/dev/shm",
                                    "name": "dshm",
                                    "readOnly": False,
                                },
                            ],
                        },
                    ],
                    "volumes": [
                        {"emptyDir": {"medium": "Memory"}, "name": "dshm"},
                    ],
                },
            },
        },
    }
