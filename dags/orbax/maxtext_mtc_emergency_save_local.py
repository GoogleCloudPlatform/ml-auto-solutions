"""
A DAG to run MaxText multi-tier checkpointing tests.

This DAG performs a series of tests to save and validate checkpoints
for the MaxText model. It runs tests in two modes: one with the replicator
service enabled (Multi-tier Checkpointing). The tests are executed on a TPU multi-pod cluster.
"""

import datetime
from dataclasses import dataclass
import posixpath

from airflow import models
from airflow.utils.task_group import TaskGroup

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import logging_mtc as log
from dags.orbax.util import multi_tier_checkpoint_util as mtc
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_emc_and_mtc_orbax_save_local"
BASE_OUTPUT_DIRECTORY = gcs_bucket.MTC_AUTOMATION_BUCKET

# For this DAG, only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
)]
RAM_DISK = "/local"


@dataclass
class Testcase:
  """
  Represents the information of a checkpointing mechanism.

  Attributes:
    name: A unique name for the checkpointing configuration.
    use_replicator: Indicates whether a replicator is enabled for this test case.
  """

  name: str
  use_replicator: bool


@dataclass
class Testconfig:
  """Holds the general configuration for a checkpointing test.

  Attributes:
    cluster: The specified cluster to be used for the test.
    machine_type: The type of machine (e.g., GPU, TPU).
    accelerator: The type of accelerator (e.g., GPU, TPU) to use.
    slices: The number of slices to be used.
    model_name: The name of the model being tested.
    short_id (str): A short id to be used for naming the test run.
    replicator_min: The time the replicator takes to backup checkpoints to bucket.
    step: The current step of the training process.
    local_checkpoint_step: The step interval for local checkpoints.
    checkpoint_step: The step interval for the checkpoints store in the bucket.
    ram_disk_mi: The size about the RAM disk in the CSI driver, in Mi.
  """

  cluster: XpkClusters
  machine_type: str
  accelerator: str
  slices: list[int]
  model_name: str
  short_id: str
  replicator_min: int
  step: int
  local_checkpoint_step: int
  checkpoint_step: int
  ram_disk_mi: str


with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 12),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "emergency_checkpoint_manager",
        "multitier_checkpointing",
        "nightly",
        "orbax",
    ],
    description="A DAG to run MaxText multi-tier checkpointing tests.",
    doc_md="",
    concurrency=2,
    params={},
) as dag:
  test_configs = [
      Testconfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER_ORBAX,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-sv",
          replicator_min=30,
          step=100,
          local_checkpoint_step=20,
          checkpoint_step=25,
          ram_disk_mi="800000Mi",
      ),
  ]
  tests_to_run_seq = []
  # Individual test cases for Multi-tier Checkpointing and  Emergency Checkpointing
  for tc in [
      Testcase(name="mtc", use_replicator=True),
      Testcase(name="emc", use_replicator=False),
  ]:
    folder = f"maxtext_{tc.name}_orbax_save_local"
    run_dir_out = posixpath.join(BASE_OUTPUT_DIRECTORY, folder)
    with TaskGroup(group_id=f"maxtext_{tc.name}_orbax_save_local") as task:
      for mode, image in DOCKER_IMAGES:
        for test_config in test_configs:
          cpc_conf = mtc.CheckpointConfiguration(
              project_id=test_config.cluster.project,
              region=zone_to_region(test_config.cluster.zone),
              cluster_name=test_config.cluster.name,
              gcs_bucket=gcs_bucket.MTC_AUTOMATION_BUCKET.removeprefix("gs://"),
              ram_disk_memory_in_mi=test_config.ram_disk_mi,
              machine_type=test_config.machine_type,
          )
          for slice_num in test_config.slices:
            # We conditionally set the trigger_rule on the first task.
            # If first task group failed the next one can execute.
            init_delete_cpc = mtc.initiate_cpc_deletion(cpc_conf)
            wait_delete_cpc = mtc.wait_for_cpc_deletion.override(
                trigger_rule="all_done"
            )(cpc_conf)
            apply_cpc = mtc.apply_cpc(cpc_conf)
            run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            run_name = f"{test_config.short_id}-{tc.name}-{slice_num}x-{test_config.accelerator}-{run_time}"
            workload_command = (
                "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
                "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
                "python3 -m MaxText.train MaxText/configs/base.yml remat_policy=full "
                f"global_parameter_scale=1 base_output_directory={run_dir_out} "
                f"dataset_type=synthetic steps={test_config.step} per_device_batch_size=1 "
                f"max_target_length=256 model_name={test_config.model_name} per_device_batch_size=2 "
                "reuse_example_batch=1 enable_emergency_checkpoint=true "
                f"local_checkpoint_directory={RAM_DISK} local_checkpoint_period={test_config.local_checkpoint_step} "
                f"use_replicator_service={tc.use_replicator} replicator_backup_interval_minutes={test_config.replicator_min} "
                f"run_name={run_name}",
            )

            start_time = log.generate_timestamp()
            maxtext_chkpt_run_test = gke_config.get_gke_config(
                num_slices=slice_num,
                cluster=test_config.cluster,
                time_out_in_min=60,
                test_name=f"{test_config.short_id}-{tc.name}",
                run_model_cmds=workload_command,
                docker_image=image.value,
                test_owner=test_owner.CAMILO_Q,
            ).run(
                ramdisk_directory=RAM_DISK,
                mtc_enabled=True,
                xpk_branch=BRANCH_ABHINAV_MTC,
                skip_post_process=True,
            )

            cleanup_command = (f"rm -rf {RAM_DISK}/*",)
            ram_disk_cleanup = gke_config.get_gke_config(
                num_slices=slice_num,
                cluster=test_config.cluster,
                time_out_in_min=60,
                test_name=f"{test_config.short_id}-cl",
                run_model_cmds=cleanup_command,
                docker_image=image.value,
                test_owner=test_owner.CAMILO_Q,
            ).run(
                ramdisk_directory=RAM_DISK,
                mtc_enabled=True,
                xpk_branch=BRANCH_ABHINAV_MTC,
                skip_post_process=True,
            )

            end_time = log.generate_timestamp()
            vali_step = test_config.step - 1
            vali_step_list = [
                i
                for i in range(0, vali_step, test_config.local_checkpoint_step)
            ]
            vali_step_list.append(vali_step)

            # Here we are looking for the string '(blocking + background)'.
            # We will compare expected steps with the ones we found when query this regex. Should be the same
            # If for some reason the restore start from 0 this task will fail because len(valid_step_list) != len(founded_steps)
            validate_log = log.validate_log_with_step(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                text_filter="(blocking + background).",
                start_time=start_time,
                end_time=end_time,
                vali_step_list=vali_step_list,
            )

            (
                init_delete_cpc
                >> wait_delete_cpc
                >> apply_cpc
                >> start_time
                >> maxtext_chkpt_run_test
                >> ram_disk_cleanup
                >> end_time
                >> validate_log
            )
    # Add to a global list of test to be run in a sequential way
    tests_to_run_seq.append(task)

  # Chain the task groups sequentially
  for idx_test in range(len(tests_to_run_seq) - 1):
    tests_to_run_seq[idx_test] >> tests_to_run_seq[idx_test + 1]
