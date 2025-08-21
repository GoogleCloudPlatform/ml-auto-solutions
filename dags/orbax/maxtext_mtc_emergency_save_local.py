"""
A DAG to run MaxText multi-tier checkpointing tests.

This DAG performs a series of tests to save and validate checkpoints
for the MaxText model. It runs tests in two modes: one with the replicator
service enabled (Multi-tier Checkpointing). The tests are executed on a TPU
multi-pod cluster.
"""

import datetime
from dataclasses import dataclass
import posixpath
from typing import Optional

from airflow import models
from airflow.utils.task_group import TaskGroup

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import validation_util
from dags.orbax.util import checkpoint_util
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_emc_and_mtc_orbax_save_local"
DEFAULT_BUCKET = gcs_bucket.MTC_AUTOMATION_BUCKET

# Only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_ORBAX_HEAD,
)]
RAM_DISK = "/local"


@dataclass
class Checkpointing:
  """
  Represents the information of a checkpointing mechanism.

  Attributes:
    name: A unique name for the checkpointing configuration.
    enable_multi_tier_checkpointing: Indicates whether multi-tier checkpointing is enabled.
  """

  name: str
  enable_multi_tier_checkpointing: bool = True


@dataclass
class TestConfig:
  """Holds the general configuration for a checkpointing test."""

  cluster: XpkClusters
  machine_type: str
  accelerator: str
  slices: list[int]
  model_name: str
  short_id: str
  replicator_backup_time: int
  step: int
  local_checkpoint_step: int
  checkpoint_step: int
  ram_disk_size: str
  cpc_config: checkpoint_util.CheckpointConfiguration

  def __init__(
      self,
      cluster: XpkClusters,
      machine_type: str,
      accelerator: str,
      slices: list[int],
      model_name: str,
      short_id: str,
      replicator_backup_time: int,
      step: int,
      local_checkpoint_step: int,
      ram_disk_size_in_mi: str,
      checkpoint_step: Optional[int] = None,
  ):
    """
    Initializes the test configurations.

    Args:
      cluster: The specified cluster to be used for the test.
      machine_type: The type of machine (e.g., GPU, TPU).
      accelerator: The type of accelerator (e.g., GPU, TPU) to use.
      slices: The number of slices to be used.
      model_name: The name of the model being tested.
      short_id: A short identifier for the test run.
      replicator_backup_time: The allowed time for replicator takes to backup
        and store checkpoint to bucket
      step: The current step of the training process.
      local_checkpoint_step: The step interval for local checkpoints.
      checkpoint_step: The step interval for the checkpoints store in the
        bucket.
      ram_disk_size_in_mi: The size in mebibytes (Mi) about the RAM disk in the
        CSI driver. The unit is in mebibytes (Mi) but the value should be passed
        as a string with the unit, e.g., "2G" or "2048M". Defaults to "100G"".
    """

    self.cluster = cluster
    self.machine_type = machine_type
    self.accelerator = accelerator
    self.slices = slices
    self.model_name = model_name
    self.short_id = short_id
    self.replicator_backup_time = replicator_backup_time
    self.step = step
    self.local_checkpoint_step = local_checkpoint_step
    self.checkpoint_step = checkpoint_step
    self.ram_disk_size = ram_disk_size_in_mi
    self.cpc_config = checkpoint_util.CheckpointConfiguration(
        project_id=self.cluster.project,
        region=zone_to_region(self.cluster.zone),
        cluster_name=self.cluster.name,
        gcs_bucket=DEFAULT_BUCKET.removeprefix("gs://"),
        ramdisk_memory_in_mi=self.ram_disk_size,
        machine_type=self.machine_type,
    )

  def generate_workload_command(
      self,
      cp: Checkpointing,
      checkpoint_dir: str,
      out_folder: str,
      slice_number: int,
  ) -> str:
    run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{self.short_id}-{cp.name}-{slice_number}x-{self.accelerator}-{run_time}"
    return (
        "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
        "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
        "python3 -m MaxText.train MaxText/configs/base.yml "
        "remat_policy=full "
        "global_parameter_scale=1 "
        f"base_output_directory={posixpath.join(DEFAULT_BUCKET, out_folder)} "
        "dataset_type=synthetic "
        f"steps={self.step} "
        "per_device_batch_size=1 "
        "max_target_length=256 "
        f"model_name={self.model_name} "
        "per_device_batch_size=2 "
        "reuse_example_batch=1 "
        "enable_emergency_checkpoint=true "
        f"local_checkpoint_directory={checkpoint_dir} "
        f"local_checkpoint_period={self.local_checkpoint_step} "
        f"enable_multi_tier_checkpointing={cp.enable_multi_tier_checkpointing} "
        f"multi_tier_checkpointing_backup_interval_minutes={self.replicator_backup_time} "
        f"run_name={run_name}",
    )


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
    doc_md="""
      # Emergency Checkpoint Manager and Multi-tier Checkpoint Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      checkpoint saving when using both **Emergency Checkpoint Manager**
      and **Multi-tier Checkpoint** features. The flag that controls the
      behaviour of using MTC or EMC is **user_replicatior**.
      Also the steps flag controls how many steps the job will run.

      ### Prerequisites
      To run this test, you need an existing cluster with the Multi-tier
      Checkpointing configuration enabled, as well as a bucket with HNS
      (Hierarchical Namespace) enabled.

      ### Procedures
      1.  **Apply Configuration:** A Checkpoint Configuration YAML file is
      applied to the cluster, enabling Multi-tier Checkpoint (MTC) features.
      2.  **Run Maxtext Jobsets:** The DAG runs a Maxtext jobset twice.
      One run has `replicator_enabled` set to `True` (for MTC), and the
      other has it set to `False` (for Emergency Checkpoint Manager).
      3.  **Validate Checkpoints:** The DAG validates that **local checkpoints**
      are being saved correctly in the `local/` directory for both job runs.

      4.  The validation logic is the same for both the MTC and Emergency
      Checkpoint Manager job runs because their local checkpoint saving
      behavior is similar. This is why a single validation task is used for both.
    """,
    concurrency=2,
) as dag:
  # Only one set of test configurations (e.g., v5p-128) is supported at the moment.
  # Other configurations (e.g., v5e and/or v6e) may be introduced later.
  test_configs = [
      TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER_ORBAX,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-sv-loc",
          replicator_backup_time=30,
          step=100,
          local_checkpoint_step=20,
          ram_disk_size_in_mi="800000Mi",
      ),
  ]

  task_groups = []

  for checkpointing in [
      Checkpointing(
          name="mtc",  # Multi-tier Checkpointing
          enable_multi_tier_checkpointing=True,
      ),
      Checkpointing(
          name="emc",  # Emergency Checkpointing
          enable_multi_tier_checkpointing=False,
      ),
  ]:
    with TaskGroup(
        group_id=f"maxtext_{checkpointing.name}_orbax_save_local",
    ) as group:
      for mode, image in DOCKER_IMAGES:
        for test_config in test_configs:
          for slice_num in test_config.slices:
            # We conditionally set the trigger_rule on the first task.
            # If first task group failed the next one can execute.
            wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
                trigger_rule="all_done"
            )(test_config.cpc_config)
            apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)
            workload_command = test_config.generate_workload_command(
                cp=checkpointing,
                checkpoint_dir=RAM_DISK,
                out_folder=group.group_id,
                slice_number=slice_num,
            )

            start_time = validation_util.generate_timestamp()
            maxtext_chkpt_run_test = gke_config.get_gke_config(
                num_slices=slice_num,
                cluster=test_config.cluster,
                time_out_in_min=60,
                test_name=f"{test_config.short_id}-{checkpointing.name}",
                run_model_cmds=workload_command,
                docker_image=image.value,
                test_owner=test_owner.CAMILO_Q,
            ).run(
                ramdisk_directory=RAM_DISK,
                mtc_enabled=True,
                xpk_branch=BRANCH_ABHINAV_MTC,
                skip_post_process=True,
            )

            end_time = validation_util.generate_timestamp()

            total_steps = test_config.step
            k = test_config.local_checkpoint_step
            last_step = test_config.step - 1
            steps_to_validate = [*range(0, total_steps, k), last_step]

            validate_local_check_steps = (
                validation_util.validate_checkpoint_at_steps_are_saved(
                    project_id=test_config.cluster.project,
                    location=zone_to_region(test_config.cluster.zone),
                    cluster_name=test_config.cluster.name,
                    ram_disk=RAM_DISK,
                    start_time=start_time,
                    end_time=end_time,
                    steps_to_validate=steps_to_validate,
                )
            )

            (
                wait_delete_cpc
                >> apply_cpc
                >> start_time
                >> maxtext_chkpt_run_test
                >> end_time
                >> validate_local_check_steps
            )
      # Add to a list of test to chain them sequentially.
      task_groups.append(group)

  # Chain all task groups sequentially.
  for idx in range(len(task_groups) - 1):
    task_groups[idx] >> task_groups[idx + 1]
