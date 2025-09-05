"""A DAG to run MaxText EMC.
Validates the local checkpoints are restored as expected
"""

import datetime
import posixpath
from typing import Optional
from dataclasses import dataclass

from airflow import models
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags import gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import validation_util
from dags.orbax.util import checkpoint_util
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region

DAG_TEST_NAME = "maxtext_emc_orbax_res_local"
BASE_OUTPUT_DIR = gcs_bucket.MTC_AUTOMATION_BUCKET
SCHEDULE = "0 23 * * *" if composer_env.is_prod_env() else None

DOCKER_IMAGES = [(SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_JAX_ORBAX_HEAD)]
RAM_DISK = "/local"


@dataclass
class Checkpointing:
  """Represents the information of a checkpointing mechanism.

  Attributes:
    name: A unique name for the checkpointing configuration.
    use_replicator: Indicates whether a replicator is enabled.
  """

  name: str
  use_replicator: bool


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
    """Initializes the test configurations.

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
      ram_disk_size_in_mi: The size in mebibytes (Mi) about the RAM disk in the
        CSI driver. The unit is in mebibytes (Mi) but the value should be passed
        as a string with the unit, e.g., "2G" or "2048M". Defaults to "100G"".
      checkpoint_step: The step interval for the checkpoints store in the
        bucket.
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
        gcs_bucket=gcs_bucket.MTC_AUTOMATION_BUCKET.removeprefix("gs://"),
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
        f"base_output_directory={posixpath.join(BASE_OUTPUT_DIR, out_folder)} "
        "dataset_type=synthetic "
        f"steps={self.step} "
        "per_device_batch_size=1 "
        "max_target_length=256 "
        f"model_name={self.model_name} "
        "per_device_batch_size=2 "
        "reuse_example_batch=1 "
        "enable_emergency_checkpoint=true "
        f"checkpoint_period={self.checkpoint_step} "
        f"local_checkpoint_directory={checkpoint_dir} "
        f"local_checkpoint_period={self.local_checkpoint_step} "
        f"run_name={run_name}",
    )


with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 30),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "nightly",
        "orbax",
    ],
    description="DAG to verify MaxText's emergency restore from local checkpoints after a node interruption.",
    doc_md="""
      # MaxText Emergency Restore from Local Checkpoint Validation DAG

      ### Description
      This DAG validates the emergency restore capability of MaxText from local
      checkpoints. It simulates a node failure during a training job and verifies
      that the job can successfully resume from the last saved local checkpoint.
      This test is critical for ensuring the resilience of long-running training
      jobs against hardware failures.

      ### Prerequisites
      - An existing GKE cluster with the Multi-tier Checkpointing (MTC)
        configuration enabled, which provides the necessary CSI driver for
        RAM disk (`/local`).
      - A GCS bucket for storing logs and base model outputs.

      ### Procedures
      1.  **Apply MTC Configuration:** A `CheckpointingPolicyConfiguration` (CPC)
          is applied to the cluster to set up the MTC environment.
      2.  **Run MaxText with Interruption:** A MaxText training job is initiated.
          During its execution, a node interruption is simulated to trigger the
          emergency restore mechanism.
      3.  **Validate Restore:** The DAG inspects the application logs to confirm 
          that an `'emergency_restore'` event occurred.
      4.  **Validate Checkpoint Integrity:** It then verifies that the training job 
          resumed and continued to save checkpoints correctly after the restore, 
          ensuring no data was lost.
      """,
    concurrency=2,
) as dag:
  test_configs = [
      TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER_ORBAX,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-res-loc",
          replicator_backup_time=30,
          step=150,
          checkpoint_step=20,
          local_checkpoint_step=20,
          ram_disk_size_in_mi="800000Mi",
      ),
  ]

  checkpointing = Checkpointing(name="emc", use_replicator=False)

  for mode, image in DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule="all_done"
        )(test_config.cpc_config)

        apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)
        workload_command = test_config.generate_workload_command(
            cp=checkpointing,
            checkpoint_dir=RAM_DISK,
            out_folder="maxtext_emc_orbax_res_local",
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
            test_owner=test_owner.DEPP_L,
        ).run_with_node_interruption(
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
            test_owner=test_owner.DEPP_L,
        ).run(
            ramdisk_directory=RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )

        end_time = validation_util.generate_timestamp()

        validate_is_restoring = validation_util.validate_log_exist(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter="\"'event_type': 'emergency_restore'\"",
            start_time=start_time,
            end_time=end_time,
        )

        total_steps = test_config.step
        k = test_config.local_checkpoint_step
        last_step = test_config.step - 1
        steps_to_validate = [*range(0, total_steps, k), last_step]

        validate_log = validation_util.validate_checkpoint_at_steps_are_saved(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            start_time=start_time,
            end_time=end_time,
            steps_to_validate=steps_to_validate,
        )

        (
            wait_delete_cpc
            >> apply_cpc
            >> start_time
            >> maxtext_chkpt_run_test
            >> ram_disk_cleanup
            >> end_time
            >> validate_is_restoring
            >> validate_log
        )
