import posixpath
from typing import Optional
from dataclasses import dataclass


from dags import gcs_bucket
from xlml.utils.gke import zone_to_region
from dags.common.vm_resource import XpkClusters
from dags.orbax.util import checkpoint_util

DEFAULT_BUCKET = gcs_bucket.MTC_AUTOMATION_BUCKET
DEFAULT_RAM_DISK = "/local"


@dataclass
class Checkpointing:
  """Represents the information of a checkpointing mechanism.

  Attributes:
    name: A unique name for the checkpointing configuration.
    en: Indicates whether a replicator is enabled.
  """

  name: str
  enable_multi_tier_checkpointing: bool


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
  base_dir: str
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
      base_dir: str,
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
    self.base_dir = base_dir
    self.cpc_config = checkpoint_util.CheckpointConfiguration(
        project_id=self.cluster.project,
        region=zone_to_region(self.cluster.zone),
        cluster_name=self.cluster.name,
        gcs_bucket=gcs_bucket.MTC_AUTOMATION_BUCKET.removeprefix("gs://"),
        ramdisk_memory_in_mi=self.ram_disk_size,
        machine_type=self.machine_type,
    )

  def generate_step_to_validate(self, is_local: bool) -> list[int]:
    total_steps = self.step
    k = self.local_checkpoint_step if is_local else self.checkpoint_step
    last_step = self.step - 1
    return [*range(0, total_steps, k), last_step]

  def generate_workload_command(
      self,
      checkpoint_dir: str,
      out_folder: str,
      run_name: str,
      enable_multi_tier_checkp: bool,
  ) -> str:
    return (
        "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
        "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
        "python3 -m MaxText.train MaxText/configs/base.yml "
        "remat_policy=full "
        "global_parameter_scale=1 "
        f"base_output_directory={posixpath.join(self.base_dir, out_folder)} "
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
        f"enable_multi_tier_checkpointing={enable_multi_tier_checkp} "
        f"multi_tier_checkpointing_backup_interval_minutes={self.replicator_backup_time} "
        f"run_name={run_name}",
    )
