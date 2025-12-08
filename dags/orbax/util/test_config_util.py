"""Test Configuration Class utility for orbax testcases"""

import posixpath
from typing import Optional
from dataclasses import dataclass

from airflow.exceptions import AirflowFailException
from absl import logging
import math
import re

from dags import gcs_bucket
from xlml.utils.gke import zone_to_region
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.orbax.util import checkpoint_util
from dags.multipod.configs.common import SetupMode


DEFAULT_BUCKET = gcs_bucket.ORBAX_AUTOMATION_BUCKET_EUROPE_WEST4
DEFAULT_RAM_DISK = "/local"

# Only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_ORBAX_HEAD,
)]


# Valid models and sizes for current Maxtext Repository.
MODELS = {
    "deepseek2",
    "deepseek3",
    "gemma",
    "gemma2",
    "gemma3",
    "gpt",
    "gpt3",
    "llama2",
    "llama3",
    "llama3.1",
    "llama3.3",
    "llama4",
    "mistral",
    "qwen3",
}


@dataclass
class Checkpointing:
  """Configuration for checkpointing mechanisms in MaxText training.

  This class defines the checkpointing behavior for training jobs, including
  emergency checkpointing and multi-tier checkpointing options.

  Attributes:
    name: A unique identifier for this checkpointing configuration (e.g., 'mtc', 'emc', 'reg').
    enable_multi_tier_checkpointing: Whether to enable multi-tier checkpointing
      with replicator service for automatic backup to GCS.
    enable_emergency_checkpoint: Whether to enable emergency checkpointing
      for local recovery in case of training interruptions. Defaults to True.
  """

  name: str
  enable_multi_tier_checkpointing: bool
  enable_emergency_checkpoint: bool = True


class TestConfig:
  """Holds the general configuration for a checkpointing test."""

  cluster: XpkClusters
  machine_type: str
  accelerator: str
  slices: list[int]
  model_name: str
  short_id: str
  multi_tier_checkpointing_backup_interval_minutes: Optional[int]
  steps: int
  local_checkpoint_period: Optional[int]
  checkpoint_period: Optional[int]
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
      steps: int,
      base_dir: str,
      multi_tier_checkpointing_backup_interval_minutes: Optional[int] = 0,
      local_checkpoint_period: Optional[int] = 0,
      checkpoint_period: Optional[int] = 10_000,
  ):
    """Initializes the test configurations.

    Args:
      cluster: The specified cluster to be used for the test.
      machine_type: The type of machine (e.g., GPU, TPU).
      accelerator: The type of accelerator (e.g., GPU, TPU) to use.
      slices: The number of slices to be used.
      model_name: The name of the model being tested.
      short_id: A short identifier for the test run.
      steps: The current step of the training process.
      base_dir: The base directory for storing checkpoints and outputs.
      multi_tier_checkpointing_backup_interval_minutes: Optional. The allowed time for replicator takes to backup
        and store checkpoint to bucket.
      local_checkpoint_period: Optional. The step interval for local checkpoints.
      checkpoint_period: Optional. The step interval for the checkpoints store in the
        bucket.
    """

    self.cluster = cluster
    self.machine_type = machine_type
    self.accelerator = accelerator
    self.slices = slices
    self.model_name = model_name
    self.short_id = short_id
    self.multi_tier_checkpointing_backup_interval_minutes = (
        multi_tier_checkpointing_backup_interval_minutes
    )
    self.steps = steps
    self.local_checkpoint_period = local_checkpoint_period
    self.checkpoint_period = checkpoint_period
    self.ram_disk_size = f"{self._get_disk_size(slice_num=max(self.slices), mode='mib',multiplier=60)}Mi"
    self.base_dir = base_dir
    self.cpc_config = checkpoint_util.CheckpointConfiguration(
        project_id=self.cluster.project,
        region=zone_to_region(self.cluster.zone),
        cluster_name=self.cluster.name,
        gcs_bucket=DEFAULT_BUCKET.removeprefix("gs://"),
        ramdisk_memory_in_mi=self.ram_disk_size,
        machine_type=self.machine_type,
    )

  def generate_step_to_validate(self, is_local: bool) -> list[int]:
    total_steps = self.steps
    k = self.local_checkpoint_period if is_local else self.checkpoint_period
    last_step = self.steps - 1
    return [*range(0, total_steps, k), last_step]

  def generate_workload_command(
      self,
      checkpoint_dir: str,
      out_folder: str,
      run_name: str,
      slice_num: int,
      enable_multi_tier_checkpointing: bool,
      enable_emergency_checkpoint: bool = True,
      enable_single_replica_ckpt_restoring: bool = False,
  ) -> tuple[str]:
    tpu_premmapped_size = self._get_disk_size(slice_num, mode="bytes")
    logging.info(f"Checkpoint Size per TPU: {tpu_premmapped_size}")

    # Base command with required parameters
    command = (
        f"export TPU_PREMAPPED_BUFFER_SIZE={tpu_premmapped_size} && "
        f"export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES={tpu_premmapped_size} && "
        "python3 -m MaxText.train MaxText/configs/base.yml "
        "remat_policy=full "
        "global_parameter_scale=1 "
        f"base_output_directory={posixpath.join(self.base_dir, out_folder)} "
        "dataset_type=synthetic "
        f"steps={self.steps} "
        "max_target_length=256 "
        f"model_name={self.model_name} "
        "per_device_batch_size=2 "
        "reuse_example_batch=1 "
        f"enable_emergency_checkpoint={enable_emergency_checkpoint} "
        f"checkpoint_period={self.checkpoint_period} "
        f"enable_single_replica_ckpt_restoring={enable_single_replica_ckpt_restoring} "
        f"run_name={run_name} "
    )

    # Add emergency checkpoint parameters only if enabled
    if enable_emergency_checkpoint:
      command += f"local_checkpoint_directory={checkpoint_dir} "
      if self.local_checkpoint_period and self.local_checkpoint_period > 0:
        command += f"local_checkpoint_period={self.local_checkpoint_period} "

    # Add multi-tier checkpointing parameters only if enabled and emergency checkpoint is enabled
    if enable_emergency_checkpoint and enable_multi_tier_checkpointing:
      command += (
          f"enable_multi_tier_checkpointing={enable_multi_tier_checkpointing} "
      )
      if (
          self.multi_tier_checkpointing_backup_interval_minutes
          and self.multi_tier_checkpointing_backup_interval_minutes > 0
      ):
        command += f"multi_tier_checkpointing_backup_interval_minutes={self.multi_tier_checkpointing_backup_interval_minutes} "

    # Return as tuple for k8s yaml compatibility - GKE config expects a list of commands
    return (command,)

  def _get_disk_size(
      self, slice_num: int, mode="bytes", multiplier: float = 1
  ) -> float | int:
    """Calculates disk size for a model checkpoint."""
    model_pattern = r"^(.*)-([0-9.]+b)$"
    model_pattern = r"^(.*)-([^-]+)$"
    match_model = re.match(model_pattern, self.model_name)
    match_chips = re.match(model_pattern, self.accelerator)
    if match_model and match_chips:
      model_name = match_model.group(1)
      size_model = match_model.group(2)[:-1]
      num_chips = match_chips.group(2)
    else:
      raise ValueError("Failed to parse model_name or accelerator format.")

    if model_name not in MODELS:
      raise AirflowFailException(
          f"Model '{model_name}' not supported. Please choose from: {sorted(MODELS)}"
      )

    total_checkpoint = int(size_model) * 10
    replicas_per_node = int(num_chips) / slice_num
    checkpoint_size_in_gb = (total_checkpoint / replicas_per_node) * multiplier

    match mode:
      case "bytes":
        unaligned_disk_size_bytes = math.ceil(
            checkpoint_size_in_gb * 1_000_000_000
        )
        return self._align_to_page_size(unaligned_disk_size_bytes)
      case "mib":
        return math.ceil((checkpoint_size_in_gb * 1_000_000_000) / (2**20))
      case _:
        raise ValueError(
            f"Invalid mode '{mode}'. Supported modes are 'bytes' or 'mib'."
        )

  def _align_to_page_size(self, size: int, page_size: int = 4096) -> int:
    """Rounds a size up to the nearest multiple of the page size."""
    return int(math.ceil(size / page_size) * page_size)
