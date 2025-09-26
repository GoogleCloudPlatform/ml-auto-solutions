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


DEFAULT_BUCKET = gcs_bucket.MTC_AUTOMATION_BUCKET
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
  """Represents the information of a checkpointing mechanism.

  Attributes:
    name: A unique name for the checkpointing configuration.
    en: Indicates whether a replicator is enabled.
  """

  name: str
  enable_multi_tier_checkpointing: bool


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
    self.ram_disk_size = f"{self._get_disk_size(slice_num=max(self.slices), mode='mib',multiplier=60)}Mi"
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
      slice_num: int,
      enable_multi_tier_checkp: bool,
  ) -> str:
    tpu_premmapped_size = self._get_disk_size(slice_num, mode="bytes")
    logging.info(f"Checkpoint Size per TPU: {tpu_premmapped_size}")
    return (
        f"export TPU_PREMAPPED_BUFFER_SIZE={tpu_premmapped_size} && "
        f"export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES={tpu_premmapped_size} && "
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

  def _get_disk_size(
      self, slice_num: int, mode="bytes", multiplier: float = 1
  ) -> float | int:
    """Calculates disk size for a model checkpoint."""
    try:
      model_pattern = r"^(.*)-([0-9.]+b)$"
      model_pattern = r"^(.*)-([^-]+)$"
      match_model = re.match(model_pattern, self.model_name)
      match_chips = re.match(model_pattern, self.accelerator)
      if match_model and match_chips:
        model_name = match_model.group(1)
        size_model = match_model.group(2)[:-1]
        num_chips = match_chips.group(2)
    except (AttributeError, IndexError) as e:
      raise ValueError("Invalid model_name or accelerator format.") from e

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
