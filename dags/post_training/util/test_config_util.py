"""Test Configuration Class utility for post training testcases."""

from dataclasses import dataclass

from dags import gcs_bucket
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs.common import SetupMode


DEFAULT_BUCKET = gcs_bucket.RL_AUTOMATION_BUCKET

# Docker images for GRPO/RL training
DOCKER_IMAGES_RL = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_POST_TRAINING_RL,
)]


@dataclass
class RLTestConfig:
  """Configuration for RL (GRPO) training tests.

  This class holds the configuration parameters specific to reinforcement learning
  training jobs, including model parameters, infrastructure settings, and RL-specific
  training configurations.

  Attributes:
    cluster: The GKE cluster to use for training.
    accelerator: The type of accelerator (e.g., v5p-128).
    slices: List of slice numbers to test with.
    model_name: The name of the model being trained (e.g., llama3.1-70b).
    short_id: A short identifier for the test run.
    base_dir: Base GCS directory for outputs.
    tokenizer_path: Path to the tokenizer (HuggingFace model path or local path).
    load_parameters_path: GCS path to load model parameters from.
    rl_config_path: Path to the RL configuration YAML file (relative to MaxText root).
  """

  cluster: XpkClusters
  accelerator: str
  slices: list[int]
  model_name: str
  short_id: str
  base_dir: str
  tokenizer_path: str
  load_parameters_path: str
  rl_config_path: str = "src/MaxText/configs/rl.yml"

  def __init__(
      self,
      cluster: XpkClusters,
      accelerator: str,
      slices: list[int],
      model_name: str,
      short_id: str,
      base_dir: str,
      tokenizer_path: str,
      load_parameters_path: str,
      rl_config_path: str = "src/MaxText/configs/rl.yml",
  ):
    """Initializes the RL test configurations.

    Args:
      cluster: The specified cluster to be used for the test.
      accelerator: The type of accelerator (e.g., v5p-128, v6e-256) to use.
      slices: The number of slices to be used.
      model_name: The name of the base model being tested (e.g., llama3.1-70b).
      short_id: A short identifier for the test run.
      base_dir: The base GCS directory for storing outputs.
      tokenizer_path: Path to the tokenizer (HuggingFace model path).
      load_parameters_path: GCS path to load pretrained model parameters from.
      rl_config_path: Path to the RL configuration YAML file (default: src/MaxText/configs/rl.yml).
    """
    self.cluster = cluster
    self.accelerator = accelerator
    self.slices = slices
    self.model_name = model_name
    self.short_id = short_id
    self.base_dir = base_dir
    self.tokenizer_path = tokenizer_path
    self.load_parameters_path = load_parameters_path
    self.rl_config_path = rl_config_path
