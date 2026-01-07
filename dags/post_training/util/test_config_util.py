"""Test Configuration Class utility for post training testcases."""

from dataclasses import dataclass
from enum import Enum

from dags import gcs_bucket
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs.common import SetupMode


DEFAULT_BUCKET = gcs_bucket.RL_AUTOMATION_BUCKET

# Docker images for post-training
POST_TRAINING_DOCKER_IMAGES = [
    (SetupMode.STABLE, DockerImage.MAXTEXT_POST_TRAINING_STABLE),
    (SetupMode.NIGHTLY, DockerImage.MAXTEXT_POST_TRAINING_NIGHTLY),
]


class LossAlgo(Enum):
  """Enum for RL loss algorithms."""

  GRPO = "grpo"
  GSPO = "gspo"

  @property
  def loss_name(self) -> str:
    """Returns the specific loss algorithm string used for computation."""
    return {
        LossAlgo.GRPO: "grpo",
        LossAlgo.GSPO: "gspo-token",
    }[self]


@dataclass
class RLTestConfig:
  """Configuration for RL (GRPO) training tests.

  This class holds the configuration parameters specific to reinforcement
  learning training jobs, including model parameters, infrastructure
  settings, and RL-specific training configurations.

  Attributes:
    cluster: The GKE cluster to use for training.
    accelerator: The type of accelerator (e.g., v5p-128).
    slices: List of slice numbers to test with.
    model_name: The name of the model being trained
        (e.g., llama3.1-70b).
    base_dir: Base GCS directory for outputs.
    tokenizer_path: Path to the tokenizer (HuggingFace model
        path or local path).
    load_parameters_path: GCS path to load model parameters
        from.
    loss_algos: List of loss algorithms to test (e.g., [LossAlgo.GRPO]).
    rl_config_path: Path to the RL configuration YAML file
        (relative to MaxText root).
  """

  cluster: XpkClusters
  accelerator: str
  slices: list[int]
  model_name: str
  base_dir: str
  tokenizer_path: str
  load_parameters_path: str
  loss_algos: list[LossAlgo]
  rl_config_path: str = "src/MaxText/configs/rl.yml"

  def __init__(
      self,
      cluster: XpkClusters,
      accelerator: str,
      slices: list[int],
      model_name: str,
      base_dir: str,
      tokenizer_path: str,
      load_parameters_path: str,
      loss_algos: list[LossAlgo],
      rl_config_path: str = "src/MaxText/configs/rl.yml",
  ):
    """Initializes the RL test configurations.

    Args:
      cluster: The specified cluster to be used for the test.
      accelerator: The type of accelerator (e.g., v5p-128,
          v6e-256) to use.
      slices: The number of slices to be used.
      model_name: The name of the base model being tested
          (e.g., llama3.1-70b).
      base_dir: The base GCS directory for storing outputs.
      tokenizer_path: Path to the tokenizer (HuggingFace
          model path).
      load_parameters_path: GCS path to load pretrained model
          parameters from.
      loss_algos: List of loss algorithms to test.
      rl_config_path: Path to the RL configuration YAML file
          (default: src/MaxText/configs/rl.yml).
    """
    self.cluster = cluster
    self.accelerator = accelerator
    self.slices = slices
    self.model_name = model_name
    self.base_dir = base_dir
    self.tokenizer_path = tokenizer_path
    self.load_parameters_path = load_parameters_path
    self.loss_algos = loss_algos
    self.rl_config_path = rl_config_path

  def generate_rl_training_command(
      self, loss_algo: LossAlgo, run_name: str, hf_token: str
  ) -> tuple[str]:
    """Generates the RL training command as a tuple for GKE compatibility.

    Args:
      loss_algo: The loss algorithm to use (e.g., LossAlgo.GRPO).
      run_name: The run name for the training job.
      hf_token: The HuggingFace token for authentication.

    Returns:
      A tuple containing the RL training command string.
    """
    command = (
        f"export HF_TOKEN={hf_token} && "
        "export TPU_MIN_LOG_LEVEL=0 && "
        "export TF_CPP_MIN_LOG_LEVEL=0 && "
        "export TPU_STDERR_LOG_LEVEL=0 && "
        "export JAX_PLATFORMS=proxy,cpu && "
        "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 && "
        "export ENABLE_PATHWAYS_PERSISTENCE='1' && "
        f"python -m src.MaxText.rl.train_rl "
        f"{self.rl_config_path} run_name={run_name} "
        f"model_name={self.model_name} "
        f"tokenizer_path={self.tokenizer_path} "
        f"load_parameters_path={self.load_parameters_path} "
        f"base_output_directory={self.base_dir} "
        f"loss_algo={loss_algo.loss_name}"
    )

    # Return as tuple for k8s yaml compatibility.
    return (command,)
