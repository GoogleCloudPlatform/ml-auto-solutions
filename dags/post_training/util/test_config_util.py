"""Test Configuration Class utility for post training testcases."""

from dataclasses import dataclass
from enum import Enum

from dags import gcs_bucket
from dags.common.vm_resource import (
    XpkClusters,
    DockerImage,
    Project,
    Region,
)
from dags.multipod.configs.common import SetupMode


@dataclass(frozen=True)
class VertexAIConfig:
  """Configuration for Vertex AI TensorBoard instance."""

  project_id: str
  region: str
  tensorboard_id: str


class VertexAI:
  """Vertex AI TensorBoard instances for post-training dashboards."""

  POST_TRAINING = VertexAIConfig(
      project_id=Project.CLOUD_TPU_MULTIPOD_DEV.value,
      region=Region.US_CENTRAL1.value,
      tensorboard_id="9105049192543289344",
  )


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
      self,
      loss_algo: LossAlgo,
      run_name: str,
      hf_token: str,
      num_slices: int = 1,
  ) -> tuple[str]:
    """Generates the RL training command as a tuple for GKE compatibility."""
    command_params = [
        f"run_name={run_name}",
        f"model_name={self.model_name}",
        f"tokenizer_path={self.tokenizer_path}",
        f"load_parameters_path={self.load_parameters_path}",
        f"base_output_directory={self.base_dir}",
        f"rl.loss_algo={loss_algo.loss_name}",
    ]

    num_trainer = 1
    num_samplers = max(1, num_slices - num_trainer)

    if num_slices > 1:
      command_params.extend([
          f"num_trainer_slices={num_trainer}",
          f"num_samplers_slices={num_samplers}",
          f"rollout_data_parallelism={num_samplers * 2}",
      ])

    rl_command = (
        "python -m src.MaxText.rl.train_rl "
        f"{self.rl_config_path} " + " ".join(command_params)
    )

    environment_variables = [
        f"export HF_TOKEN={hf_token}",
        "export TPU_MIN_LOG_LEVEL=0",
        "export TF_CPP_MIN_LOG_LEVEL=0",
        "export TPU_STDERR_LOG_LEVEL=0",
        "export JAX_PLATFORMS=proxy,cpu",
        "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000",
        "export ENABLE_PATHWAYS_PERSISTENCE='1'",
    ]

    if num_slices > 1:
      environment_variables.extend([
          f"export NUM_SLICES={num_samplers}",
          "export JAX_RANDOM_WEIGHTS=true",
          "export VLLM_ENABLE_V1_MULTIPROCESSING=0",
      ])

    full_command = " && ".join(environment_variables + [rl_command])
    return (full_command,)


@dataclass
class SFTTestConfig:
  """Configuration for SFT (Supervised Fine-Tuning) training tests.

  This class holds the configuration parameters specific to SFT jobs,
  including model parameters, infrastructure settings, and SFT-specific
  training configurations.

  Attributes:
    cluster: The GKE cluster to use for training.
    accelerator: The type of accelerator (e.g., v5p-128).
    slices: List of slice numbers to test with.
    model_name: The name of the model being trained (e.g., llama3.1-70b).
    steps: The number of training steps to run.
    short_id: A short identifier for the test run (e.g., 'msft').
    base_dir: Base GCS directory for outputs.
    tokenizer_path: Path to the tokenizer (HuggingFace model path or local
      path).
    load_parameters_path: GCS path to load model parameters from.
    sft_config_path: Path to the SFT configuration YAML file (relative to
      MaxText root).
  """

  cluster: XpkClusters
  accelerator: str
  slices: list[int]
  model_name: str
  steps: int
  short_id: str
  base_dir: str
  tokenizer_path: str
  load_parameters_path: str
  sft_config_path: str = "src/maxtext/configs/post_train/sft.yml"

  def __init__(
      self,
      cluster: XpkClusters,
      accelerator: str,
      slices: list[int],
      model_name: str,
      steps: int,
      short_id: str,
      base_dir: str,
      tokenizer_path: str,
      load_parameters_path: str,
      sft_config_path: str = "src/maxtext/configs/post_train/sft.yml",
  ):
    """Initializes the SFT test configurations.

    Args:
      cluster: The specified cluster to be used for the test.
      accelerator: The type of accelerator (e.g., v5p-128) to use.
      slices: The number of slices to be used.
      model_name: The name of the base model being tested (e.g.,
        llama3.1-70b).
      steps: The number of training steps to run.
      short_id: A short identifier for the test run.
      base_dir: The base GCS directory for storing outputs.
      tokenizer_path: Path to the tokenizer (HuggingFace model path).
      load_parameters_path: GCS path to load pretrained model parameters
        from.
      sft_config_path: Path to the SFT configuration YAML file (default:
        src/maxtext/configs/post_train/sft.yml).
    """
    self.cluster = cluster
    self.accelerator = accelerator
    self.slices = slices
    self.model_name = model_name
    self.steps = steps
    self.short_id = short_id
    self.base_dir = base_dir
    self.tokenizer_path = tokenizer_path
    self.load_parameters_path = load_parameters_path
    self.sft_config_path = sft_config_path

  def generate_sft_training_command(
      self, run_name: str, hf_token: str
  ) -> tuple[str]:
    """Generates the SFT training command as a tuple for GKE compatibility.

    Args:
      run_name: The run name for the training job.
      hf_token: The HuggingFace token for authentication.

    Returns:
      A tuple containing the SFT training command string.
    """
    sft_command = (
        "python3 -m MaxText.sft.sft_trainer "
        f"{self.sft_config_path} run_name={run_name} "
        f"base_output_directory={self.base_dir} "
        f"model_name={self.model_name} "
        f"load_parameters_path={self.load_parameters_path} "
        f"hf_access_token={hf_token} "
        f"tokenizer_path={self.tokenizer_path} "
        f"steps={self.steps} "
        "per_device_batch_size=1 "
        "checkpoint_storage_use_zarr3=False "
        "checkpoint_storage_use_ocdbt=False "
        "enable_single_controller=True"
    )
    command = " && ".join([
        f"export HF_TOKEN={hf_token}",
        "export JAX_PLATFORMS=proxy",
        "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000",
        "export ENABLE_PATHWAYS_PERSISTENCE=1",
        "export TPU_MIN_LOG_LEVEL=0",
        "export TF_CPP_MIN_LOG_LEVEL=0",
        "export TPU_STDERR_LOG_LEVEL=0",
        sft_command,
    ])

    # Return as tuple for k8s yaml compatibility.
    return (command,)
