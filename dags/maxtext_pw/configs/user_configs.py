import dataclasses

@dataclasses.dataclass
class XpkClusterConfig:
  """Holds details related to a XPK cluster to run workloads on."""

  cluster_name: str
  project: str
  zone: str
  device_type: str

@dataclasses.dataclass
class PathwaysConfig:
  server_image: str = None
  proxy_server_image: str = None
  runner_image: str = None
  colocated_python_sidecar_image: str = None
  server_flags: str = ''
  proxy_flags: str = ''
  worker_flags: str = ''
  headless: bool = False

@dataclasses.dataclass
class UserConfig:
  """The default configuration can be modified here."""
  # gcp configuration
  user: str = 'user_name'
  cluster_name: str = 'test-v5e-32-cluster'
  project: str = 'cloud-tpu-cluster'
  zone: str = 'us-south1-a'
  device_type: str = 'v5litepod-32'

  # Images for env
  server_image: str = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:latest'
  proxy_image: str = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:latest'
  runner: str = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest'
  colocated_python_image: str = None

  # model configuration
  benchmark_steps: int = 20
  selected_model_framework: list[str] = dataclasses.field(default_factory=list)
  selected_model_names: list[str] = dataclasses.field(default_factory=list)
  num_slices_list: list[int] = dataclasses.field(default_factory=lambda: [2])
  
  xpk_path: str = '~/xpk'

  def __post_init__(self):
    """Automatically generate derived attributes after the object is created."""
    self.cluster_config = XpkClusterConfig(
        cluster_name=self.cluster_name,
        project=self.project,
        zone=self.zone,
        device_type=self.device_type,
    )
    
    self.region = '-'.join(self.zone.split('-')[:-1])
    self.headless = False

    self.pathways_config = PathwaysConfig(
        server_image=self.server_image,
        proxy_server_image=self.proxy_image,
        runner_image=self.runner,
        colocated_python_sidecar_image=self.colocated_python_image,
        headless=self.headless,

        server_flags="",
        proxy_flags="",
        worker_flags="",
    )
    self.headless_workload_name = f'{self.user[:3]}-headless'
    self.base_output_directory = f'gs://{self.user}-{self.region}/{self.user}-'

    # # Iterate through the list of user-selected model frameworks, validating each one
    # for model_framework in self.selected_model_framework:
    #     if model_framework not in ms.AVAILABLE_MODELS_FRAMEWORKS:
    #         raise ValueError(
    #             f"Model framework '{model_framework}' not available. "
    #             f"Available model frameworks are: {list(ms.AVAILABLE_MODELS_FRAMEWORKS)}"
    #         )
        
    # # Initialize the model_set list to store the user's selected model configurations
    # device_base_type = self.device_type.split('-')[0]
    # if device_base_type not in ms.AVAILABLE_MODELS_NAMES:
    #     raise ValueError(f"Unknown device type: {device_base_type}")

    # # Iterate through the list of user-selected model names, validating each one
    # for model_name in self.selected_model_names:
    #     if model_name not in ms.AVAILABLE_MODELS_NAMES[device_base_type]:
    #         raise ValueError(
    #             f"Model name '{model_name}' not available for device type '{device_base_type}'. "
    #             f"Available model names are: {list(ms.AVAILABLE_MODELS_NAMES[device_base_type].keys())}"
    #         )
    
    # Build the model configuration
    self.models = {}
    for model_framework in self.selected_model_framework:
        self.models[model_framework] = []
        for model_name in self.selected_model_names:
            self.models[model_framework].append(model_name)

if __name__ == '__main__':
  user_config = UserConfig(
      user='lidanny',
      cluster_name='bodaborg-v6e-256-lcscld-c',
      project='tpu-prod-env-one-vm',
      zone='southamerica-west1-a',
      device_type='v6e-256',
      runner='gcr.io/tpu-prod-env-one-vm/lidanny_latest',
      benchmark_steps=20,
      num_slices_list=[2],
      selected_model_framework=['pathways', 'mcjax'],
      selected_model_names=['llama3_1_8b_8192', "llama3_1_70b_8192"]
  )
  
  # Access the generated attributes
  for key, value in user_config.__dict__.items():
    print(f'{key}: {value}')