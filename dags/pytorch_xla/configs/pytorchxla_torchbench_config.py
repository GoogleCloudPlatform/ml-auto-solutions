# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to construct configs for pytorchxla_torchbench DAG."""

import datetime
import enum
from typing import Tuple
from xlml.apis import gcp_config, metric_config, task, test_config
import dags.vm_resource as resource
from dags import test_owner


GCS_SUBFOLDER_PREFIX = test_owner.Team.PYTORCH_XLA.value


class VERSION(enum.Enum):
  NIGHTLY = enum.auto()
  R2_2 = enum.auto()
  R2_3 = enum.auto()
  R2_4 = enum.auto()
  R2_5 = enum.auto()


class VERSION_MAPPING:

  class NIGHTLY(enum.Enum):
    TORCH_XLA_TPU_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.6.0.dev-cp310-cp310-linux_x86_64.whl"
    TORCH_XLA_CUDA_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.6.0.dev-cp310-cp310-linux_x86_64.whl"
    TORCH = "torch"
    TORCHVISION = "torchvision"
    TORCHAUDIO = "torchaudio"
    TORCH_XLA_GPU_DOCKER = "us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_cuda_12.1"
    TORCH_INDEX_CPU_URL = "https://download.pytorch.org/whl/nightly/cpu"
    TORCH_INDEX_CUDA_URL = "https://download.pytorch.org/whl/nightly/cu121"
    TORCH_REPO_BRANCH = "-b main"
    TORCH_XLA_REPO_BRANCH = "-b master"

  class R2_2(enum.Enum):
    TORCH_XLA_TPU_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl"
    TORCH_XLA_CUDA_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.2.0-cp310-cp310-manylinux_2_28_x86_64.whl"
    TORCH = "torch==2.2.0"
    TORCHVISION = "torchvision==0.17.0"
    TORCHAUDIO = "torchaudio==2.2.0"
    TORCH_XLA_GPU_DOCKER = "us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.2.0_3.10_cuda_12.1"
    TORCH_INDEX_CPU_URL = "https://download.pytorch.org/whl/cpu"
    TORCH_INDEX_CUDA_URL = "https://download.pytorch.org/whl/cu121"
    TORCH_REPO_BRANCH = "-b v2.2.0"
    TORCH_XLA_REPO_BRANCH = "-b v2.2.0"

  # TODO(@siyuan): Please update the 2.3 rc to the latest.
  class R2_3(enum.Enum):
    TORCH_XLA_TPU_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.3.0-cp310-cp310-linux_x86_64.whl"
    TORCH_XLA_CUDA_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.3.0-cp310-cp310-linux_x86_64.whl"
    TORCH = "torch==2.3.0"
    TORCHVISION = "torchvision==0.18.0"
    TORCHAUDIO = "torchaudio==2.3.0"
    TORCH_XLA_GPU_DOCKER = "us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.3.0_3.10_cuda_12.1"
    TORCH_INDEX_CPU_URL = "https://download.pytorch.org/whl/test/cpu"
    TORCH_INDEX_CUDA_URL = "https://download.pytorch.org/whl/test/cu121"
    TORCH_REPO_BRANCH = "-b v2.3.0-rc12"
    TORCH_XLA_REPO_BRANCH = "-b v2.3.0-rc12"

  class R2_4(enum.Enum):
    TORCH_XLA_TPU_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.4.0-cp310-cp310-linux_x86_64.whl"
    TORCH_XLA_CUDA_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.4.0-cp310-cp310-linux_x86_64.whl"
    TORCH = "torch==2.4.0"
    TORCHVISION = "torchvision==0.19.0"
    TORCHAUDIO = "torchaudio==2.4.0"
    TORCH_XLA_GPU_DOCKER = "us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.4.0_3.10_cuda_12.1"
    TORCH_INDEX_CPU_URL = "https://download.pytorch.org/whl/test/cpu"
    TORCH_INDEX_CUDA_URL = "https://download.pytorch.org/whl/test/cu121"
    TORCH_REPO_BRANCH = "-b v2.4.0-rc8"
    TORCH_XLA_REPO_BRANCH = "-b v2.4.0-rc8"

  class R2_5(enum.Enum):
    TORCH_XLA_TPU_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.5.0rc1-cp310-cp310-linux_x86_64.whl"
    TORCH_XLA_CUDA_WHEEL = "https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.1/torch_xla-2.5.0rc1-cp310-cp310-linux_x86_64.whl"
    TORCH = "torch==2.5.0"
    TORCHVISION = "torchvision==0.19.0"
    TORCHAUDIO = "torchaudio==2.5.0"
    TORCH_XLA_GPU_DOCKER = "us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:r2.5.0_3.10_cuda_12.1"
    TORCH_INDEX_CPU_URL = "https://download.pytorch.org/whl/test/cpu"
    TORCH_INDEX_CUDA_URL = "https://download.pytorch.org/whl/test/cu121"
    TORCH_REPO_BRANCH = "-b v2.5.0-rc1"
    TORCH_XLA_REPO_BRANCH = "-b v2.5.0-rc1"


def get_version_mapping(test_version):
  """Get version dependecies based on version type.
  Args:
    output_file: If not None, model installation message will be piped to a file.

  Returns:
    Version mapping of the version type.

  Raises:
    Version type not found value error.
  """
  if test_version == VERSION.NIGHTLY:
    version_mapping = VERSION_MAPPING.NIGHTLY
  elif test_version == VERSION.R2_2:
    version_mapping = VERSION_MAPPING.R2_2
  elif test_version == VERSION.R2_3:
    version_mapping = VERSION_MAPPING.R2_3
  elif test_version == VERSION.R2_4:
    version_mapping = VERSION_MAPPING.R2_4
  elif test_version == VERSION.R2_5:
    version_mapping = VERSION_MAPPING.R2_5
  else:
    raise ValueError("version number does not exist in VERSION enum")
  return version_mapping


def set_up_torchbench_tpu(
    model_name: str = "",
    test_version: VERSION = VERSION.NIGHTLY,
    use_startup_script: bool = False,
    use_xla2: bool = False,
) -> Tuple[str]:
  """Common set up for TorchBench."""

  version_mapping = get_version_mapping(test_version)

  def model_install_cmds(output_file=None) -> str:
    """Installs torchbench models.

    Args:
      output_file: If not None, model installation message will be piped to a file.

    Returns:
      Command to install the model.
    """
    pipe_file_cmd = f" > {output_file}" if output_file else ""
    if not model_name or model_name.lower() == "all":
      return f"python install.py --continue_on_fail {pipe_file_cmd}"
    return f"python install.py models {model_name} {pipe_file_cmd}"

  install_torch_xla2_dependency = (
      (
          # TODO(piz): torch_xla2 only support nightly test at this time.
          # "pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html",
          "pip3 uninstall -y libtpu-nightly jax jaxlib",
          "cd ~/xla/experimental/torch_xla2/",
          "pip3 install --user -e .[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html",
      )
      if use_xla2
      else ()
  )

  return (
      "pip3 install -U 'setuptools>=70.0.0,<71.0.0'",
      "sudo systemctl stop unattended-upgrades",
      "sudo apt-get -y update",
      "sudo apt install -y libopenblas-base",
      "sudo apt install -y libsndfile-dev",
      "sudo apt-get install libgl1 -y",
      "pip3 install --user numpy pandas",
      (
          f"pip3 install --user --pre {version_mapping.TORCH.value} {version_mapping.TORCHVISION.value} {version_mapping.TORCHAUDIO.value} --index-url {version_mapping.TORCH_INDEX_CPU_URL.value}"
      ),
      (
          f"pip3 install --user 'torch_xla[tpu] @{version_mapping.TORCH_XLA_TPU_WHEEL.value}' -f https://storage.googleapis.com/libtpu-releases/index.html"
      ),
      "pip3 install --user psutil",
      "cd; git clone https://github.com/pytorch/benchmark.git",
      f"cd benchmark && {model_install_cmds()}",
      f"cd; git clone {version_mapping.TORCH_REPO_BRANCH.value} https://github.com/pytorch/pytorch.git",
      f"cd; git clone {version_mapping.TORCH_XLA_REPO_BRANCH.value} https://github.com/pytorch/xla.git",
      *install_torch_xla2_dependency,
  )


def get_torchbench_tpu_config(
    tpu_version: resource.TpuVersion,
    tpu_cores: int,
    project: resource.Project,
    tpu_zone: resource.Zone,
    runtime_version: resource.RuntimeVersion,
    time_out_in_min: int,
    use_xla2: bool = False,
    reserved: bool = True,
    preemptible: bool = False,
    network: str = "default",
    subnetwork: str = "default",
    test_version: VERSION = VERSION.NIGHTLY,
    model_name: str = "",
    extraFlags: str = "",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project.value,
      zone=tpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = set_up_torchbench_tpu(
      model_name, test_version, use_xla2=use_xla2
  )
  local_output_location = "~/xla/benchmarks/output/metric_report.jsonl"

  if not model_name or model_name.lower() == "all":
    run_filter = ""
  else:
    run_filter = f"--filter={model_name}"
  run_script_cmds_xla1 = (
      "export HUGGING_FACE_HUB_TOKEN=hf_AbCdEfGhIjKlMnOpQ",  # Use a fake token to bypass torchbench hf init.
      (
          "export PJRT_DEVICE=TPU && cd ~/xla/benchmarks && python experiment_runner.py"
          " --suite-name=torchbench --xla=PJRT --accelerator=tpu --progress-bar"
          f" {run_filter}"
      ),
      "rm -rf ~/xla/benchmarks/output/metric_report.jsonl",
      "python ~/xla/benchmarks/result_analyzer.py --output-format=jsonl",
      f"gsutil cp {local_output_location} {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )

  run_script_cmds_xla2 = (
      "export HUGGING_FACE_HUB_TOKEN=hf_AbCdEfGhIjKlMnOpQ",  # Use a fake token to bypass torchbench hf init.
      (
          "export PJRT_DEVICE=TPU && cd ~/xla/benchmarks && python experiment_runner.py"
          " --suite-name=torchbench --xla=PJRT --accelerator=tpu --torch-xla2=torch_export"
          f" --torch-xla2=extract_jax --progress-bar {run_filter}"
      ),
      "rm -rf ~/xla/benchmarks/output/metric_report.jsonl",
      "python ~/xla/benchmarks/result_analyzer.py --output-format=jsonl",
      f"gsutil cp {local_output_location} {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )
  run_script_cmds = run_script_cmds_xla2 if use_xla2 else run_script_cmds_xla1

  test_name = f"torchbench_{model_name}" if model_name else "torchbench_all"
  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version.value,
          network=network,
          subnetwork=subnetwork,
          reserved=reserved,
          preemptible=preemptible,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_script_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.PEI_Z,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/torchbench",
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


# Below is the setup for torchbench GPU run.
def set_up_torchbench_gpu(
    model_name: str,
    test_version: VERSION,
    nvidia_driver_version: str = "n/a",
    use_self_docker: bool = True,
    use_xla2: bool = False,
) -> Tuple[str]:
  """Common set up for TorchBench."""
  version_mapping = get_version_mapping(test_version)

  def model_install_cmds(output_file=None) -> str:
    """Installs torchbench models.

    Args:
      output_file: If not None, model installation message will be piped to a file.

    Returns:
      Command to install the model.
    """
    pipe_file_cmd = f" > {output_file}" if output_file else ""
    if not model_name or model_name.lower() == "all":
      return f"python install.py --continue_on_fail {pipe_file_cmd}"
    return f"python install.py models {model_name} {pipe_file_cmd}"

  def get_nvidia_driver_install_cmd(driver_version: str) -> str:
    nvidia_driver_install = (
        "curl -s https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py",
        # Command `apt update/upgrade` receives 403 bad gateway error when connecting to the google apt repo.
        # This can be a transient error. We use the following command to fix the issue for now.
        # TODO(piz): remove the following statement for temporary fix once the `apt update/upgrade` is removed or updated.
        "sed -i '/^\\s*run(\"apt update\")/,/^\\s*return True/ s/^/# /'  install_gpu_driver.py",
        f"sed -i 's/^\\(DRIVER_VERSION = \\).*/\\1\"{driver_version}\"/' install_gpu_driver.py",
        "sudo python3 install_gpu_driver.py --force",
        "sudo nvidia-smi",
    )
    return nvidia_driver_install

  install_torch_xla2_dependency = (
      (
          # TODO(piz): torch_xla2 only support nightly test at this time.
          "pip3 uninstall -y libtpu-nightly jax jaxlib",  # in case libtpu is installed from torch_xla
          "cd /tmp/xla/experimental/torch_xla2/",
          "pip3 install --user -e .[cuda] -f https://storage.googleapis.com/libtpu-releases/index.html",
      )
      if use_xla2
      else ()
  )

  docker_cmds_ls = (
      "apt-get update",
      "apt-get install -y libgl1",
      "apt install -y liblapack-dev libopenblas-dev",
      # Below are the required dependencies for building detectron2_* models.
      "wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusparse-dev-12-1_12.1.0.106-1_amd64.deb",
      "wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcusolver-dev-12-1_11.4.5.107-1_amd64.deb",
      "wget -nv https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libcublas-dev-12-1_12.1.3.1-1_amd64.deb",
      "dpkg -i libcusparse* libcusolver* libcublas*",
      # Below are the dependencies for benchmark data processing:
      "pip3 install --user numpy pandas",
      # torch related dependencies
      f"pip3 install --user --pre {version_mapping.TORCH.value} {version_mapping.TORCHVISION.value} {version_mapping.TORCHAUDIO.value} --index-url {version_mapping.TORCH_INDEX_CUDA_URL.value}",
      "cd /tmp/",
      "git clone https://github.com/pytorch/benchmark.git",
      "cd benchmark",
      f"{model_install_cmds()}",
      "cd /tmp/",
      f"git clone {version_mapping.TORCH_REPO_BRANCH.value} https://github.com/pytorch/pytorch.git",
      f"git clone {version_mapping.TORCH_XLA_REPO_BRANCH.value} https://github.com/pytorch/xla.git",
      *install_torch_xla2_dependency,
  )

  if not use_self_docker:
    return docker_cmds_ls
  else:
    docker_cmds = "\n".join(docker_cmds_ls)
    return (
        *get_nvidia_driver_install_cmd(nvidia_driver_version),
        "sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common",
        "distribution=$(. /etc/os-release;echo $ID$VERSION_ID)",
        "curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -",
        "curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list",
        "sudo apt-get install -y nvidia-container-toolkit",
        # Stabilize clock freqs
        "sudo nvidia-smi --lock-gpu-clocks=1200,1200",
        "sudo systemctl restart docker",
        "sudo nvidia-smi -pm 1",
        f"sudo docker pull {version_mapping.TORCH_XLA_GPU_DOCKER.value}",
        (
            "sudo docker run --shm-size 16g --gpus all -it -d --network host --name ml-automation-torchbench"
            f" {version_mapping.TORCH_XLA_GPU_DOCKER.value}"
        ),
        f"sudo docker exec -i ml-automation-torchbench /bin/bash -c '{docker_cmds}'",
    )


def get_torchbench_gpu_config(
    machine_type: resource.MachineVersion,
    image_project: resource.ImageProject,
    image_family: resource.ImageFamily,
    accelerator_type: resource.GpuVersion,
    count: int,
    gpu_zone: resource.Zone,
    time_out_in_min: int,
    use_xla2: bool = False,
    nvidia_driver_version: str = "525.125.06",
    test_version: VERSION = VERSION.NIGHTLY,
    model_name: str = "",
    extraFlags: str = "",
) -> task.GpuCreateResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=resource.Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      zone=gpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = set_up_torchbench_gpu(
      model_name,
      test_version,
      nvidia_driver_version=nvidia_driver_version,
      use_self_docker=True,
      use_xla2=use_xla2,
  )
  local_output_location = "/tmp/xla/benchmarks/output/metric_report.jsonl"

  if not model_name or model_name.lower() == "all":
    run_filter = " "
  else:
    run_filter = f" --filter={model_name}"
  cmd_list_xla1 = (
      "export PJRT_DEVICE=CUDA",
      f"export GPU_NUM_DEVICES={count}",
      "export HUGGING_FACE_HUB_TOKEN=hf_AbCdEfGhIjKlMnOpQ",  # Use a fake token to bypass torchbench hf init.
      "cd /tmp/xla/benchmarks",
      f"python experiment_runner.py --suite-name=torchbench --accelerator=cuda --progress-bar --xla=PJRT {run_filter}",
      "rm -rf /tmp/xla/benchmarks/output/metric_report.jsonl",
      "python /tmp/xla/benchmarks/result_analyzer.py --output-format=jsonl",
  )

  cmd_list_xla2 = (
      "export PJRT_DEVICE=CUDA",
      f"export GPU_NUM_DEVICES={count}",
      "export HUGGING_FACE_HUB_TOKEN=hf_AbCdEfGhIjKlMnOpQ",  # Use a fake token to bypass torchbench hf init.
      "cd /tmp/xla/benchmarks",
      f"python experiment_runner.py --suite-name=torchbench --accelerator=cuda --progress-bar --xla=PJRT --torch-xla2=torch_export --torch-xla2=extract_jax {run_filter}",
      "rm -rf /tmp/xla/benchmarks/output/metric_report.jsonl",
      "python /tmp/xla/benchmarks/result_analyzer.py --output-format=jsonl",
  )
  cmd_list = cmd_list_xla2 if use_xla2 else cmd_list_xla1
  cmds = "\n".join(cmd_list)
  run_script_cmds = (
      (
          "sudo docker exec -i $(sudo docker ps | awk 'NR==2 { print $1 }')"
          f" /bin/bash -c '{cmds}'"
      ),
      (
          "sudo docker cp $(sudo docker ps | awk 'NR==2 { print $1 }')"
          f":{local_output_location} ./"
      ),
      f"gsutil cp metric_report.jsonl {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )

  test_name = f"torchbench_{model_name}" if model_name else "torchbench_all"
  job_test_config = test_config.GpuVmTest(
      test_config.Gpu(
          machine_type=machine_type.value,
          image_family=image_family.value,
          count=count,
          accelerator_type=accelerator_type.value,
          runtime_version=resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_script_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.PEI_Z,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/torchbench",
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.GpuCreateResourceTask(
      image_project.value,
      image_family.value,
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


def get_torchbench_gpu_gke_config(
    machine_type: resource.MachineVersion,
    image_family: resource.ImageFamily,
    accelerator_type: resource.GpuVersion,
    gpu_zone: resource.Zone,
    time_out_in_min: int,
    count: int = 1,
    use_xla2: bool = False,
    test_version: VERSION = VERSION.NIGHTLY,
    project_name: resource.Project = resource.Project.CLOUD_ML_AUTO_SOLUTIONS,
    cluster_name: str = "gpu-uc1",
    model_name: str = "",
    extraFlags: str = "",
) -> task.GpuGkeTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name.value,
      zone=gpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  version_mapping = get_version_mapping(test_version)
  gpu_docker_image = version_mapping.TORCH_XLA_GPU_DOCKER.value

  local_output_location = "/tmp/xla/benchmarks/output/metric_report.jsonl"

  if not model_name or model_name.lower() == "all":
    run_filter = ""
  else:
    run_filter = f"--filter={model_name}"

  setup_script_cmds = set_up_torchbench_gpu(
      model_name,
      test_version,
      use_self_docker=False,
      use_xla2=use_xla2,
  )

  set_lib_path = (
      "export PATH=/usr/local/nvidia/bin${PATH:+:${PATH}}",
      "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}",
      "nvidia-smi",
  )

  # installs gsutil for uploading results into GCS
  install_gsutil = (
      "curl -s -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-472.0.0-linux-x86_64.tar.gz",
      "tar -xf google-cloud-cli-472.0.0-linux-x86_64.tar.gz",
      "./google-cloud-sdk/install.sh -q",
      "source /google-cloud-sdk/path.bash.inc",
      "which gsutil",
  )
  run_script_cmds_xla1 = (
      "export PJRT_DEVICE=CUDA",
      f"export GPU_NUM_DEVICES={count}",
      "export XLA_FALLBACK_CUDA=true",
      "export HUGGING_FACE_HUB_TOKEN=hf_AbCdEfGhIjKlMnOpQ",  # Use a fake token to bypass torchbench hf init.
      "cd /tmp/xla/benchmarks",
      f"python experiment_runner.py --suite-name=torchbench --accelerator=cuda --progress-bar --xla=PJRT --xla=None --dynamo=None --dynamo=openxla --dynamo=inductor {run_filter}",
      "rm -rf /tmp/xla/benchmarks/output/metric_report.jsonl",
      "python /tmp/xla/benchmarks/result_analyzer.py --output-format=jsonl",
      f"gsutil cp {local_output_location} {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )
  run_script_cmds_xla2 = (
      "export PJRT_DEVICE=CUDA",
      "export JAX_PLATFORMS=CUDA",
      f"export GPU_NUM_DEVICES={count}",
      "export HUGGING_FACE_HUB_TOKEN=hf_AbCdEfGhIjKlMnOpQ",  # Use a fake token to bypass torchbench hf init.
      "cd /tmp/xla/benchmarks",
      f"python experiment_runner.py --suite-name=torchbench --accelerator=cuda --progress-bar --xla=PJRT --torch-xla2=torch_export --torch-xla2=extract_jax {run_filter}",
      "rm -rf /tmp/xla/benchmarks/output/metric_report.jsonl",
      "python /tmp/xla/benchmarks/result_analyzer.py --output-format=jsonl",
      f"gsutil cp {local_output_location} {metric_config.SshEnvVars.GCS_OUTPUT.value}",
  )
  run_script_cmds = run_script_cmds_xla2 if use_xla2 else run_script_cmds_xla1
  command_script = ["bash", "-cxue"]
  command_script.append(
      "\n".join(
          install_gsutil + set_lib_path + setup_script_cmds + run_script_cmds
      )
  )
  test_name = (
      f"torchbench-{model_name.lower().replace('_', '-')}"
      if model_name
      else "torchbench-all"
  )
  job_test_config = test_config.GpuGkeTest(
      accelerator=test_config.Gpu(
          machine_type=machine_type.value,
          image_family=image_family.value,
          count=count,
          accelerator_type=accelerator_type.value,
      ),
      test_name=test_name,
      entrypoint_script=command_script,
      test_command="",
      timeout=datetime.timedelta(minutes=time_out_in_min),
      docker_image=gpu_docker_image,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/torchbench",
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.GpuGkeTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
      cluster_name=cluster_name,
  )
