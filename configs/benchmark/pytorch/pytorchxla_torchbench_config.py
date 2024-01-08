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

from typing import Tuple
from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner, vm_resource

PROJECT_NAME = vm_resource.PROJECT_CLOUD_ML_AUTO_SOLUTIONS
RUNTIME_IMAGE = vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value
IS_TPU_RESERVED = True


# TODO(ranran or PyTroch/XLA team): this is an example for benchmark test with hardcode compatible versions,
# we need to dynamically generate date on daily basis.
def set_up_torchbench_tpu(model_name: str = "") -> Tuple[str]:
  """Common set up for TorchBench."""

  def model_install_cmds() -> str:
    if not model_name or model_name.lower() == "all":
      return "python install.py --continue_on_fail"
    return f"python install.py models {model_name}"

  return (
      "pip install -U setuptools",
      "sudo systemctl stop unattended-upgrades",
      "sudo apt-get -y update",
      "sudo apt install -y libopenblas-base",
      "sudo apt install -y libsndfile-dev",
      "sudo apt-get install libgl1 -y",
      "sudo chmod 777 /usr/local/lib/python3.10/dist-packages/",
      "sudo chmod 777 /usr/local/bin/",
      "pip install numpy pandas",
      (
          "pip install --user --pre torchvision torchaudio torchtext -i"
          " https://download.pytorch.org/whl/nightly/cpu"
      ),
      (
          "pip install --user 'torch_xla[tpuvm] @"
          " https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl'"
      ),
      "pip install psutil",
      "cd; git clone https://github.com/pytorch/benchmark.git",
      f"cd benchmark && {model_install_cmds()}",
      "cd; git clone https://github.com/pytorch/xla.git",
  )


# TODO(ranran or PyTroch/XLA team) & notes:
# 1) If you want to run all models, do not pass in model_name
# 2) All filters of benchmark can be passed via extraFlags
def get_torchbench_tpu_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    model_name: str = "",
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    extraFlags: str = "",
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = set_up_torchbench_tpu(model_name)
  local_output_location = "~/xla/benchmarks/output/metric_report.jsonl"
  gcs_location = (
      f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/torchbench_config/metric_report_tpu.jsonl"
  )
  if not model_name or model_name.lower() == "all":
    run_filter = " "
  else:
    run_filter = f" --filter={model_name} "
  run_script_cmds = (
      (
          "cd ~/xla/benchmarks && python experiment_runner.py"
          " --suite-name=torchbench --xla=PJRT --accelerator=tpu --progress-bar"
          f" {run_filter}"
      ),
      "rm -rf ~/xla/benchmarks/output/metric_report.jsonl",
      "python ~/xla/benchmarks/result_analyzer.py --output-format=jsonl",
      f"gsutil cp {local_output_location} {gcs_location}",
  )

  test_name = f"torchbench_{model_name}" if model_name else "torchbench_all"
  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          network=network,
          subnetwork=subnetwork,
          reserved=IS_TPU_RESERVED,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_script_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.PEI_Z,
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig(
          file_location=gcs_location,
      )
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )


# Below is the setup for torchbench GPU run.
def set_up_torchbench_gpu(model_name: str = "") -> Tuple[str]:
  """Common set up for TorchBench."""

  # TODO(piz): There is issue with driver install through fabric.
  # Currently we use pre-installed driver to avoid driver reinstall.
  def model_install_cmds() -> str:
    if not model_name or model_name.lower() == "all":
      return "python install.py --continue_on_fail"
    return f"python install.py models {model_name}"

  nvidia_install_clean = (
      "sudo /usr/bin/nvidia-uninstall",
      (
          'sudo apt-get -y --purge remove "*cuda*" "*cublas*" "*cufft*"'
          ' "*cufile*" "*curand*" "*cusolver*" "*cusparse*" "*gds-tools*"'
          ' "*npp*" "*nvjpeg*" "nsight*" "*nvvm*"'
      ),
      'sudo apt-get -y --purge remove "*nvidia*" "libxnvctrl*"',
      "sudo apt-get -y autoremove",
      "sudo rm -rf /usr/local/cuda*",
  )
  nvidia_driver_install = (
      ("lsof -n -w /dev/nvidia* | awk '{print $2}' | sort -u | xargs -I {}" " kill {}"),
      (
          "wget -q"
          " https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run"
      ),
      (
          "sudo service lightdm stop ; sudo sh cuda_12.1.0_530.30.02_linux.run"
          " --silent --driver --toolkit --no-drm --override"
      ),
      "cat /var/log/nvidia-installer.log",
  )
  docker_cmds = (
      " apt-get update && apt-get install -y libgl1 &&"
      " pip install numpy pandas &&"
      " pip install --pre torchvision torchaudio -i"
      " https://download.pytorch.org/whl/nightly/cu121 &&"
      " cd /tmp/ && git clone https://github.com/pytorch/benchmark.git &&"
      f" cd benchmark && {model_install_cmds()} &&"
      " cd /tmp/ && git clone https://github.com/pytorch/xla.git"
  )
  return (
      "sudo apt-get install -y nvidia-container-toolkit",
      "sudo nvidia-smi -pm 1",
      (
          "sudo docker pull"
          " us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1"
      ),
      (
          "sudo docker run --gpus all -it -d --network host "
          " us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.8_cuda_12.1"
      ),
      (
          "sudo docker exec -i $(sudo docker ps | awk 'NR==2 { print $1 }')"
          f" /bin/bash -c '{docker_cmds}'"
      ),
  )


def get_torchbench_gpu_config(
    machine_type: str,
    image_project: str,
    image_family: str,
    accelerator_type: str,
    count: int,
    gpu_zone: str,
    time_out_in_min: int,
    model_name: str = "",
    extraFlags: str = "",
) -> task.GpuCreateResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=gpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = set_up_torchbench_gpu(model_name)
  local_output_location = "/tmp/xla/benchmarks/output/metric_report.jsonl"
  gcs_location = (
      f"{gcs_bucket.BENCHMARK_OUTPUT_DIR}/torchbench_config/metric_report_gpu.jsonl"
  )

  if not model_name or model_name.lower() == "all":
    run_filter = " "
  else:
    run_filter = f" --filter={model_name} "
  cmds = (
      f" export PJRT_DEVICE=CUDA && export GPU_NUM_DEVICES={count} &&"
      " cd /tmp/xla/benchmarks &&"
      " python experiment_runner.py  --suite-name=torchbench --accelerator=cuda"
      f" --progress-bar --xla=PJRT --xla=None {run_filter} &&"
      " rm -rf /tmp/xla/benchmarks/output/metric_report.jsonl &&"
      " python /tmp/xla/benchmarks/result_analyzer.py --output-format=jsonl"
  )
  run_script_cmds = (
      (
          "sudo docker exec -i $(sudo docker ps | awk 'NR==2 { print $1 }')"
          f" /bin/bash -c '{cmds}'"
      ),
      (
          "sudo docker cp $(sudo docker ps | awk 'NR==2 { print $1 }')"
          f":{local_output_location} ./"
      ),
      f"gsutil cp metric_report.jsonl {gcs_location}",
  )

  test_name = f"torchbench_{model_name}" if model_name else "torchbench_all"
  job_test_config = test_config.GpuVmTest(
      test_config.Gpu(
          machine_type=machine_type,
          image_family=image_family,
          count=count,
          accelerator_type=accelerator_type,
          runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE.value,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_script_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.PEI_Z,
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig(
          file_location=gcs_location,
      )
  )

  return task.GpuCreateResourceTask(
      image_project,
      image_family,
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
