# Copyright 2024 Google LLC
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

"""Utilities to construct configs for inference benchmark DAG."""


import datetime
import json
from typing import Dict
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner
from dags.multipod.configs import common
from dags.vm_resource import MachineVersion, ImageFamily, ImageProject, GpuVersion, TpuVersion, Project, RuntimeVersion, Zone

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.SOLUTIONS_TEAM.value
ROOT_DIRECTORY = "/home/ml-auto-solutions"


def get_vllm_gpu_setup_cmds():
  setup_cmds = (
      "pip install --upgrade pip",
      "sudo apt-get -y update",
      "sudo apt-get -y install python3.10-venv",
      "sudo apt-get -y install jq",
      "python -m venv .env",
      "source .env/bin/activate",
      "rm -rf vllm && git clone https://github.com/vllm-project/vllm.git",
      "cd vllm",
      # Hack - remove this
      "git checkout f2bd246c17ba67d7749a2560a30711f74cd19177",
      "pip install -e .",
      # Download dataset
      'cd .. && wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json',
      # Download benchmark
      "git clone https://github.com/GoogleCloudPlatform/ai-on-gke",
  )
  return setup_cmds


def get_vllm_tpu_setup_cmds():
  setup_cmds = (
      # Update environment and installs basic deps
      "pip install --upgrade pip",
      "sudo apt-get -y update",
      "sudo apt install -y libopenblas-base libopenblas-dev",
      "sudo apt-get -y install python3.10-venv",
      "sudo apt-get -y install jq",
      "python -m venv .env",
      "source .env/bin/activate",
      # Install vllm at head
      "rm -rf vllm && git clone https://github.com/vllm-project/vllm",
      "cd vllm",
      # Hack - remove this
      "git checkout f2bd246c17ba67d7749a2560a30711f74cd19177",
      # From https://docs.vllm.ai/en/latest/getting_started/tpu-installation.html
      "pip uninstall torch torch-xla -y",
      'export DATE="20240828"',
      'export TORCH_VERSION="2.5.0"',
      "pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl",
      "pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-${TORCH_VERSION}.dev${DATE}-cp310-cp310-linux_x86_64.whl",
      # Install JAX and Pallas.
      "pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html",
      "pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html",
      # Install other build dependencies.
      "pip install -r requirements-tpu.txt",
      # Build vLLM
      'VLLM_TARGET_DEVICE="tpu" python setup.py develop',
      # Download dataset
      'cd .. && wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json',
      # Download benchmark
      "git clone https://github.com/GoogleCloudPlatform/ai-on-gke",
  )

  return setup_cmds


def get_vllm_benchmark_cmds(model_id: str, num_chips: int, model_configs: Dict = {}):
  base_model_id = model_id.split("/")[-1]
  request_rates = model_configs["request_rates"].split(',')
  num_prompts = 1000

  run_cmds = [
    # TODO: Find a better way to do this
    "export HF_TOKEN=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",
    # Start virtual environment
    "source .env/bin/activate",
    # Start vllm in the background
    "cd vllm",
    f'vllm serve {model_id} --swap-space 16  --disable-log-requests --tensor_parallel_size={num_chips} --max-model-len=2048 &',
    # Wait for server to come up
    'sleep 600',
    "cd ..",
  ]

  metadata = {"test": "test"}
  for request_rate in request_rates:
    benchmark_cmds = [
      # Run benchmark
      f'python ai-on-gke/benchmarks/benchmark/tools/profile-generator/container/benchmark_serving.py --host localhost --port 8000 --num-prompts {num_prompts} --max-input-length 1024 --max-output-length 1024 --dataset ShareGPT_V3_unfiltered_cleaned_split.json --save-json-results --model "{model_id}" --tokenizer "{model_id}" --request-rate {request_rate} --additional-metadata-metrics-to-save "{json.dumps(metadata)}"',
      # Process result json files
      f'export OUTPUT_FORMAT="*vllm*{base_model_id}*"',
      'export BENCHMARK_OUTPUT=$(find . -name $OUTPUT_FORMAT -type f -printf "%T@ %Tc %p\n" | sort -n | head -1 | awk \'NF>1{print $NF}\')',
      "cat ${BENCHMARK_OUTPUT} >> metric_report.jsonl",
      "gsutil cp metric_report.jsonl gs://us-west4-ricliu-736a999d-bucket/logs",
      f"gsutil cp metric_report.jsonl {metric_config.SshEnvVars.GCS_OUTPUT.value}",
    ]
    run_cmds.extend(benchmark_cmds)

  # Kill background process
  run_cmds.append("pkill -P $$")

  return tuple(run_cmds)


def get_gpu_inference_gce_config(
    machine_type: MachineVersion,
    image_project: ImageProject,
    image_family: ImageFamily,
    accelerator_type: GpuVersion,
    count: int,
    gpu_zone: Zone,
    time_out_in_min: int,
    test_name: str,
    project: Project,
    network: str,
    subnetwork: str,
    model_configs: Dict = {},
):

  job_gcp_config = gcp_config.GCPConfig(
      project_name=project.value,
      zone=gpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = get_vllm_gpu_setup_cmds()

  run_model_cmds = get_vllm_benchmark_cmds(
      model_id=model_configs["model_id"],
      num_chips=count,
      model_configs=model_configs
  )

  job_test_config = test_config.GpuVmTest(
      test_config.Gpu(
          machine_type=machine_type.value,
          image_family=image_family.value,
          count=count,
          accelerator_type=accelerator_type.value,
          runtime_version=RUNTIME_IMAGE,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.RICHARD_L,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/inference_benchmark",
  )

  job_gcp_config = gcp_config.GCPConfig(
      project_name=project.value,
      zone=gpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
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


def get_tpu_inference_gce_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    backend: str,
    time_out_in_min: int,
    test_name: str,
    test_mode: common.SetupMode,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = True,
    num_slices: int = 1,
    model_configs: Dict = {},
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  set_up_cmds = get_vllm_tpu_setup_cmds()

  run_model_cmds = get_vllm_benchmark_cmds(
      model_id=model_configs["model_id"],
      num_chips=tpu_cores,
      model_configs=model_configs
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.RICHARD_L,
      num_slices=num_slices,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/inference_benchmark",
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
