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

"""Utilities to construct configs for TensorRT-LLM inference DAG."""

from typing import Tuple
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner, vm_resource
from dags.vm_resource import GpuVersion, Project, RuntimeVersion

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
# TODO is this default ?
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value

def get_tensorrt_llm_gpu_config(
    machine_type: vm_resource.MachineVersion,
    image_project : vm_resource.ImageProject,
    image_family: vm_resource.ImageFamily,
    accelerator_type: vm_resource.GpuVersion,
    count: int,
    gpu_zone : vm_resource.Zone,
    time_out_in_min: int,
    test_name: str
)-> task.GpuCreateResourceTask:

  set_up_cmds = (
    "pip install --upgrade pip",
    # Install Nvidia driver.
    "wget -c https://us.download.nvidia.com/tesla/550.54.15/NVIDIA-Linux-x86_64-550.54.15.run",
    # TODO: always same version ?
    "chmod u+x NVIDIA-Linux-x86_64-550.54.15.run",
    "sudo ./NVIDIA-Linux-x86_64-550.54.15.run -x-module-path=/usr/lib/xorg/modules --ui=none -x-library-path=/usr/lib -q",
    # Install TensorRT-LLM.
    "sudo apt-get update",
    "sudo apt-get -y install git git-lfs",
    "git clone https://github.com/NVIDIA/TensorRT-LLM.git",
    "cd TensorRT-LLM",
    "git submodule update --init --recursive",
    "git lfs install",
    "git lfs pull",
    "sudo nvidia-smi -pm 1",
    "make -C docker release_build",
    "make -C docker release_run DOCKER_RUN_ARGS='--detach' RUN_CMD='sleep infinity'",
  )

  jsonl_output_path = "metric_report.jsonl"
  jsonl_converter_py_lines = (
  'import sys, csv, jsonlines, glob',
  'csv_files = glob.glob(\\"*.csv\\")',
  'csv_metric_path = csv_files[0]',
  'def make_json(csv_path, jsonl_path):',
  '  data = dict()',
  '  data[\\"dimensions\\"] = {\\"framework\\":\\"TensorRT-LLM\\"}',
  '  data[\\"metrics\\"] = dict()',
  '  with open(csv_path, encoding=\\"utf-8\\") as csvf:',
  '      reader = csv.DictReader(csvf)',
  '      for rows in reader:',
  '          for key in rows:',
  '              try:',
  '                  float(rows[key])',
  '                  data[\\"metrics\\"][key] = float(rows[key])',
  '              except:',
  '                  data[\\"dimensions\\"][key] = rows[key]',
  '  with jsonlines.open(jsonl_path, \\"w\\") as writter:',
  '      writter.write(data)',
  'if __name__ == \\"__main__\\":',
  '  make_json(csv_metric_path, sys.argv[1])',
  )
  docker_container_name = 'tensorrt_llm-release-cloud-ml-auto-solutions'
  py_script = '\n'.join(jsonl_converter_py_lines)
  make_jsonl_convert_cmd = f"echo '{py_script}' > jsonl_converter.py"
  docker_cmds = ("cd benchmarks/python",
                 "pip install jsonlines",
                 "python benchmark.py -m llama_7b --mode plugin --batch_size 8 --input_output_len 128,128 --csv",
                 make_jsonl_convert_cmd,
                 f"python jsonl_converter.py {jsonl_output_path}",)
  docker_cmd = " && ".join(docker_cmds)
  run_model_cmds = (
    f"docker exec -i {docker_container_name} /bin/bash -c \"{docker_cmd}\"",
    f"docker cp {docker_container_name}:/app/tensorrt_llm/benchmarks/python/{jsonl_output_path} {jsonl_output_path}",
    f"cat {jsonl_output_path}",
    f"gsutil cp {jsonl_output_path} {metric_config.SshEnvVars.GCS_OUTPUT.value}",
 )

  job_test_config = test_config.GpuVmTest(
      test_config.Gpu(
          machine_type=machine_type.value,
          image_family=image_family.value,
          count=count,
          accelerator_type=accelerator_type.value,
          runtime_version=RUNTIME_IMAGE,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.YIJIA_J,
      # TODO: Do we need to create this ?
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/tensorrt_llm",
  )

  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=gpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )

  job_metric_config = metric_config.MetricConfig(
      json_lines=metric_config.JSONLinesConfig("metric_report.jsonl"),
      use_runtime_generated_gcs_folder=True,
  )

  return task.GpuCreateResourceTask (
    image_project.value,
    image_family.value,
    job_test_config,
    job_gcp_config,
    job_metric_config,
  )
