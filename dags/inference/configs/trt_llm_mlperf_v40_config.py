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

"""Utilities to construct configs for MLPerf4.0 Reproduce DAG."""

import datetime
from typing import Dict
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags.common import vm_resource
from dags.common.vm_resource import Project, RuntimeVersion

RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value


def get_trt_llm_mlperf_v40_gpu_config(
    machine_type: vm_resource.MachineVersion,
    image_project: vm_resource.ImageProject,
    image_family: vm_resource.ImageFamily,
    accelerator_type: vm_resource.GpuVersion,
    count: int,
    gpu_zone: vm_resource.Zone,
    time_out_in_min: int,
    test_name: str,
    project: Project,
    network: str,
    subnetwork: str,
    existing_instance_name: str = None,
    model_configs: Dict = {},
) -> task.GpuCreateResourceTask:
  docker_container_name = "mlperf-inference"
  set_up_cmds = (
      # Install Nvidia driver
      "wget -c https://us.download.nvidia.com/tesla/550.54.15/NVIDIA-Linux-x86_64-550.54.15.run",
      "chmod u+x NVIDIA-Linux-x86_64-550.54.15.run",
      "sudo ./NVIDIA-Linux-x86_64-550.54.15.run -x-module-path=/usr/lib/xorg/modules --ui=none -x-library-path=/usr/lib -q",
      "sudo nvidia-smi -pm 1",
      # Format and mount multiple Local SSD
      "sudo apt update && sudo apt install mdadm --no-install-recommends",
      "find /dev/ | grep google-local-nvme-ssd",
      "sudo mdadm --create /dev/md0 --level=0 --raid-devices=$(find /dev/ -name 'google-local-nvme-ssd*' | wc -l) $(find /dev/ -name 'google-local-nvme-ssd*')",
      "sudo mdadm --detail --prefer=by-id /dev/md0",
      "sudo mkfs.ext4 -F /dev/md0",
      "sudo mkdir -p /scratch",
      "sudo mount /dev/md0 /scratch",
      "sudo chmod a+w /scratch",
      "cd /scratch",
      # Prepare data
      "gcloud storage cp --no-clobber --recursive gs://tohaowu/mlpinf-v40/mlperf_inf_dlrmv2 .",
      "gcloud storage cp --no-clobber --recursive gs://tohaowu/mlpinf-v40/models .",
      "gcloud storage cp --no-clobber --recursive gs://tohaowu/mlpinf-v40/preprocessed_data .",
      "mv models/Llama2/fp8-quantized-ammo/llama2-70b-chat-hf-tp2pp1-fp8/ models/Llama2/fp8-quantized-ammo/llama2-70b-tp2pp1-fp8/",
      "git clone https://github.com/mlcommons/inference_results_v4.0",
      "cd /scratch/inference_results_v4.0/closed/Google",
      "export MLPERF_SCRATCH_PATH=/scratch",
      "cp /scratch/inference_results_v4.0/closed/{NVIDIA,Google}/Makefile.docker",
      "sed -i '27i\ARCH=x86_64' Makefile",
      "sed -i '29i\ARCH=x86_64' Makefile.docker",
      "sudo usermod -a -G docker $USER",
      # Build and launch a docker container
      "make prebuild DOCKER_DETACH=1",
      "make docker_add_user",
      f"make launch_docker DOCKER_NAME={docker_container_name} DOCKER_ARGS='-v /scratch/mlperf_inf_dlrmv2:/home/mlperf_inf_dlrmv2 -d'",
  )

  jsonl_output_path = "metric_report.jsonl"
  jsonl_converter_py_lines = (
      "import sys, json, glob, jsonlines",
      "metadata_log_pattern = '/scratch/inference_results_v4.0/closed/Google/build/logs/*/*/*/*/metadata.json'",
      "metadata_log_paths = glob.glob(metadata_log_pattern)",
      "def convert_to_jsonl(json_path, jsonl_path):",
      "  data = dict()",
      "  data['dimensions'] = dict()",
      "  data['metrics'] = dict()",
      "  with open(json_path, 'r') as file:",
      "      metadatadata = json.load(file)",
      "      for key in metadatadata:",
      "          try:",
      "              float(metadatadata[key])",
      "              data['metrics'][key] = float(metadatadata[key])",
      "          except:",
      "              data['dimensions'][key] = metadatadata[key]",
      "  with jsonlines.open(jsonl_path, 'a') as writer:",
      "      writer.write(data)",
      "if __name__ == '__main__':",
      "  for metadata_log_path in metadata_log_paths:",
      "    convert_to_jsonl(metadata_log_path, sys.argv[1])",
  )
  py_script = "\n".join(jsonl_converter_py_lines)
  make_jsonl_converter_cmd = f'echo "{py_script}" > jsonl_converter.py'

  docker_cmds = (
      # "make link_dirs",
      # "make build BUILD_TRTLLM=1",
      # "pip install huggingface_hub==0.24.7",
      f'make run RUN_ARGS="--benchmarks={model_configs["model_name"]} --scenarios={model_configs["scenario"]} --config_ver={model_configs["config_ver"]} --test_mode=PerformanceOnly"',
  )
  docker_cmd = " && ".join(docker_cmds)
  run_model_cmds = (
      "pip install jsonlines",
      f"docker restart {docker_container_name}",
      f'docker exec -i {docker_container_name} /bin/bash -c "{docker_cmd}"',
      make_jsonl_converter_cmd,
      "cat jsonl_converter.py",
      f"python3 jsonl_converter.py {jsonl_output_path}",
      f"cat {jsonl_output_path}",
      f"gcloud storage cp {jsonl_output_path} {metric_config.SshEnvVars.GCS_OUTPUT.value}",
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
      task_owner=test_owner.YIJIA_J,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/trt_llm_mlperf_v40",
      use_existing_instance=existing_instance_name is not None,
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

  return task.GpuCreateResourceTask(
      image_project.value,
      image_family.value,
      job_test_config,
      job_gcp_config,
      job_metric_config,
      existing_instance_name=existing_instance_name,
  )
