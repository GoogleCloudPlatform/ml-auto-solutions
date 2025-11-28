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
from typing import Dict, List
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags.common import vm_resource
from dags.common.vm_resource import GpuVersion, Project, RuntimeVersion

RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value


def get_trt_llm_mlperf_gpu_config(
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
    benchmark_configs: Dict = {},
    model_parameters: Dict = {},
    parameter_positions: Dict = {},
    binary_search_steps: int = 1,
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
      'sudo mdadm --create /dev/md0 --level=0 --raid-devices=$(find /dev/ -name "google-local-nvme-ssd*" | wc -l) $(find /dev/ -name "google-local-nvme-ssd*")',
      "sudo mdadm --detail --prefer=by-id /dev/md0",
      "sudo mkfs.ext4 -F /dev/md0",
      "sudo mkdir -p /scratch",
      "sudo mount /dev/md0 /scratch",
      "sudo chmod a+w /scratch",
      "cd /scratch",
      # Prepare data
      "gcloud storage cp --no-clobber --recursive gs://yijiaj/mlperf/v41/Google_GPU .",
      "gcloud storage cp --no-clobber --recursive gs://tohaowu/mlpinf-v40/mlperf_inf_dlrmv2 .",
      f"gcloud storage cp --no-clobber --recursive {benchmark_configs['models']} .",
      f"gcloud storage cp --no-clobber --recursive {benchmark_configs['preprocessed_data']} .",
      f"gcloud storage cp --no-clobber --recursive {benchmark_configs['docker_config']} .",
      "curl -sSL https://get.docker.com/ | sh",
      "sudo mkdir -p /home/cloud-ml-auto-solutions/.docker",
      "sudo touch ~/.docker/config.json",
      "sudo cp config.json ~/.docker/config.json",
      "sudo chown cloud-ml-auto-solutions:cloud-ml-auto-solutions /home/cloud-ml-auto-solutions",
      "sudo chmod a+w /home/cloud-ml-auto-solutions/.docker",
      "cd Google_GPU",
      "export MLPERF_SCRATCH_PATH=/scratch",
      "sed -i '27i\ARCH=x86_64' Makefile",
      "sed -i '29i\ARCH=x86_64' Makefile.docker",
      "sed -i '29i\ARCH=x86_64' Makefile.const",
      "sudo usermod -a -G docker $USER",
      # Build and launch a docker container
      "PARTNER_DROP=1 make prebuild DOCKER_DETACH=1",
      "make docker_add_user",
      f"make launch_docker DOCKER_NAME={docker_container_name} DOCKER_ARGS='-v /scratch/mlperf_inf_dlrmv2:/home/mlperf_inf_dlrmv2 -d'",
  )

  jsonl_output_path = "metric_report.jsonl"
  jsonl_converter_py_lines = (
      "import sys, json, glob, jsonlines",
      "metadata_log_pattern = '/scratch/Google_GPU/build/logs/*/*/*/*/metadata.json'",
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

  model_parameters_sweep_cmds = []
  for model_name in benchmark_configs["model_name"].split(","):
    scenario = ",".join(model_parameters[model_name])
    if accelerator_type == GpuVersion.L4:
      model_parameters_sweep_cmds.append(
          f"CUDA_VISIBLE_DEVICES=0 make generate_engines RUN_ARGS='--benchmarks={model_name} --scenarios={scenario}'"
      )
    else:
      model_parameters_sweep_cmds.append(
          f"make generate_engines RUN_ARGS='--benchmarks={model_name} --scenarios={scenario}'"
      )

  for model_name in benchmark_configs["model_name"].split(","):
    for scenario in model_parameters[model_name]:
      for parameter in model_parameters[model_name][scenario]:
        steps = 2 ** (binary_search_steps - 1) + 1
        step_interval = round(
            (
                model_parameters[model_name][scenario][parameter][1]
                - model_parameters[model_name][scenario][parameter][0]
            )
            / (steps - 1),
            2,
        )
        parameter_current_value = model_parameters[model_name][scenario][
            parameter
        ][0]
        while steps > 0:
          if accelerator_type == GpuVersion.L4:
            model_parameters_sweep_cmds.append(
                f"CUDA_VISIBLE_DEVICES=0 make run_harness RUN_ARGS='--benchmarks={model_name} --scenarios={scenario}'"
            )
          else:
            model_parameters_sweep_cmds.append(
                f"make run_harness RUN_ARGS='--benchmarks={model_name} --scenarios={scenario}'"
            )
          current_value_str = str(parameter_current_value)
          parameter_current_value = parameter_current_value + step_interval
          next_value_str = str(parameter_current_value)
          model_parameters_sweep_cmds.append(
              f"sed -i '{parameter_positions[model_name][scenario][parameter]}s/{current_value_str}/{next_value_str}/' configs/{model_name}/{scenario}/__init__.py"
          )
          steps = steps - 1

  docker_cmds = [
      "make link_dirs",
      "make build BUILD_TRTLLM=1",
      "pip install huggingface_hub==0.24.7",
      "lscpu",
  ]
  if accelerator_type == GpuVersion.L4:
    docker_cmds.append(
        "sed -i '310s/16/24/' code/common/systems/known_hardware.py"
    )
  docker_cmds.extend(model_parameters_sweep_cmds)
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
          attach_local_ssd=True
          if accelerator_type != GpuVersion.H100
          else False,
          disk_size_gb=1000,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.YIJIA_J,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/trt_llm_mlperf_v41",
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
