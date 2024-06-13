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

from typing import Dict
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner, vm_resource
from dags.vm_resource import Project, RuntimeVersion

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value


def get_trt_llm_mlperf_v4_0_gpu_config(
    machine_type: vm_resource.MachineVersion,
    image_project: vm_resource.ImageProject,
    image_family: vm_resource.ImageFamily,
    accelerator_type: vm_resource.GpuVersion,
    count: int,
    gpu_zone: vm_resource.Zone,
    time_out_in_min: int,
    test_name: str,
    model_configs: Dict = dict(model_name="llama2-70b"),
) -> task.GpuCreateResourceTask:

  docker_container_name = "mlperf-inference"
  set_up_cmds = (
      "sudo mkdir -p /scratch",
      "cd /scratch",
      # Prepare data
      "gsutil -m cp -r gs://tohaowu/mlpinf-v40/data .",
      "gsutil -m cp -r gs://tohaowu/mlpinf-v40/models .",
      "gsutil -m cp -r gs://tohaowu/mlpinf-v40/preprocessed_data .",
      "git clone https://github.com/mlcommons/inference_results_v4.0",
      "export MLPERF_SCRATCH_PATH=/scratch",
      "cd /scratch/inference_results_v4.0/closed/Google",
      "cp /scratch/inference_results_v4.0/closed/{NVIDIA,Google}/Makefile.docker",
      "sudo usermod -a -G docker $USER",
      # Build and launch a docker container
      "make prebuild DOCKER_DETACH=1",
      "make docker_add_user",
      f"make launch_docker DOCKER_NAME={docker_container_name} DOCKER_ARGS='-d'",
  )

  jsonl_output_path = "metric_report.jsonl"
  jsonl_converter_py_lines = (
      "import sys, json, glob, jsonlines",
      "metadata_log_pattern = './build/logs/*/*/*/*/metadata.json'",
      "metadata_log_path = glob.glob(metadata_log_pattern)[0]",
      "def convert_to_jsonl(json_path, jsonl_path):",
      "  data = dict()",
      "  data['dimensions'] = {'framework':'TensorRT-LLM'}",
      "  data['metrics'] = dict()",
      "  with open(json_path, 'r') as file:",
      "      metadatadata = json.load(file)",
      "      for key in metadatadata:",
      "          try:",
      "              float(metadatadata[key])",
      "              data['metrics'][key] = float(metadatadata[key])",
      "          except:",
      "              data['dimensions'][key] = metadatadata[key]",
      "  with jsonlines.open(jsonl_path, 'w') as writter:",
      "      writter.write(data)",
      "if __name__ == '__main__':",
      "  convert_to_jsonl(metadata_log_path, sys.argv[1]))",
  )
  py_script = "\n".join(jsonl_converter_py_lines)
  make_jsonl_converter_cmd = f'echo "{py_script}" > jsonl_converter.py'

  docker_cmds = (
      "make build BUILD_TRTLLM=1",
      f"make run RUN_ARGS=\"--benchmarks={model_configs['model_mode']} --scenarios=Offline\"",
  )
  docker_cmd = " && ".join(docker_cmds)
  run_model_cmds = (
      f'docker exec -i {docker_container_name} /bin/bash -c "{docker_cmd}"',
      make_jsonl_converter_cmd,
      f"python jsonl_converter.py {jsonl_output_path}",
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
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/trt_llm_mlperf_4_0",
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

  return task.GpuCreateResourceTask(
      image_project.value,
      image_family.value,
      job_test_config,
      job_gcp_config,
      job_metric_config,
  )
