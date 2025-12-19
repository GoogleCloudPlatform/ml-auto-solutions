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

import datetime
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags.common import vm_resource
from dags.common.vm_resource import Project, RuntimeVersion

RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
GCS_SUBFOLDER_PREFIX = test_owner.Team.INFERENCE.value


def get_trt_llm_gpu_config(
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
) -> task.GpuCreateResourceTask:
  set_up_cmds = (
      "pip install --upgrade pip",
      # Install Nvidia driver.
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
      # Install TensorRT-LLM.
      "gcloud storage cp --recursive gs://tohaowu/llama_3_8B_Instruct_HF_model .",
      "gcloud storage cp --recursive gs://tohaowu/llama_3.1_70B_Instruct_HF_model .",
      "gcloud storage cp --recursive gs://tohaowu/Mixtral-8x22B-Instruct-v0.1 .",
      "gcloud storage cp --recursive gs://yijiaj/gemma/gemma-2-27b-it .",
      "sudo apt-get update",
      "sudo apt-get -y install git git-lfs",
      "git clone https://github.com/NVIDIA/TensorRT-LLM.git",
      "cd TensorRT-LLM",
      "git submodule update --init --recursive",
      "git lfs install",
      "git lfs pull",
      "make -C docker release_build",
      "make -C docker release_run DOCKER_RUN_ARGS='--detach -v /scratch:/scratch' RUN_CMD='sleep infinity'",
  )

  jsonl_output_path = "metric_report.jsonl"
  jsonl_converter_py_lines = (
      "import sys, csv, jsonlines, glob",
      'csv_files = glob.glob(\\"*.csv\\")',
      "def make_json(csv_path, jsonl_path):",
      "  data = dict()",
      '  data[\\"dimensions\\"] = {\\"framework\\":\\"TensorRT-LLM\\"}',
      '  data[\\"metrics\\"] = dict()',
      '  with open(csv_path, encoding=\\"utf-8\\") as csvf:',
      "      reader = csv.DictReader(csvf)",
      "      for rows in reader:",
      "          for key in rows:",
      "              try:",
      "                  float(rows[key])",
      '                  data[\\"metrics\\"][key] = float(rows[key])',
      "              except:",
      '                  data[\\"dimensions\\"][key] = rows[key]',
      '  with jsonlines.open(jsonl_path, \\"a\\") as writter:',
      "      writter.write(data)",
      'if __name__ == \\"__main__\\":',
      "  for csv_metric_path in csv_files:",
      "    make_json(csv_metric_path, sys.argv[1])",
  )
  docker_container_name = "tensorrt_llm-release-cloud-ml-auto-solutions"
  py_script = "\n".join(jsonl_converter_py_lines)
  make_jsonl_convert_cmd = f"echo '{py_script}' > jsonl_converter.py"
  docker_cmds = (
      "pip install jsonlines",
      "cd examples/llama",
      "python convert_checkpoint.py --model_dir /scratch/llama_3_8B_Instruct_HF_model/ --output_dir /scratch/tllm_checkpoint_1gpu_tp1 --dtype float16 --tp_size 1",
      "trtllm-build --checkpoint_dir /scratch/tllm_checkpoint_1gpu_tp1 --output_dir /scratch/llama/8B/trt_engines/fp16/1-gpu/ --gemm_plugin auto",
      "python convert_checkpoint.py --model_dir /scratch/llama_3.1_70B_Instruct_HF_model/ --output_dir /scratch/tllm_checkpoint_8gpu_tp8 --dtype float16 --tp_size 8",
      "trtllm-build --checkpoint_dir /scratch/tllm_checkpoint_8gpu_tp8 --output_dir /scratch/llama/70B/trt_engines/fp16/8-gpu/ --gemm_plugin auto",
      "python ../llama/convert_checkpoint.py --model_dir /scratch/Mixtral-8x22B-Instruct-v0.1 --output_dir /scratch/tllm_checkpoint_mixtral_8gpu --dtype float16 --tp_size 8 --moe_tp_size 2 --moe_ep_size 4",
      "trtllm-build --checkpoint_dir /scratch/tllm_checkpoint_mixtral_8gpu --output_dir /scratch/trt_engines/mixtral/tp2ep4",
      "cd ../gemma",
      "python3 convert_checkpoint.py --ckpt-type hf --model-dir /scratch/gemma-2-27b-it/ --dtype bfloat16 --world-size 1 --output-model-dir /scratch/checkpoints/tmp_27b_it_tensorrt_llm/bf16/tp1/",
      "trtllm-build --checkpoint_dir /scratch/checkpoints/tmp_27b_it_tensorrt_llm/bf16/tp1/ --gemm_plugin auto --max_batch_size 8 --max_input_len 3000 --max_seq_len 3100 --output_dir /scratch/gemma2/27b/bf16/1-gpu/",
      "cd ../../benchmarks/python",
      "python benchmark.py -m dec --engine_dir /scratch/llama/8B/trt_engines/fp16/1-gpu/ --csv",
      "OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -n 8 python benchmark.py -m dec --engine_dir /scratch/llama/70B/trt_engines/fp16/8-gpu/ --csv",
      "OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -n 8 python benchmark.py -m dec --engine_dir /scratch/trt_engines/mixtral/tp2ep4 --csv",
      "python benchmark.py -m dec --engine_dir /scratch/gemma2/27b/bf16/1-gpu/ --dtype bfloat16 --csv",
      make_jsonl_convert_cmd,
      f"python jsonl_converter.py {jsonl_output_path}",
  )
  docker_cmd = " && ".join(docker_cmds)
  run_model_cmds = (
      f'docker exec -i {docker_container_name} /bin/bash -c "{docker_cmd}"',
      f"docker cp {docker_container_name}:/app/tensorrt_llm/benchmarks/python/{jsonl_output_path} {jsonl_output_path}",
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
          disk_size_gb=1000,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.YIJIA_J,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/trt_llm",
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
