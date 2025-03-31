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

"""DAGs to run Mantaray benchmarks."""


import datetime
from airflow import models
from xlml.utils import mantaray
import yaml
from dags import composer_env
from dags.pytorch_xla.configs import pytorchxla_torchbench_config as config
import dags.common.vm_resource as resource
import re
import tempfile
from airflow.decorators import task
from airflow.decorators import task_group
from airflow.hooks.subprocess import SubprocessHook
from dags.common import test_owner
from xlml.utils import gpu, metric, name_format, ssh, tpu, xpk, gke
from airflow.models import Variable


# Skip running this script in unit test because gcs loading will fail.
if composer_env.is_prod_env() or composer_env.is_dev_env():
  # Download xlml_jobs.yaml from the borgcron GCS bucket, which
  # is pulled nightly from google3.
  xlml_jobs_yaml = mantaray.load_file_from_gcs(
      f"{mantaray.MANTARAY_G3_GS_BUCKET}/xlml_jobs/xlml_jobs.yaml"
  )
  xlml_jobs = yaml.safe_load(xlml_jobs_yaml)

  # Create a DAG for PyTorch/XLA tests
  pattern = r"^(ptxla|pytorchxla).*"
  workload_file_name_list = []
  for job in xlml_jobs:
    if re.match(pattern, job["task_name"]):
      workload_file_name_list.append(job["file_name"])


  HF_TOKEN = Variable.get("HF_TOKEN", None)

  def run_test_code_on_persistent_TPUVM():
      """
      Run nightly vLLM inference benchmarking on persistent TPU.
      """
      gcloud_command = (
          f"set -x && "
          "set -u && "
          "project=$(curl -sS \"http://metadata.google.internal/computeMetadata/v1/project/project-id\" -H \"Metadata-Flavor: Google\") && "
          "zone=europe-west4-a && "
          "tpu_name=manfei-2025-v6e-4-cloud-ml-auto-solu && "
          "[ -f /scripts/id_rsa ] && sudo rm /scripts/id_rsa && sudo rm /scripts/id_rsa.pub; sudo ssh-keygen -t rsa -f /scripts/id_rsa -q -N \"\" && "
          "echo \"xl-ml-test:$(cat /scripts/id_rsa.pub)\" > ssh-keys.txt && "
          "echo 'echo Running startup script' > startup-script.txt && "
          "sudo apt-get -y update && "
          "sudo apt-get -y install lsof && "
          "sudo dpkg --configure -a && "
          "sudo apt-get -y install nfs-common && "
          "yes '' | gcloud compute config-ssh && "
          "ls /home/airflow/.ssh/ && "
          "echo ${project} && "
          "echo ${zone} && "
          "echo ${tpu_name} && "
          "yes 'y' | sudo gcloud alpha compute tpus tpu-vm ssh manfei-2025-v6e-4-cloud-ml-auto-solu --zone=europe-west4-a "
          "--project=cloud-ml-auto-solutions --ssh-key-file=/home/airflow/.ssh/google_compute_engine --strict-host-key-checking=no "
          "--internal-ip --worker=all --command ' \
              sudo docker ps -a --filter \"name=testooo\" -q | grep -q . && sudo docker rm -f testooo; \
              sudo docker run --privileged --net host --shm-size=16G --name testooo \
              docker.io/vllm/vllm-tpu:270a5da495d24e947a71e2fa0c56635f4fad2dc3 bash -c \" \
                  export HF_TOKEN={HF_TOKEN} && \
                  VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --disable-log-requests \
                  --max-num-seq=320 --gpu-memory-utilization=0.95 --tensor-parallel-size=4 --max-model-len=8192 --port 8009 & sleep 900 && \
                  git clone -b inference-benchmark-script https://github.com/ManfeiBai/vllm.git vllmscript && \
                  bash vllmscript/benchmarks/inference_benchmark_script.sh \
              \" && sudo docker stop testooo && sudo docker rm testooo' "
      )
      return gcloud_command


  @task
  def run_on_v6e_4_persistant_TPUVM():
    with tempfile.TemporaryDirectory() as tmpdir:
      hook = SubprocessHook()
  
      result = hook.run_command(
          [
              "bash",
              "-c",
              run_test_code_on_persistent_TPUVM(),
          ],
          cwd=tmpdir,
      )
      assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


  @task_group(prefix_group_id=False)
  def run_vllm_nightly_on_v6e_4_persistant_TPUVM():
      GCS_SUBFOLDER_PREFIX_PYTORCH_XLA = test_owner.Team.PYTORCH_XLA.value
      output_location = name_format.generate_gcs_folder_location(
              f"{GCS_SUBFOLDER_PREFIX_PYTORCH_XLA}/vllm_benchmark_nightly",
              f'vllm-nightly-v6e-4',
          )
      run_on_v6e_4_persistant_TPUVM_func = run_on_v6e_4_persistant_TPUVM()
      run_on_v6e_4_persistant_TPUVM_func


  # merge all PyTorch/XLA tests ino one Dag
  with models.DAG(
      dag_id="pytorch_xla_model_regression_test_on_trillium",
      schedule="0 0 * * *",  # everyday at midnight # job["schedule"],
      tags=["mantaray", "pytorchxla", "xlml", "vllm"],
      start_date=datetime.datetime(2024, 4, 22),
      catchup=False,
  ) as dag:
    for workload_file_name in workload_file_name_list:
      run_workload = mantaray.run_workload.override(
          task_id=workload_file_name.split(".")[0]
      )(
          workload_file_name=workload_file_name,
      )
      run_workload
    run_vllm_nightly_on_v6e_4_persistant_TPUVM()

  # Create a DAG for each job from maxtext
  for job in xlml_jobs:
    if not re.match(pattern, job["task_name"]):
      with models.DAG(
          dag_id=job["task_name"],
          schedule=job["schedule"],
          tags=["mantaray"],
          start_date=datetime.datetime(2024, 4, 22),
          catchup=False,
      ) as dag:
        run_workload = mantaray.run_workload(
            workload_file_name=job["file_name"],
        )
    run_workload
else:
  print(
      "Skipping creating Mantaray DAGs since not running in Prod or Dev composer environment."
  )
