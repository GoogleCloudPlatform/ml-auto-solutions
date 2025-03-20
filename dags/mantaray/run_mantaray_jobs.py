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


  def run_test_code_on_persistent_TPUVM():
    gcloud_command = (
        f"gcloud compute tpus tpu-vm ssh manfei-2025-v6e-4 --zone=us-east5-b --project=cloud-ml-benchmarking --ssh-flag='-t' --worker=all \
        --command=\"sudo docker run -it --privileged --net host --shm-size=16G --name testooo docker.io/vllm/vllm-tpu:270a5da495d24e947a71e2fa0c56635f4fad2dc3 \
        bash -c 'export HF_TOKEN=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 && \
  VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --disable-log-requests \
  --max-num-seq=320 --gpu-memory-utilization=0.95 --tensor-parallel-size=4 --max-model-len=8192 --port 8009 & sleep 1200 && \
  wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json && \
  pip install --upgrade google-cloud-storage && rm -rf inference-benchmark && git clone https://github.com/AI-Hypercomputer/inference-benchmark && \
  echo \"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main\" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
  apt-get update && apt-get install -y google-cloud-sdk && apt-get -y install jq && export HF_TOKEN=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 && \
  export PJRT_DEVICE=TPU && \
  python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer=meta-llama/Meta-Llama-3-8B --request-rate=1 --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 \
  --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B \"--output-bucket=gs://manfeipublic\"' && sudo docker stop testooo && sudo docker rm testooo\" \
  ",
    )
    return gcloud_command


  def make_sure_docker_container_cleaned_on_persistent_TPUVM():
    gcloud_command = (
        f"gcloud compute tpus tpu-vm ssh manfei-2025-v6e-4 --zone=us-east5-b --project=cloud-ml-benchmarking --ssh-flag='-t -4 -L 6009:localhost:6009' --worker=all --command=\"sudo docker stop testooo && sudo docker rm testooo\"",
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
              ";".join(
                  run_test_code_on_persistent_TPUVM()
                  + make_sure_docker_container_cleaned_on_persistent_TPUVM()
              ),
          ],
          cwd=tmpdir,
      )
      assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


  # merge all PyTorch/XLA tests ino one Dag
  with models.DAG(
      dag_id="pytorch_xla_model_regression_test_on_trillium",
      schedule="0 0 * * *",  # everyday at midnight # job["schedule"],
      tags=["mantaray", "pytorchxla", "xlml"],
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
    run_on_v6e_4_persistant_TPUVM()

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
