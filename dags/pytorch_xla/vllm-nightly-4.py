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

"""A DAG to run all PyTorch/XLA tests with nightly version."""

import datetime
import tempfile
from airflow import models
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from xlml.utils import mantaray
import yaml
from dags import composer_env
from dags.pytorch_xla.configs import pytorchxla_torchbench_config as config
import dags.common.vm_resource as resource
import re


# Schudule the job to run everyday at 3:00AM PST (11:00AM UTC).
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None


# delete "--ssh-flag='-t' " due to Airflow error:
# [2025-03-20 21:38:53.600859+00:00] {subprocess.py:93} INFO - Pseudo-terminal will not be allocated because stdin is not a terminal.
# [2025-03-20 21:41:03.733700+00:00] {subprocess.py:93} INFO - ssh: connect to host 34.162.99.201 port 22: Connection timed out
# [2025-03-20 21:41:03.734472+00:00] {subprocess.py:93} INFO - ERROR: (gcloud.compute.tpus.tpu-vm.ssh) [/usr/bin/ssh] exited with return code [255].
# [2025-03-20 21:41:04.093228+00:00] {subprocess.py:97} INFO - Command exited with return code 255
def run_test_code_on_persistent_TPUVM():
    """
    Generuje polecenie gcloud do uruchomienia testowego kodu na trwałej maszynie wirtualnej TPU.
    """
    print("point 7: after enter run_test_code_on_persistent_TPUVM, before code run !!!")
    gcloud_command = (
        f"project=$(curl -sS \"http://metadata.google.internal/computeMetadata/v1/project/project-id\" -H \"Metadata-Flavor: Google\") && "
        "zone=$(curl -sS \"http://metadata.google.internal/computeMetadata/v1/instance/zone\" -H \"Metadata-Flavor: Google\" | awk -F'/' '{print $4}') && "
        "tpu_name=manfei-2025-v6e-4 && "
        "ssh-keygen -t rsa -f /scripts/id_rsa -q -N \"\" && "
        "echo \"xl-ml-test:$(cat /scripts/id_rsa.pub)\" > ssh-keys.txt && "
        "echo %(startupScript)s > startup-script.txt && "
        "gcloud compute tpus tpu-vm ssh manfei-2025-v6e-4 "
        "--zone=us-east5-b "
        "--project=cloud-ml-benchmarking "
        "--worker=all "
        "--command=\"sudo docker run -it --privileged --net host --shm-size=16G --name testooo "
        "docker.io/vllm/vllm-tpu:270a5da495d24e947a71e2fa0c56635f4fad2dc3 bash -c '"
        "export HF_TOKEN=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 && "
        "VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server "
        "--model meta-llama/Meta-Llama-3-8B --disable-log-requests "
        "--max-num-seq=320 --gpu-memory-utilization=0.95 --tensor-parallel-size=4 "
        "--max-model-len=8192 --port 8009 & sleep 1200 && "
        "wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json && "
        "pip install --upgrade google-cloud-storage && rm -rf inference-benchmark && "
        "git clone https://github.com/AI-Hypercomputer/inference-benchmark && "
        "echo \\\"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main\\\" > /etc/apt/sources.list.d/google-cloud-sdk.list && "
        "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && "
        "apt-get update && apt-get install -y google-cloud-sdk && apt-get -y install jq && "
        "export HF_TOKEN=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 && export PJRT_DEVICE=TPU && "
        "python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 "
        "--dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B "
        "--request-rate=1 --backend=vllm --num-prompts=300 --max-input-length=1024 "
        "--max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B "
        "\\\"--output-bucket=gs://manfeipublic\\\"' && docker stop testooo && docker rm testooo\"" # Usunięto sudo
    )
    print("point 8: after enter run_test_code_on_persistent_TPUVM, after gcloud_command = (... !!!")
    return gcloud_command


def make_sure_docker_container_cleaned_on_persistent_TPUVM():
  print("point 9: after enter make_sure_docker_container_cleaned_on_persistent_TPUVM, before code run!!!")
  gcloud_command = (
      f"gcloud compute tpus tpu-vm ssh manfei-2025-v6e-4 --zone=us-east5-b --project=cloud-ml-benchmarking --ssh-flag='-t -4 -L 6009:localhost:6009' --worker=all --command=\"sudo docker stop testooo && sudo docker rm testooo\"",
  )
  print("point 9: after enter make_sure_docker_container_cleaned_on_persistent_TPUVM, after gcloud_command = (... !!!")
  return gcloud_command


@task
def run_on_v6e_4_persistant_TPUVM():
  print("point 2: enter run_on_v6e_4_persistant_TPUVM(), and before code run !!!")
  with tempfile.TemporaryDirectory() as tmpdir:
    print("point 3: after with tempfile.TemporaryDirectory() as tmpdir: !!!")
    hook = SubprocessHook()
    print("point 4: after hook = SubprocessHook() !!!")

    result = hook.run_command(
        [
            "bash",
            "-c",
            run_test_code_on_persistent_TPUVM(),
        ],
        cwd=tmpdir,
    )
    print("point 5: after result = hook.run_command(...) !!!")
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"
    print("point 6: after assert result.exit_code... !!!")


@task
def clean_docker_container_on_v6e_4_persistant_TPUVM():
  print("point 2: enter run_on_v6e_4_persistant_TPUVM(), and before code run !!!")
  with tempfile.TemporaryDirectory() as tmpdir:
    print("point 3: after with tempfile.TemporaryDirectory() as tmpdir: !!!")
    hook = SubprocessHook()
    print("point 4: after hook = SubprocessHook() !!!")

    result = hook.run_command(
        [
            "bash",
            "-c",
            make_sure_docker_container_cleaned_on_persistent_TPUVM(),
        ],
        cwd=tmpdir,
    )
    print("point 5: after result = hook.run_command(...) !!!")
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"
    print("point 6: after assert result.exit_code... !!!")

with models.DAG(
    dag_id="pytorchxla-vllm-nightly",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "nightly", "torchbench"],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
    # follow example in https://github.com/GoogleCloudPlatform/ml-auto-solutions/blob/bda4d59ed7fd9dd3b244a8b2612385c4f5c9a8a9/dags/multipod/maxtext_gpu_end_to_end.py#L41
    print("point 1: before the total function running!!!")
    run_on_v6e_4_persistant_TPUVM()
    clean_docker_container_on_v6e_4_persistant_TPUVM()
    print("point final: after the total function running!!!")









              # project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google")
              # zone=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F'/' '{print $4}')
              # tpu_name=tpu-${POD_UID}
              # ssh-keygen -t rsa -f /scripts/id_rsa -q -N ""

              # echo "
              # gcloud alpha compute tpus tpu-vm delete -q --async ${tpu_name} --zone=${zone}
              # sleep 60
              # " > /scripts/cleanup.sh

              # echo "xl-ml-test:$(cat /scripts/id_rsa.pub)" > ssh-keys.txt
              # echo %(startupScript)s > startup-script.txt

              # # Retry every 30 seconds for up to 10 minutes
              # start_time="$(date -u +%%s)"
              # for i in {1..40}; do
              #   set +e
              #   gcloud alpha compute tpus tpu-vm create ${tpu_name} \
              #     --accelerator-type=%(acceleratorName)s \
              #     --version=%(softwareVersion)s  \
              #     --metadata-from-file='ssh-keys=ssh-keys.txt,startup-script=startup-script.txt' \
              #     --labels='test-name=%(testName)s' \
              #     --zone=${zone}

              #   exit_code=$?
              #   set -e

              #   current_time="$(date -u +%%s)"
              #   elapsed_seconds=$(($current_time-$start_time))
              #   # Break if command passed or 10-minute limit reached
              #   test $exit_code = 0 && break
              #   test $elapsed_seconds -gt 600 && break
              #   sleep 30
              # done

              # if [ $exit_code -ne 0 ]; then
              #   exit $exit_code
              # fi

              # echo ${zone} > /scripts/zone
              # echo ${tpu_name} > /scripts/tpu_name
              # gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --format="value(networkEndpoints[0].ipAddress)" > /scripts/tpu_ip
              # gcloud compute tpus describe ${tpu_name} --project=${project} --zone=${zone} --flatten="networkEndpoints[]" --format="csv[no-heading](networkEndpoints.ipAddress)" > /scripts/all_tpu_ips

              # sleep %(sleepTime)d
