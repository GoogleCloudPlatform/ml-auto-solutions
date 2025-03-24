from xlml.utils import gpu, metric, name_format, ssh, tpu
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


def vllm_nightly_cmds() -> str:
  return """
gcloud compute tpus tpu-vm ssh manfei-2025-v6e-4 --zone=us-east5-b   --project=cloud-ml-benchmarking   --ssh-flag='-t'   --worker=all   --command="sudo docker run -it --privileged --net host --shm-size=16G --name testooo docker.io/vllm/vllm-tpu:270a5da495d24e947a71e2fa0c56635f4fad2dc3 bash -c 'export HF_TOKEN=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 && \
VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --disable-log-requests --max-num-seq=320 --gpu-memory-utilization=0.95 --tensor-parallel-size=4 --max-model-len=8192 --port 8009 & sleep 1200 && \
wget --no-verbose https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json && \
pip install --upgrade google-cloud-storage && rm -rf inference-benchmark && git clone https://github.com/AI-Hypercomputer/inference-benchmark && \
echo \"deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main\" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && apt-get update && \
apt-get install -y google-cloud-sdk && apt-get -y install jq && export HF_TOKEN=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 && export PJRT_DEVICE=TPU && \
python inference-benchmark/benchmark_serving.py --save-json-results --port=8009 --dataset=ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer=meta-llama/Meta-Llama-3-8B --request-rate=1 --backend=vllm --num-prompts=300 --max-input-length=1024 --max-output-length=1024 --file-prefix=benchmark --models=meta-llama/Meta-Llama-3-8B \"--output-bucket=gs://manfeipublic\"' && \
sudo docker stop testooo && sudo docker rm testooo"
"""


def vllm_nightly_cmds_run_task(
    queued_resource: airflow.XComArg, ssh_keys: airflow.XComArg
):
  cmds = vllm_nightly_cmds()
  ssh_keys = ssh.generate_ssh_keys()

  return tpu.ssh_persistant_tpu.override(
      task_id='vllm_nightly_cmds_run_task',
      execution_timeout=1200,
  )(
      cmds,
      ssh_keys,
      False,
  )


with models.DAG(
    dag_id="pytorchxla-vllm-nightly-2",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "nightly", "torchbench"],
    start_date=datetime.datetime(2025, 1, 1),
    catchup=False,
) as dag:
  vllm_nightly_cmds_run_task()

