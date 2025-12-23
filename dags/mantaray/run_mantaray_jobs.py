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
import re
import tempfile
import yaml
import os
import dags.common.vm_resource as resource
from airflow import models
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.decorators import task as task_decorators
from airflow.decorators import task_group
from airflow.hooks.subprocess import SubprocessHook
from dags import composer_env
from dags.pytorch_xla.configs import pytorchxla_torchbench_config as config
from dags.common import test_owner
from dags.common.vm_resource import AcceleratorType, GpuVersion, TpuVersion, Region, Zone, Project, RuntimeVersion, V6E_GCE_NETWORK, V6E_GCE_SUBNETWORK
from xlml.utils import gpu, metric, name_format, ssh, tpu, xpk, gke, bigquery, composer, mantaray
from xlml.apis import gcp_config, metric_config, task, test_config
from typing import Dict, Iterable, List, Optional


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

  # generate metadata for test result to push to BigQuery Database
  def new_add_test_config_metadata(
      base_id: str,
      metadata: List[List[bigquery.MetricHistoryRow]],
  ) -> List[List[bigquery.MetricHistoryRow]]:
    for index in range(len(metadata)):
      uuid = metric.generate_row_uuid(str(base_id), index)
      test_config_meta = []

      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="accelerator",
              metadata_value="v6e-4",
          )
      )
      test_config_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid,
              metadata_key="project",
              metadata_value=Project.CLOUD_ML_BENCHMARKING.value,
          )
      )
      if True:
        test_config_meta.append(
            bigquery.MetadataHistoryRow(
                job_uuid=uuid,
                metadata_key="num_slices",
                metadata_value=1,
            )
        )
        test_config_meta.append(
            bigquery.MetadataHistoryRow(
                job_uuid=uuid,
                metadata_key="multislice_topology",
                metadata_value=("1" "v6e-4"),
            )
        )

      metadata[index].extend(test_config_meta)

    return metadata

  # generate metadata for test result to push to BigQuery Database
  def new_add_airflow_metadata(
      base_id: str,
      project_name: str,
      metadata: List[List[bigquery.MetricHistoryRow]],
  ) -> List[List[bigquery.MetricHistoryRow]]:
    """Add airflow metadata: run_id, prev_start_date_success,
    and airflow_dag_run_link.

    Args:
      base_id: The base id to generate uuid.
      project_name: Project name.
      metadata: The data to append airflow metadata.

    Returns:
      The data with airflow metadata.
    """
    for index in range(len(metadata)):
      uuid = metric.generate_row_uuid(str(base_id), index)
      airflow_meta = []

      airflow_meta.append(
          bigquery.MetadataHistoryRow(
              job_uuid=uuid, metadata_key="base_id", metadata_value=base_id
          )
      )
      metadata[index].extend(airflow_meta)
    return metadata

  # post-process to push test result to BigQuery Database via self-generated metadata
  @task_decorators
  def new_process_metrics(
      base_id: str,
      task_metric_config: Optional[metric_config.MetricConfig],
      use_startup_script: bool = False,
      folder_location: Optional[str] = None,
      current_request_rate: int = 1,
  ) -> None:
    LOCAL_TPUVM_TPUVERSION_VALUE = TpuVersion.TRILLIUM.value
    benchmark_id = f"reqeust-rate-{current_request_rate}-vllm-benchmark-tpu-nightly-LLaMA3.1-8B-v{LOCAL_TPUVM_TPUVERSION_VALUE}-4"
    print("benchmark_id: ", benchmark_id)
    current_time = datetime.datetime.now()
    print("current_time: ", current_time)
    has_profile = False
    metric_history_rows_list = [[]]
    metadata_history_rows_list = [[]]
    profile_history_rows_list = []

    # process metrics, metadata, and profile
    if task_metric_config:
      print("task_metric_config: ", task_metric_config)
      if task_metric_config.json_lines:
        absolute_path = folder_location
        print("absolute_path: ", absolute_path)
        (
            metric_history_rows_list,
            metadata_history_rows_list,
        ) = metric.process_json_lines(base_id, absolute_path)

    print("metric_history_rows_list: ", metric_history_rows_list)
    print("metadata_history_rows_list: ", metadata_history_rows_list)

    # add default airflow metadata
    metadata_history_rows_list = new_add_airflow_metadata(
        base_id,
        Project.CLOUD_ML_AUTO_SOLUTIONS.value,
        metadata_history_rows_list,
    )
    print("metadata_history_rows_list: ", metadata_history_rows_list)

    metadata_history_rows_list = new_add_test_config_metadata(
        base_id,
        metadata_history_rows_list,
    )
    print("metadata_history_rows_list: ", metadata_history_rows_list)

    test_run_rows = []

    dataset_name = metric.update_dataset_name_if_needed(
        metric_config.DatasetOption.BENCHMARK_DATASET
    )
    print("dataset_name: ", dataset_name)

    bigquery_metric = bigquery.BigQueryMetricClient(
        Project.CLOUD_ML_AUTO_SOLUTIONS.value, dataset_name
    )
    print("after bigquery_metric")

    for index in range(len(metadata_history_rows_list)):
      u_uuid = metric.generate_row_uuid(str(base_id), index)
      print("check current u_uuid: ", u_uuid)
      job_history_row = bigquery.JobHistoryRow(
          uuid=u_uuid,
          timestamp=current_time,
          owner=test_owner.Team.PYTORCH_XLA.value,
          job_name=benchmark_id,
          job_status=bigquery.JobStatus.MISSED.value,
      )
      test_run_row = bigquery.TestRun(
          job_history_row,
          metric_history_rows_list[index],
          metadata_history_rows_list[index],
      )
      test_run_rows.append(test_run_row)

    print("Test run rows:", test_run_rows)
    bigquery_metric.insert(test_run_rows)

  HF_TOKEN_LLaMA3_8B = Variable.get("HF_TOKEN_LLaMA3_8B", None)

  # commands for vllm nightly benchmarking
  def run_test_code_on_persistent_TPUVM(
      output_location: str, current_request_rate: int
  ):
    """
    Run nightly vLLM inference benchmarking on persistent TPU.
    """
    print("output_location: ", output_location)
    print("current_request_rate: ", current_request_rate)
    gcloud_command = (
        f"set -x && "
        "set -u && "
        'project=$(curl -sS "http://metadata.google.internal/computeMetadata/v1/project/project-id" -H "Metadata-Flavor: Google") && '
        "zone=europe-west4-a && "
        "tpu_name=manfei-2025-v6e-4-cloud-ml-auto-solu && "
        '[ -f /scripts/id_rsa ] && sudo rm /scripts/id_rsa && sudo rm /scripts/id_rsa.pub; sudo ssh-keygen -t rsa -f /scripts/id_rsa -q -N "" && '
        'echo "xl-ml-test:$(cat /scripts/id_rsa.pub)" > ssh-keys.txt && '
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
        '--internal-ip --worker=all --command \' \
          sudo docker ps -a --filter "name=testooo" -q | grep -q . && sudo docker rm -f testooo && sudo docker image rmi us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm; \
          sudo docker run --privileged --net host --shm-size=16G --name testooo \
          us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm bash -c " \
            pip uninstall -y torch torchvision torch_xla jax jaxlib libtpu && \
            git clone https://github.com/vllm-project/vllm.git && cd vllm && \
            pip install -r requirements/tpu.txt && '
        f"VLLM_TARGET_DEVICE='tpu' python setup.py develop && \
            export PJRT_DEVICE=TPU && "
        f"export HF_TOKEN={HF_TOKEN_LLaMA3_8B} && \
            VLLM_USE_V1=1 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B --disable-log-requests \
            --max-num-seq=320 --gpu-memory-utilization=0.95 --tensor-parallel-size=4 --max-model-len=8192 --port 8009 & sleep 800 && \
            git clone -b inference-benchmark-script https://github.com/ManfeiBai/vllm.git vllmscript && "
        f"bash vllmscript/benchmarks/inference_benchmark_script.sh {current_request_rate} && "
        f"gcloud storage cp metric_result.jsonl {output_location} && ls \
          \" && sudo docker stop testooo && sudo docker rm testooo && sudo docker image rmi us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla:nightly_3.10_tpuvm' \
        "
    )
    return gcloud_command

  # execute commands for vllm nightly benchmarking
  @task_decorators
  def run_on_v6e_4_persistent_TPUVM(
      output_location: str, current_request_rate: int
  ):
    with tempfile.TemporaryDirectory() as tmpdir:
      print("output_location: ", output_location)
      hook = SubprocessHook()
      result = hook.run_command(
          [
              "bash",
              "-c",
              run_test_code_on_persistent_TPUVM(
                  output_location, current_request_rate
              ),
          ],
          cwd=tmpdir,
      )
      print("Command finished with code: {result.exit_code}")

  # generated GCS path for vllm nightly benchmark test result
  @task_decorators
  def get_path(current_request_rate: int):
    ### output_location path get to save metric_result.jsonl
    GCS_SUBFOLDER_PREFIX_PYTORCH_XLA = test_owner.Team.PYTORCH_XLA.value
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_location = f"gs://ml-auto-solutions/output/pytorch-xla/vllm_benchmark_nightly/vllm-nightly-v6e-8-{current_datetime}-{current_request_rate}/metric_report.jsonl"
    return output_location

  # call execute commands for vllm nightly benchmark test result, and push test result to BigQuery Database in post-process
  @task_group(prefix_group_id=False)
  def run_vllm_nightly_on_v6e_4_persistent_TPUVM(current_request_rate: int):
    ### output_location path get to save metric_result.jsonl
    output_location = get_path(current_request_rate)
    print("output_location: ", output_location)

    ### run test code
    run_on_v6e_4_persistent_TPUVM_func = run_on_v6e_4_persistent_TPUVM(
        output_location, current_request_rate
    )

    ### push test result to BigQuery Database
    ### task_metric_config: generate metric configs to push test result
    job_metric_config = metric_config.MetricConfig(
        json_lines="metric_report.jsonl",
        use_runtime_generated_gcs_folder=True,
    )

    process_id = metric.generate_process_id.override(retries=0)()
    print("process_id: ", process_id)
    post_process = new_process_metrics(
        str(process_id),
        job_metric_config,
        folder_location=output_location,
    )

    ### order of execute
    run_on_v6e_4_persistent_TPUVM_func >> post_process

  # split func defination with different func name to show on Airflow Dashboard
  @task_group(prefix_group_id=False)
  def vllm_request_rate_1_nightly_on_v6e_4_persistent_TPUVM():
    run_vllm_nightly_on_v6e_4_persistent_TPUVM(1)

  @task_group(prefix_group_id=False)
  def vllm_request_rate_4_nightly_on_v6e_4_persistent_TPUVM():
    run_vllm_nightly_on_v6e_4_persistent_TPUVM(4)

  @task_group(prefix_group_id=False)
  def vllm_request_rate_16_nightly_on_v6e_4_persistent_TPUVM():
    run_vllm_nightly_on_v6e_4_persistent_TPUVM(16)

  @task_group(prefix_group_id=False)
  def vllm_request_rate_inf_nightly_on_v6e_4_persistent_TPUVM():
    run_vllm_nightly_on_v6e_4_persistent_TPUVM(
        0
    )  # use 0 to present inf in this program

  # merge all PyTorch/XLA tests ino one Dag
  with models.DAG(
      dag_id="pytorch_xla_model_regression_test_on_trillium",
      schedule="0 0 * * *",
      tags=[
          "mantaray",
          "pytorchxla",
          "xlml",
          "TPU",
          "v6e-4",
          "v6e-8",
          "v6e-32",
      ],
      start_date=datetime.datetime(2024, 4, 22),
      catchup=False,
  ) as dag:
    # Training model(LLaMA3-8B, LLaMA3-70B, Mixtral8_7B, SD2)
    for workload_file_name in workload_file_name_list:
      run_workload = mantaray.run_workload.override(
          task_id=workload_file_name.split(".")[0]
      )(
          workload_file_name=workload_file_name,
      )
      run_workload

    # vLLM nightly Benchmark via inference_benchmark github repo script
    (
        vllm_request_rate_1_nightly_on_v6e_4_persistent_TPUVM()
        >> vllm_request_rate_4_nightly_on_v6e_4_persistent_TPUVM()
        >> vllm_request_rate_16_nightly_on_v6e_4_persistent_TPUVM()
        >> vllm_request_rate_inf_nightly_on_v6e_4_persistent_TPUVM()
    )

  # Create a DAG for each job from maxtext
  for job in xlml_jobs:
    if not re.match(pattern, job["task_name"]):
      with models.DAG(
          dag_id=job["task_name"],
          schedule=job["schedule"],
          tags=["mantaray", "TPU"],
          start_date=datetime.datetime(2024, 4, 22),
          catchup=False,
      ) as dag:
        run_workload = mantaray.run_workload.override(
            owner=test_owner.BHAVYA_B
        )(
            workload_file_name=job["file_name"],
        )
    run_workload
else:
  print(
      "Skipping creating Mantaray DAGs since not running in Prod or Dev composer environment."
  )
