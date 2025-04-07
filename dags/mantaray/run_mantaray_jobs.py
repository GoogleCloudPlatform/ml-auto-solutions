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
# import _datetime
from airflow import models
from xlml.utils import mantaray
import yaml
from dags import composer_env
import re
from airflow.decorators import task
from xlml.utils import xpk
from dags.common import test_owner

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

  @task.python(multiple_outputs=True)
  def hello_world_vllm(params: dict = None):
    dag.log.info(params)
    print("Hello world vLLM!")

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
    # hello_world_vllm
    workload_id="nightly-vllm-"+datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    cluster_name="b397493880-repo3"
    cluster_project="cloud-tpu-multipod-dev"
    zone="europe-west4-b"
    region="europe-west4"
    workload_provision_timeout = datetime.timedelta(seconds=30).total_seconds()
    workload_run_timeout = datetime.timedelta(minutes=3).total_seconds()
    hello_world_vllm_xpk = xpk.run_workload.override(owner=test_owner.MANFEI_B)(
      task_id="run_workload_vllm_xpk",
      cluster_project=cluster_project,
      zone=zone,
      cluster_name=cluster_name, # "b397493880-manfei3",
      benchmark_id="xlml.vllm.llama3-8b.1slice.v5p_128_xpk",
      workload_id=workload_id,
      gcs_path=f"gs://vllmnightlyxpk/vllmnightlyxpk/workload_id",
      docker_image="gcr.io/cloud-tpu-v2-images/vllm-tpu-nightly:latest",
      accelerator_type="v5p-128",
      run_cmds=f"export HF_TOKEN=xxxxx && \
      export VLLM_SOURCE_CODE_LOC=./ && \
      vllm serve meta-llama/Meta-Llama-3.1-8B --swap-space 16  --disable-log-requests --tensor_parallel_size=8 --max-model-len=2048 --num-scheduler-steps=4 & sleep 600 \
      ",
      num_slices=1,
      use_vertex_tensorboard=False,
      use_pathways=False,
    )

    wait_for_workload_start = xpk.wait_for_workload_start.override(
        timeout=workload_provision_timeout
    )(
        workload_id=workload_id,
        project_id=cluster_project,
        region=region,
        cluster_name=cluster_name,
    )

    wait_for_workload_completion = xpk.wait_for_workload_completion.override(
        timeout=workload_run_timeout
    )(
        workload_id=workload_id,
        project_id=cluster_project,
        region=region,
        cluster_name=cluster_name,
    )

    (
        hello_world_vllm_xpk
        >> wait_for_workload_start
        >> wait_for_workload_completion
    )

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
