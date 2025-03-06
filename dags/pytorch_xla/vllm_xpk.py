# Copyright 2024 Google LLC
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

"""This Airflow DAG runs a vllm benchmark on a GKE cluster

Copy and reference from https://github.com/GoogleCloudPlatform/ml-auto-solutions/blob/master/dags/mlcompass/maxtext_gke.py

Usage:
gcloud composer environments run ml-automation-solutions \
  --project=cloud-ml-auto-solutions \
  --location=us-central1 dags trigger \
  -- \
  pytorch_tpu_vllm_xpk \
  --conf={\\\"uuid\\\":\\\"abc\\\"}
"""

import datetime
import json
from airflow import models
from airflow.decorators import task
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from dags.common import test_owner
from xlml.utils import xpk


with models.DAG(
    dag_id="pytorch_xla_model_regression_test_on_trillium",
    schedule="0 0 * * *",  # everyday at midnight # job["schedule"],
    tags=["mantaray", "pytorchxla", "xlml"],
    start_date=datetime.datetime(2024, 4, 22),
    catchup=False,
    # dag_id="pytorch_xla_model_regression_test_on_trillium", # pytorch_tpu_vllm_xpk",
    # schedule=None,
    # tags=["pytorch", "vllm", "xpk"],
    # start_date=datetime.datetime(2025, 3, 5),
    # catchup=False,
    # params={
    #     "uuid": "",
    # },
    # default_args={
    #     "retries": 0,
    # },
) as dag:

  # @task.python(multiple_outputs=True)
  # def load_xlml_state(params: dict = None):
  #   dag.log.info(params)
  #   uuid = params["uuid"]
  #   if not uuid:
  #     raise RuntimeError("uuid is not set")
  #   gcs_hook = GCSHook()
  #   file_content = gcs_hook.download(
  #       "mlcompass-jax-artifacts", f"xlml/{uuid}/xlml_state.json"
  #   )
  #   return json.loads(file_content)

  # xlml_state = load_xlml_state()

  # cluster_name = xlml_state["cluster_name"]
  # cluster_project = xlml_state["cluster_project"]
  # cluster_region = xlml_state["cluster_region"]
  # cluster_zone = xlml_state["cluster_zone"]
  # benchmark_id = xlml_state["test_name"]

  # docker_image_path = xlml_state["docker_image_path"]
  # accelerator_type = xlml_state["accelerator_type"]
  # num_slices = xlml_state["num_slices"]

  # model_name = xlml_state["model_name"]
  # workdir_bucket = xlml_state["workdir_bucket"]
  # workdir_path = xlml_state["workdir_path"]
  # gcs_path = f"gs://{workdir_bucket}/{workdir_path}"
  # workload_id = f'mlc-{xlml_state["uuid"]}'

  # workload_provision_timeout = datetime.timedelta(minutes=300).total_seconds()
  # workload_run_timeout = datetime.timedelta(minutes=60).total_seconds()

  run_workload_server = xpk.run_workload.override(owner=test_owner.MANFEI_B)(
      task_id="run_workload",
      cluster_project="cloud-tpu-multipod-dev",
      zone="europe-west4-b",
      cluster_name="b397493880-manfei3",
      benchmark_id="xlml.vllm.llama3-8b.1slice.v5p_128_xpk",
      workload_id="nightly-vllm-"+datetime.now(),
      gcs_path=f"gs://vllmnightlyxpk/vllmnightlyxpk/workload_id",
      docker_image="gcr.io/cloud-tpu-v2-images/vllm-tpu-nightly:latest",
      accelerator_type="v5p-8",
      run_cmds=f"bash nightly-benchmarks/scripts/run-nightly-benchmarks.sh",
      num_slices=1,
      use_vertex_tensorboard=False,
      use_pathways=False,
  )

  # run_workload_inference = xpk.run_workload.override(owner=test_owner.MANFEI_B)(
  #     task_id="run_workload",
  #     cluster_project="cloud-tpu-multipod-dev",
  #     zone="europe-west4-b",
  #     cluster_name="b397493880-manfei3",
  #     benchmark_id="xlml.vllm.llama3-8b.1slice.v5p_128_xpk",
  #     workload_id="nightly-vllm-"+datetime.now(),
  #     gcs_path=f"gs://vllmnightlyxpk/vllmnightlyxpk/workload_id",
  #     docker_image="gcr.io/cloud-tpu-v2-images/vllm-tpu-nightly:latest",
  #     accelerator_type="v5p-8",
  #     run_cmds=f"source benchmark_run.sh;run {model_name} {gcs_path}",
  #     num_slices=num_slices,
  #     use_vertex_tensorboard=False,
  #     use_pathways=False,
  # )
  
  wait_for_workload_start = xpk.wait_for_workload_start.override(
      timeout=workload_provision_timeout
  )(
      workload_id="nightly-vllm-"+datetime.now(),
      project_id="cloud-tpu-multipod-dev",
      region="europe-west4",
      cluster_name="b397493880-manfei3",
  )

  wait_for_workload_completion = xpk.wait_for_workload_completion.override(
      timeout=workload_run_timeout
  )(
      workload_id="nightly-vllm-"+datetime.now(),
      project_id="cloud-tpu-multipod-dev",
      region="europe-west4",
      cluster_name="b397493880-manfei3",
  )

  (
      run_workload_server
      >> wait_for_workload_start
      >> wait_for_workload_completion
  )
