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

"""This Airflow DAG runs a maxtext machine learning benchmark on a GKE cluster

Usage:
gcloud composer environments run ml-automation-solutions \
  --project=cloud-ml-auto-solutions \
  --location=us-central1 dags trigger \
  -- \
  mlcompass_maxtext_gke \
  --conf={\\\"uuid\\\":\\\"abc\\\"} 70
"""

import datetime
from airflow import models
from airflow.decorators import task
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags import test_owner
from dags.vm_resource import Project, XpkClusters
from xlml.apis import gcp_config, metric_config, task as xlml_task, test_config
import json


def get_config_gke(
    docker_image: str,
    model_name: str,
    base_output_directory: str,
    task_owner: str = test_owner.ORTI_B,
    cluster: XpkClusterConfig = XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
    time_out_in_min: int = 60,
    num_slices: int = 1,
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.XLML_DATASET,
    dataset_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    composer_project: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
) -> xlml_task.XpkTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=dataset_name,
      dataset_project=dataset_project,
      composer_project=composer_project,
  )
  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=cluster.device_version,
          cores=cluster.core_count,
      ),
      test_name="maxtext",
      run_model_cmds=[
          f"source benchmark_run.sh;run {model_name} {base_output_directory}",
      ],
      set_up_cmds=None,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=task_owner,
      num_slices=num_slices,
      cluster_name=cluster.name,
      docker_image=docker_image,
  )
  return xlml_task.XpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


with models.DAG(
    dag_id="mlcompass_maxtext_gke",
    schedule=None,
    tags=["mlcompass", "maxtext"],
    start_date=datetime.datetime(2024, 9, 1),
    catchup=False,
    params={
        "uuid": "",
    },
    default_args={
        "retries": 0,
    },
) as dag:

  @task.python
  def load_xlml_state(params: dict = None):
    dag.log.info(params)
    uuid = params["uuid"]
    if not uuid:
      raise RuntimeError("uuid is not set")
    gcs_hook = GCSHook()
    file_content = gcs_hook.download(
        "mlcompass-jax-artifacts", f"xlml/{uuid}/xlml_state.json"
    )
    return json.loads(file_content)

  @task.python
  def get_docker_image_path(state: dict) -> str:
    return state["docker_image_path"]

  @task.python
  def get_model_name(state: dict) -> str:
    return state["model_name"]

  @task.python
  def get_base_output_directory(state: dict) -> str:
    bucket = state["workdir_bucket"]
    path = state["workdir_path"]
    return f"gs://{bucket}/{path}"

  xlml_state = load_xlml_state()
  docker_image_path = get_docker_image_path(xlml_state)
  model_name_arg = get_model_name(xlml_state)
  base_output_directory_arg = get_base_output_directory(xlml_state)

  default_benchmark = get_config_gke(
      docker_image=docker_image_path,
      model_name=model_name_arg,
      base_output_directory=base_output_directory_arg,
  ).run(skip_post_process=True)
