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

"""Utilities to run workloads with xpk (https://github.com/google/xpk)."""

import uuid
from absl import logging
from airflow.decorators import task
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes import client as kubernetes_client, config as kubernetes_config


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate a workload ID."""
  short_id = str(uuid.uuid4())[:8]
  return f"{benchmark_id}-{short_id}"


def run_workload(
    task_id: str,
    project_id: str,
    zone: str,
    cluster_name: str,
    benchmark_id: str,
    workload_id: str,
    docker_image: str,
    accelerator_type: str,
    run_cmds: str,
    task_owner: str,
    num_slices: int = 1,
) -> KubernetesPodOperator:
  """Run workload through xpk tool.

  The reason to use KubernetesPodOperator instead of BashOperator is that
  xpk must run with Python 3.10 or greater; however, the latest version in
  Composer is Python 3.8, and it's non-trivial to upgrade it as the  Composer
  uses docker images that bundle Airflow releases with Python and other
  libraries.
  """

  cmds = (
      "set -x",
      f"gcloud config set project {project_id}",
      f"gcloud config set compute/zone {zone}",
      "git clone -b xpk-namespace https://github.com/google/xpk.git /tmp/xpk",
      "cd /tmp/xpk",
      (
          "python3 xpk.py workload create"
          f" --cluster={cluster_name} --workload={workload_id} --command='{run_cmds}'"
          f" --tpu-type={accelerator_type} --num-slices={num_slices} --docker-image={docker_image} --namespace=default"
      ),
  )

  return KubernetesPodOperator(
      task_id=task_id,
      name=benchmark_id,
      cmds=["/bin/bash", "-c"],
      arguments=[";".join(cmds)],
      namespace="composer-user-workloads",
      image=docker_image,
      config_file="/home/airflow/composer_kube_config",
      kubernetes_conn_id="kubernetes_default",
      owner=task_owner,
  )


@task.sensor(poke_interval=60, timeout=600, mode="reschedule")
def wait_for_workload_completion(workload_id: str, cluster_config: str) -> bool:
  """Check the workload status."""

  # Load the config for the cluster with TPUs in the pool
  kubernetes_config.load_kube_config(
      config_file=f"/home/airflow/gcs/dags/configs/cluster/{cluster_config}"
  )
  core_api = kubernetes_client.CoreV1Api()

  logging.info(f"workload_id: {workload_id}")
  pods = core_api.list_namespaced_pod(
      label_selector=f"jobset.sigs.k8s.io/jobset-name={workload_id}",
      namespace="default",
  )

  if not pods.items:
    RuntimeError(f"No pod is found for workload selector: {pods}.")

  for pod in pods.items:
    if pod.status.phase in ["Pending", "Running"]:
      logging.info(f"One pod phase is: {pod.status.phase}")
      return False
    elif pod.status.phase in ["Failed", "Unknown"]:
      RuntimeError(f"Bad pod phase: {pod.status.phase}")

  logging.info("All pod(s) phase are succeeded.")
  return True
