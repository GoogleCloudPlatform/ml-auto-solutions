# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0 #
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to run workloads with KubernetesPodOperator"""

import datetime as dt
import os
from absl import logging

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.models.taskmixin import DAGNode
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.task_group import TaskGroup

_KPO_LABEL = {"kpo-label": "cli_kpo_worker"}

# pylint: disable=line-too-long
_KPO_NAMESPACE = "composer-user-workloads"
"""
**MUST** use this fixed namespace for Cloud Composer 2.
See: https://cloud.google.com/composer/docs/composer-2/use-kubernetes-pod-operator#composer-2-kpo-access-project-resources
"""
# pylint: enable=line-too-long

_COMPOSER_KUBECONFIG_PATH = "/home/airflow/composer_kube_config"


@task(
    retry_delay=dt.timedelta(seconds=15),
    retries=4,
)
def reset_kube_config() -> None:
  """Get credential for in-cluster to setup CLI command."""

  cluster_name = os.environ["COMPOSER_GKE_NAME"]
  project_id = os.environ["GCP_PROJECT"]
  region = os.environ["COMPOSER_LOCATION"]

  logging.info(f"{' LOGGING AIRFLOW CLUSTER ':=^80}")
  logging.info("CLUSTER_NAME: %s", cluster_name)
  logging.info("PROJECT_ID: %s", project_id)
  logging.info("REGION: %s", region)

  hook = SubprocessHook()
  result = hook.run_command([
      "bash",
      "-c",
      f"sudo chown -R airflow:airflow {_COMPOSER_KUBECONFIG_PATH} && "
      f"gcloud container clusters get-credentials {cluster_name} "
      f"--region {region} --project {project_id}",
  ])

  assert (
      result.exit_code == 0
  ), f"XPK clean-up failed with code {result.exit_code}"


def run_command_in_kpo(
    start_cli_command: str,
    workload_id: str,
    task_owner: str,
    provisioning_timeout: dt.timedelta,
    workload_run_timeout: dt.timedelta,
    image_full_url: str,
) -> DAGNode:
  """
  Launch an isolated Kubernetes Pod (via @task.kubernetes) and
  wait until the workload's Pods become Ready (or the wait times out).

  Airflow injects this function as a temporary Python script into the Pod and
  executes it there. Because only the Python standard library is guaranteed to
  be available at runtime (unless baked into the image), this function avoids
  third-party imports.

  Args:
    start_cli_command: Full shell command that starts the CLI ( will
      submit the workload to GKE).
    workload_id: Workload/JobSet identifier used to discover Pods (e.g., via a
      label selector).
    task_owner: The owner of the task, used for Airflow metadata and
      pod labeling.
    provisioning_timeout: Timedelta object representing the time reserved for
      resource provisioning.
    workload_run_timeout: Timedelta object representing the maximum allowed
      execution time for the Airflow task.
    image_full_url: The full URL of the Docker image.

  Returns:
    None. The task completes when readiness is observed.

  Raises:
    AirflowFailException: If the CLI command fails, or if Pods do not
      become Ready before the timeout.
  """

  airflow_task_timeout = workload_run_timeout.total_seconds()
  k8s_timeout = airflow_task_timeout - provisioning_timeout.total_seconds()

  with TaskGroup(group_id="run_command_in_kpo") as group:
    # The default worker pod's kube_config may contain leftover
    # contexts/configs from other tasks (e.g., after `gcloud container clusters
    # get-credentials`) that point to a different GKE cluster.
    # Reset it so kubectl/gcloud—and the KPO we launch—target the Composer
    # cluster and use the correct Workload Identity–backed SA.
    reset_kube_config_task = reset_kube_config.override(owner=task_owner)()

    kpo = KubernetesPodOperator(
        task_id="run_kpo-cli",
        name="cli-kpo",
        namespace=_KPO_NAMESPACE,
        config_file=_COMPOSER_KUBECONFIG_PATH,
        image=image_full_url,
        cmds=["bash", "-cx", start_cli_command],
        labels={
            **_KPO_LABEL,
            "workload_id": workload_id,
            "owner": task_owner,
        },
        active_deadline_seconds=int(k8s_timeout),
        execution_timeout=workload_run_timeout,
        retries=0,
    )

    _ = reset_kube_config_task >> kpo

  return group
