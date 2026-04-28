# Copyright 2026 Google LLC
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

"""A DAG to upgrade GKE cluster to latest available version."""

import datetime
import json
import logging
import re

from airflow import models
from airflow.decorators import task
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess

DAG_ID = "gke_cluster_version_manager"


@task
def find_available_version(node_pool_info: node_pool.Info) -> str:
  """Finds the latest available GKE version."""
  region = node_pool_info.region
  if not region:
    raise ValueError("Region not found in node_pool_info")

  command = (
      f"gcloud container get-server-config --region={region} --format='json'"
  )
  logging.info("Running command: %s", command)
  stdout = subprocess.run_exec(command)

  output_json = json.loads(stdout)
  valid_versions = output_json.get("validMasterVersions", [])

  # Filter: ^1\.(3[2-9]|[4-9][0-9])
  pattern = re.compile(r"^1\.(3[2-9]|[4-9][0-9])")
  matching_versions = [v for v in valid_versions if pattern.match(v)]

  if not matching_versions:
    raise ValueError("No matching GKE versions found")

  latest_version = matching_versions[0]
  logging.info("Found latest available version: %s", latest_version)
  return latest_version


@task
def find_current_cluster_version(node_pool_info: node_pool.Info) -> dict:
  """Finds the current version of the cluster."""
  cluster_name = node_pool_info.cluster_name
  region = node_pool_info.region
  if not cluster_name or not region:
    raise ValueError("cluster_name or region not found in node_pool_info")

  command = (
      f"gcloud container clusters describe {cluster_name} "
      f"--region={region} --format='json'"
  )
  logging.info("Running command: %s", command)
  stdout = subprocess.run_exec(command)

  output_json = json.loads(stdout)
  current_master_version = output_json.get("currentMasterVersion")
  current_node_version = output_json.get("currentNodeVersion")

  logging.info("Current Master Version: %s", current_master_version)
  logging.info("Current Node Version: %s", current_node_version)

  return {
      "currentMasterVersion": current_master_version,
      "currentNodeVersion": current_node_version,
  }


@task
def upgrade_master(
    latest_version: str, current_versions: dict, node_pool_info: node_pool.Info
):
  """Upgrades the master to the target version if needed."""
  current_master = current_versions.get("currentMasterVersion")
  cluster_name = node_pool_info.cluster_name
  region = node_pool_info.region

  if current_master != latest_version:
    logging.info(
        "Master version (%s) != Target (%s). Upgrading.",
        current_master,
        latest_version,
    )
    command = (
        f"gcloud container clusters upgrade {cluster_name} --master "
        f"--cluster-version={latest_version} --region={region} --quiet"
    )
    logging.info("Running command: %s", command)
    subprocess.run_exec(command)
  else:
    logging.info("Master is already at target version. Skipping.")


@task
def upgrade_nodes(
    latest_version: str, current_versions: dict, node_pool_info: node_pool.Info
):
  """Upgrades all node pools to the target version if needed."""
  current_node = current_versions.get("currentNodeVersion")
  cluster_name = node_pool_info.cluster_name
  region = node_pool_info.region

  if current_node != latest_version:
    logging.info(
        "Node version (%s) != Target (%s). Upgrading.",
        current_node,
        latest_version,
    )
    command = (
        f"gcloud container clusters upgrade {cluster_name} "
        f"--region={region} --cluster-version={latest_version} --quiet"
    )
    logging.info("Running command: %s", command)
    subprocess.run_exec(command)
  else:
    logging.info("Nodes are already at target version. Skipping.")


@task
def verify_upgrade(target_version: str, node_pool_info: node_pool.Info):
  """Verifies that the upgrade was successful."""
  cluster_name = node_pool_info.cluster_name
  region = node_pool_info.region

  command = (
      f"gcloud container clusters describe {cluster_name} "
      f"--region={region} --format='json'"
  )
  logging.info("Running command: %s", command)
  stdout = subprocess.run_exec(command)

  output_json = json.loads(stdout)
  current_master_version = output_json.get("currentMasterVersion")
  current_node_version = output_json.get("currentNodeVersion")

  logging.info("Verifying versions against target: %s", target_version)
  logging.info("Post-upgrade Master Version: %s", current_master_version)
  logging.info("Post-upgrade Node Version: %s", current_node_version)

  if (
      current_master_version != target_version
      or current_node_version != target_version
  ):
    raise ValueError(
        f"Verification failed! Master: {current_master_version}, "
        f"Node: {current_node_version}, Target: {target_version}"
    )
  logging.info("Upgrade verified successfully!")


with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 8, 1),
    schedule=None,
    catchup=False,
    tags=["gke", "upgrade"],
    description="DAG to upgrade GKE cluster to latest available version",
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      node_pool_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name=DAG_ID,
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      avail_ver = find_available_version(node_pool_info)
      curr_vers = find_current_cluster_version(node_pool_info)

      master_up = upgrade_master(avail_ver, curr_vers, node_pool_info)
      node_up = upgrade_nodes(avail_ver, curr_vers, node_pool_info)
      verify = verify_upgrade(avail_ver, node_pool_info)

      curr_vers >> master_up >> node_up >> verify
