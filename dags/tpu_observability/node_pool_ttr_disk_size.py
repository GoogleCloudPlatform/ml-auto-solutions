# Copyright 2025 Google LLC
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

"""A DAG to validate GKE node pool Times To Recover(TTR) metrics
by triggering a disk size update."""

import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH
from dags.tpu_observability.utils import node_pool_util as node_pool

_DISK_SIZE_INCREMENT = 50


with models.DAG(
    dag_id="node_pool_ttr_disk_size",
    start_date=datetime.datetime(2025, 6, 26),
    schedule="0 21 * * *" if composer_env.is_prod_env() else None,
    catchup=False,
    tags=[
        "gke",
        "tpu-observability",
        "node-pool-ttr-disk-size",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG verifies the GKE node pool's Times To Recover(TTR) metrics "
        "by triggering a disk size update and confirming the recovery time "
        "is recorded"
    ),
    doc_md="""
      # GKE Node Pool Times To Recover (TTR) Metric Validation DAG (Disk Resize)

      ### Description
      This DAG automates the validation of GKE node pool Times To Recover (TTR) metrics.
      It creates a temporary node pool, triggers a disk resize operation to force a node
      pool update, and checks that the TTR metric is correctly generated and reported
      to Google Cloud Monitoring.

      ### Prerequisites
      This test requires an existing GKE cluster.

      ### Procedures
      1. Create a temporary node pool.
      2. Wait for the node pool to be RUNNING.
      3. Trigger a disk resize update on the node pool.
      4. Wait for the node pool to recover and become RUNNING again.
      5. Wait for the Times To Recover (TTR) metrics to appear in Google Cloud Monitoring.
      6. Clean up the node pool after the tests.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      node_pool_info = node_pool.build_node_pool_info_from_gcs_yaml(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="node_pool_ttr_disk_size",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=node_pool_info
      )

      wait_for_provisioning = node_pool.wait_for_status.override(
          task_id="wait_for_provisioning"
      )(node_pool=node_pool_info, status=node_pool.Status.PROVISIONING)

      wait_for_running = node_pool.wait_for_status.override(
          task_id="wait_for_running"
      )(node_pool=node_pool_info, status=node_pool.Status.RUNNING)

      update_start_time = node_pool.update.override(task_id="update_node_pool")(
          node_pool=node_pool_info,
          spec=node_pool.NodePoolUpdateSpec.DiskSize(
              delta=_DISK_SIZE_INCREMENT
          ),
      )

      wait_for_recovered = node_pool.wait_for_status.override(
          task_id="wait_for_recovered"
      )(node_pool=node_pool_info, status=node_pool.Status.RUNNING)

      wait_for_ttr = node_pool.wait_for_ttr(
          node_pool=node_pool_info, operation_start_time=update_start_time
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          node_pool_info,
          create_node_pool,
          wait_for_provisioning,
          wait_for_running,
          update_start_time,
          wait_for_recovered,
          wait_for_ttr,
          cleanup_node_pool,
      )
