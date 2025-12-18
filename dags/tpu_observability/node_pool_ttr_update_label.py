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

"""A DAG to validate GKE node pool Times To Recover(TTR) metrics by triggering a label update."""

import datetime

from airflow import models
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common.vm_resource import Region, Zone
from dags.tpu_observability.configs.common import MachineConfigMap
from dags.tpu_observability.utils import node_pool_util as node_pool


with models.DAG(
    dag_id="node_pool_ttr_update_label",
    start_date=datetime.datetime(2025, 9, 30),
    schedule="0 4 * * *",
    catchup=False,
    tags=[
        "gke",
        "tpu-observability",
        "node-pool-ttr-update-label",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG verifies the GKE node pool's Times To Recover(TTR) metrics "
        "by triggering a label update and confirming the recovery time "
        "is recorded"
    ),
    doc_md="""
      # GKE Node Pool Times To Recover(TTR) Metric Validation DAG

      ### Description
      This DAG automates the validation of GKE node pool Times To Recover(TTR) metrics.
      It creates a node pool and updates its labels then verifies that the TTR metric
      is correctly generated and reported to Google Cloud Monitoring.

      ### Prerequisites
      This test requires an existing GKE cluster.

      ### Procedures
      1. Create a temporary node pool.
      2. Wait for the node pool to be RUNNING.
      3. Update the node pool label.
      4. Wait for the Times To Recover(TTR) metrics to appear in Google Cloud Monitoring.
      5. Clean up the node pool after the tests.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value
    cluster_name = "tpu-observability-automation"
    cluster_name += "-prod" if composer_env.is_prod_env() else "-dev"
    node_pool_info = node_pool.Info(
        project_id=models.Variable.get("PROJECT_ID", default_var="cienet-cmcs"),
        cluster_name=cluster_name,
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME",
            default_var="ttr-update-label-v6e-autotest",
        ),
        location=models.Variable.get(
            "LOCATION", default_var=Region.US_CENTRAL1.value
        ),
        node_locations=models.Variable.get(
            "NODE_LOCATIONS", default_var=Zone.US_CENTRAL1_B.value
        ),
        num_nodes=models.Variable.get("NUM_NODES", default_var=4),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    LABELS_TO_UPDATE = {"test_key": "test_val"}

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      task_id = "create_node_pool"
      create_node_pool = node_pool.create.override(task_id=task_id)(
          node_pool=node_pool_info,
          reservation="cloudtpu-20251107233000-1246578561",
      )

      task_id = "wait_for_provisioning"
      wait_for_provisioning = node_pool.wait_for_status.override(
          task_id=task_id
      )(node_pool=node_pool_info, status=node_pool.Status.PROVISIONING)

      task_id = "wait_for_running"
      wait_for_running = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RUNNING
      )

      task_id = "update_node_pool_label"
      update_node_pool_label = node_pool.update_labels.override(
          task_id=task_id
      )(node_pool=node_pool_info, node_labels=LABELS_TO_UPDATE)

      task_id = "wait_for_recovered"
      wait_for_recovered = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RUNNING
      )

      task_id = "wait_for_ttr"
      wait_for_ttr = node_pool.wait_for_ttr.override(task_id=task_id)(
          node_pool=node_pool_info, operation_start_time=update_node_pool_label
      )

      task_id = "cleanup_node_pool"
      cleanup_node_pool = node_pool.delete.override(
          task_id=task_id, trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_node_pool,
      )

      _ = (
          create_node_pool
          >> wait_for_provisioning
          >> wait_for_running
          >> update_node_pool_label
          >> wait_for_recovered
          >> wait_for_ttr
          >> cleanup_node_pool
      )
