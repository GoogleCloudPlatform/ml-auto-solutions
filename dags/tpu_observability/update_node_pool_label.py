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

"""
A DAG to update the label of a node pool to make node pool unavailable
"""

import datetime

from airflow import models
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common.vm_resource import Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.configs.common import MachineConfigMap
from dags.tpu_observability.utils import node_pool_util as node_pool

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="gke_node_pool_label_update",
    start_date=datetime.datetime(2025, 8, 1),
    schedule=constants.Schedule.DAILY_PST_7_30_PM,
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status", "TPU", "v6e-16"],
    description=(
        "This DAG tests whether the status of a GKE node pool changes as "
        "expected after its labels are updated, triggering reconciliation."
    ),
    doc_md="""
      # GKE Node Pool Label Update Status Validation DAG

      ### Description
      This DAG automates the process of going through the lifecycle of a GKE
      node pool and verifies whether the node pool status is reported correctly
      after a configuration change (label update) is applied.

      ### Prerequisites
      This test requires an existing cluster.

      ### Procedures
      It creates a node pool, waits for it to be running, updates a label to
      trigger reconciliation, waits for it to become running again (recovering
      from the update), and finally cleans up by deleting the node pool.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value
    cluster_name = "tpu-observability-automation"
    cluster_name += "-prod" if composer_env.is_prod_env() else "-dev"
    node_pool_info = node_pool.Info(
        project_id="cienet-cmcs",
        cluster_name=cluster_name,
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME", default_var="update-node-pool-label-v6e-autotest"
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

    LABELS_TO_UPDATE = {"env": "prod"}

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=node_pool_info,
          reservation="cloudtpu-20251107233000-1246578561",
      )

      wait_for_availability = node_pool.wait_for_availability.override(
          task_id="wait_for_initial_availability"
      )(node_pool=node_pool_info, availability=True)

      update_node_pool_label = node_pool.update_labels.override(
          task_id="update_node_pool_label"
      )(node_pool=node_pool_info, node_labels=LABELS_TO_UPDATE)

      wait_for_unavailable = node_pool.wait_for_availability.override(
          task_id="wait_for_unavailability_after_update"
      )(node_pool=node_pool_info, availability=False)

      wait_node_pool_recovered = node_pool.wait_for_availability.override(
          task_id="wait_for_recovery"
      )(node_pool=node_pool_info, availability=True)

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=[create_node_pool],
      )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          create_node_pool
          >> wait_for_availability
          >> update_node_pool_label
          >> wait_for_unavailable
          >> wait_node_pool_recovered
          >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
