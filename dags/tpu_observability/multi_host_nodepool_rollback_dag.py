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
A DAG to ensure a rollback effects the availability of a multi-host GKE node
pool as expected.
"""

import datetime

from airflow import models
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags.map_reproducibility.utils import constants
from dags.common.vm_resource import Region, Zone
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.configs.common import MachineConfigMap


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="multi-host-availability-rollback",
    start_date=datetime.datetime(2025, 8, 10),
    schedule=constants.Schedule.DAILY_PST_6_30PM,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "multi-host-availability",
        "tpu-observability",
        "rollback",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the use of a node-pool rollback to interrupt a "
        "multi-host node-pool and ensures the node-pool is interrupted and "
        "then recovers"
    ),
    doc_md="""
    # Multi-host Node-Pool Availability Test Using Node-Pool Rollback

    ### Description
    This DAG automates the process of creating a multi-host node-pool, then
    using a node-pool rollback to interrupt the node-pool, while checking if
    the availability is correct at each step. Finally the DAG cleans up the
    node-pool which was created.

    ### Prerequisites
    This test requires an existing cluster.

    ### Procedures
    First the node-pool is created, if it found to be available the rollback
    is run. Once the rollback is finished the node-pool availability is
    tested to make sure the interruption was recorded. Afterwards, a final
    measurement is taken to ensure that the node-pool recovers from the
    inerrupt. If all of these tasks succeed than the test is successful.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value
    node_pool_info = node_pool.Info(
        project_id="cienet-cmcs",
        cluster_name=Variable.get(
            "CLUSTER_NAME", default_var="tpu-observability-automation"
        ),
        node_pool_name=Variable.get(
            "NODE_POOL_NAME", default_var="multi-host-nodepool-rollback-auto"
        ),
        location=Variable.get("LOCATION", default_var=Region.US_CENTRAL1.value),
        node_locations=Variable.get(
            "NODE_LOCATIONS", default_var=Zone.US_CENTRAL1_B.value
        ),
        num_nodes=Variable.get("NUM_NODES", default_var=4),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      create_node_pool = node_pool.create(
          node_pool=node_pool_info,
          reservation="cloudtpu-20251107233000-1246578561",
      )

      wait_node_pool_available = node_pool.wait_for_availability(
          node_pool=node_pool_info, availability=True
      )

      rollback_node_pool = node_pool.rollback(node_pool=node_pool_info)

      wait_node_pool_unavailable = node_pool.wait_for_availability(
          node_pool=node_pool_info, availability=False
      )

      # A successful rollback means the availability will return to True.
      # The end of the rollback marks the start the availability, so
      # the client side should see the state change, and update the metric.
      wait_node_pool_recovered = node_pool.wait_for_availability(
          node_pool=node_pool_info, availability=True
      )

      cleanup_node_pool = node_pool.delete.override(
          trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_node_pool,
      )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          create_node_pool
          >> wait_node_pool_available
          >> rollback_node_pool
          >> wait_node_pool_unavailable
          >> wait_node_pool_recovered
          >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
