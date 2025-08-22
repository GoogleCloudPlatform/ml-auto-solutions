"""A DAG to ensure a rollback effects the availablility of a mult-host GKE node pool as expected."""

import datetime

from airflow import models
from airflow.models import Variable

from dags.common.vm_resource import Project, Region, Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import node_pool_util as node_pool


with models.DAG(
    dag_id="multi-host-availability-rollback",
    start_date=datetime.datetime(2025, 8, 10),
    schedule=constants.Schedule.WEEKDAY_PST_6PM_EXCEPT_THURSDAY,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "multi-host-availability",
        "tpu_obervability",
        "rollback",
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
  node_pool_info = node_pool.Info(
      project_id=Project.TPU_PROD_ENV_ONE_VM.value,
      cluster_name=Variable.get(
          "CLUSTER_NAME", default_var="qmcgarry-auto-test"
      ),
      node_pool_name=Variable.get(
          "NODE_POOL_NAME", default_var="nodepool-auto"
      ),
      location=Variable.get(
          "LOCATION", default_var=Region.ASIA_NORTHEAST1.value
      ),
      node_locations=Variable.get(
          "NODE_LOCATIONS", default_var=Zone.ASIA_NORTHEAST1_B.value
      ),
      num_nodes=Variable.get("NUM_NODES", default_var=4),
      machine_type=Variable.get("MACHINE_TYPE", default_var="ct6e-standard-4t"),
      tpu_topology=Variable.get("TPU_TOPOLOGY", default_var="4x4"),
  )

  create_node_pool = node_pool.create(node_pool=node_pool_info)

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

  cleanup_node_pool = node_pool.delete.override(trigger_rule="all_done")(
      node_pool=node_pool_info
  ).as_teardown(
      setups=create_node_pool,
  )

  (
      create_node_pool
      >> wait_node_pool_available
      >> rollback_node_pool
      >> wait_node_pool_unavailable
      >> wait_node_pool_recovered
      >> cleanup_node_pool
  )
