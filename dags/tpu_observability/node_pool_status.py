"""A DAG to validate the status of a GKE node pool through its lifecycle."""

import copy
import datetime

from airflow import models
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags.map_reproducibility.utils import constants
from dags.common.vm_resource import Project, Region, Zone
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.configs.common import MachineConfigMap, log_metadata


with models.DAG(
    dag_id="gke_node_pool_status",
    start_date=datetime.datetime(2025, 8, 1),
    schedule=constants.Schedule.DAILY_PST_6PM,
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status", "TPU", "v6e-16"],
    description=(
        "This DAG tests whether the status of a GKE node pool changes as "
        "expected according to its lifecycle."
    ),
    doc_md="""
      # GKE Node Pool Status Validation DAG

      ### Description
      This DAG automates the process of going through the lifecycle of a GKE
      node pool and verifies whether the node pool status is reported correctly.

      ### Prerequisites
      This test requires an existing cluster.

      ### Procedures
      It creates a node pool, waits for it from provisioning to be running,
      deletes a random node to trigger reconciliation, waits for it to become
      running again, and finally cleans up.
      It also tests the error state by creating a node pool with invalid
      parameters and verifies that the status changes to error.
      All node-pool will be cleaned up clean it up after the tests.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value
    node_pool_info = node_pool.Info(
        project_id=models.Variable.get("PROJECT_ID", default_var="cienet-cmcs"),
        cluster_name=models.Variable.get(
            "CLUSTER_NAME", default_var="tpu-observability-automation"
        ),
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME", default_var="node-pool-status-v6e-autotest"
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

    problematic_node_pool_info = copy.deepcopy(node_pool_info)
    problematic_node_pool_info.node_pool_name += "-wrong"
    # Choosing a region that is different from the cluster location but still
    # compatible with the specified TPU cause the cluster creation to fail
    # due to mismatched node locations.
    problematic_node_pool_info.node_locations = models.Variable.get(
        "WRONG_NODE_LOCATION", default_var=Zone.ASIA_EAST1_C.value
    )

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      task_id = "get_log_metadata"
      log_op = log_metadata.override(task_id=task_id)(
          cluster_project=node_pool_info.project_id,
          region=node_pool_info.region,
          zone=node_pool_info.zone,
          cluster_name=node_pool_info.cluster_name,
          node_pool_name=node_pool_info.node_pool_name,
          workload_id="",
          docker_image="",
          accelerator_type=node_pool_info.machine_type,
          num_slices="1",
      )

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

      task_id = "delete_node"
      delete_node = node_pool.delete_one_random_node.override(task_id=task_id)(
          node_pool=node_pool_info
      )

      task_id = "wait_for_repair"
      wait_for_repair = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RECONCILING
      )

      task_id = "wait_for_recovered"
      wait_for_recovered = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.RUNNING
      )

      task_id = "delete_node_pool"
      delete_node_pool = node_pool.delete.override(task_id=task_id)(
          node_pool=node_pool_info
      )

      task_id = "wait_for_stopping"
      wait_for_stopping = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=node_pool_info, status=node_pool.Status.STOPPING
      )

      task_id = "cleanup_node_pool"
      cleanup_node_pool = node_pool.delete.override(
          task_id=task_id, trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=node_pool_info).as_teardown(
          setups=create_node_pool,
      )

      # Intentionally create a node pool with problematic configurations
      # to validate that it enters the ERROR state.
      task_id = "create_problematic_node_pool_info"
      create_problematic_node_pool_info = node_pool.create.override(
          task_id=task_id
      )(
          node_pool=problematic_node_pool_info,
          # The failure is intentionally ignored because we want to validate
          # that the status of the node pool (which fails to be created) is
          # "ERROR".
          ignore_failure=True,
      )

      task_id = "wait_for_error"
      wait_for_error = node_pool.wait_for_status.override(task_id=task_id)(
          node_pool=problematic_node_pool_info, status=node_pool.Status.ERROR
      )

      task_id = "cleanup_wrong_node_pool"
      cleanup_wrong_node_pool = node_pool.delete.override(
          task_id=task_id, trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=problematic_node_pool_info).as_teardown(
          setups=create_problematic_node_pool_info,
      )

      normal_flow = (
          create_node_pool
          >> wait_for_provisioning
          >> wait_for_running
          >> delete_node
          >> wait_for_repair
          >> wait_for_recovered
          >> delete_node_pool
          >> wait_for_stopping
          >> cleanup_node_pool
      )

      flow_for_error_state = (
          create_problematic_node_pool_info
          >> wait_for_error
          >> cleanup_wrong_node_pool
      )
