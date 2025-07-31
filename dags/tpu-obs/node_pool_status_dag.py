"""Manages the lifecycle of a GKE node pool and verifies its status as an Airflow DAG.
"""

import datetime
import os
import sys

from airflow import models
import node_pool_util
from node_pool_util import Status
from airflow.operators.empty import EmptyOperator

# --------------------------------------------------------------------------------
# AIRFLOW DAG DEFINITION
# --------------------------------------------------------------------------------

with models.DAG(
    dag_id="gke_node_pool_status_dag",
    start_date=datetime.datetime(2025, 7, 30),
    schedule=datetime.timedelta(days=1),
    catchup=False,
    tags=["gke", "tpu-observability", "node-pool-status"],
    doc_md="""
    ### GKE Node Pool Lifecycle Management DAG

    This DAG automates the lifecycle of a GKE node pool for testing purposes.
    It performs the following steps:
    1.  **Normal Path**: Creates a node pool, waits for it to be running, deletes a random node to trigger reconciliation, waits for it to become running again, and finally cleans up.
    2.  **Error Path**: Concurrently, it attempts to create a node pool with invalid parameters to test the error state, and then cleans up.
    """,
) as dag:

  # Instantiate the GKENodePool class within a task
  node_pool_info = node_pool_util.Info(
      project_id=models.Variable.get(
          "PROJECT_ID", default_var="tpu-prod-env-one-vm"
      ),
      cluster_name=models.Variable.get(
          "CLUSTER_NAME", default_var="yuna-xpk-v6e-2"
      ),
      node_pool_name=models.Variable.get(
          "NODE_POOL_NAME", default_var="yuna-v6e-autotest"
      ),
      location=models.Variable.get("LOCATION", default_var="asia-northeast1"),
      node_locations=models.Variable.get(
          "NODE_LOCATIONS", default_var="asia-northeast1-b"
      ),
      num_nodes=models.Variable.get("NUM_NODES", default_var=4),
      machine_type=models.Variable.get(
          "MACHINE_TYPE", default_var="ct6e-standard-4t"
      ),
      tpu_topology=models.Variable.get("TPU_TOPOLOGY", default_var="4x4"),
  )

  error_node_pool_info = node_pool_util.Info(
      project_id=node_pool_info.project_id,
      cluster_name=node_pool_info.cluster_name,
      node_pool_name=node_pool_info.node_pool_name + "-error",
      location=node_pool_info.location,
      node_locations=models.Variable.get(
          "ERROR_NODE_LOCATIONS", default_var="asia-east1-c"
      ),
      num_nodes=node_pool_info.num_nodes,
      machine_type=node_pool_info.machine_type,
      tpu_topology=node_pool_info.tpu_topology,
  )
  # =================================================================================
  # Normal Path Tasks
  # =================================================================================

  """STEP 1: Creates the GKE node pool."""
  create_node_pool = node_pool_util.create.override(
      task_id="create_node_pool")(info=node_pool_info)  # Force success for testing purposes

  """STEP 2: Validating Provisioning Status."""
  wait_for_provisioning = node_pool_util.wait_for_status.override(task_id="wait_for_provisioning")(
      info=node_pool_info,
      status=Status.PROVISIONING,
  )

  """STEP 3: Validating Running Status."""
  wait_for_running_initial = node_pool_util.wait_for_status.override(task_id="wait_for_running_initial")(
      info=node_pool_info,
      status=Status.RUNNING
  )

  """STEP 4: Deleting a random node to trigger reconciliation."""
  delete_node = node_pool_util.delete_node(
      info=node_pool_info
  )

  """STEP 5: Validating Reconciling Status."""
  wait_for_reconciling = node_pool_util.wait_for_status.override(task_id="wait_for_reconciling")(
      info=node_pool_info,
      status=Status.RECONCILING
  )

  """STEP 6: Validating Running Status After Repair."""
  wait_for_running_after_repair = node_pool_util.wait_for_status.override(task_id="wait_for_running_after_repair")(
      info=node_pool_info,
      status=Status.RUNNING
  )

  # TODO: In this DAG, cleaning up and verifying the state is part of the
  # requirement. However, in other DAGs, resource cleanup may fail, but it is
  # not part of the verification requirement. Therefore, even if the
  # cleanup process fails, the overall task should still be considered
  # successful. Please confirm what the overall task result would be when
  # trigger_rule=ALL_DONE.
  """STEP 7: Cleaning up - Deleting Node Pool."""
  delete_node_pool = node_pool_util.delete.override(
      task_id="delete_node_pool",trigger_rule="all_done")(
      info=node_pool_info
  )

  """STEP 8: Validating Stopping Status."""
  wait_for_stopping = node_pool_util.wait_for_status.override(task_id="wait_for_stopping",
      trigger_rule="all_done")(
      info=node_pool_info,
      status=Status.STOPPING
  )

  # =================================================================================
  # Error Path Tasks
  # =================================================================================

  # This task must be successful in airflow, if it have some issues next task will go to error state.
  """STEP 1: Creating Error Node Pool."""
  create_error_node_pool = node_pool_util.create.override(task_id="create_error_node_pool")(
      info=error_node_pool_info,
      force_task_success=True
  )
  """STEP 2: Validating Error Status."""
  wait_for_error = node_pool_util.wait_for_status.override(task_id="wait_for_error")(
      info=error_node_pool_info,
      status=Status.ERROR
  )
  """STEP 3: Cleaning up - Deleting Error Node Pool."""
  delete_error_node_pool = node_pool_util.delete.override(
    task_id="delete_error_node_pool",
    trigger_rule="all_done"
    )(
    info=error_node_pool_info)

  # =================================================================================
  # Define Task Dependencies
  # =================================================================================
  end = EmptyOperator(
        task_id="final_status_check",
        trigger_rule="all_success",  # 所有上游都成功，DAG 才標成功
    )
  
  # Normal path workflow
  normal_path_flow = (
      create_node_pool
      >> wait_for_provisioning
      >> wait_for_running_initial
      >> delete_node
      >> wait_for_reconciling
      >> wait_for_running_after_repair
  )

  # The cleanup task for the normal path runs after the main flow is complete
  # (success or fail)
  normal_path_flow >> delete_node_pool >> wait_for_stopping >> end

  # Error path workflow (runs in parallel to the normal path)
  error_path_flow = (
      create_error_node_pool >> wait_for_error
  )
  
  # The cleanup task for the error path runs after it's complete
  error_path_flow >> delete_error_node_pool >> end
