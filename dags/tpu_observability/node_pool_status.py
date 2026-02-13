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

"""A DAG to validate the status of a GKE node pool through its lifecycle."""

import datetime

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout


DAG_ID = "gke_node_pool_status"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 8, 1),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
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

    @task
    def generate_problematic_node_pool_name(
        node_pool_info: node_pool.Info,
    ) -> str:
      """Generates a problematic node pool name."""
      return f"{node_pool_info.node_pool_name}-x"

    @task
    def generate_problematic_node_location(
        node_pool_info: node_pool.Info,
    ) -> str:
      """Generates a problematic node location."""
      return f"{node_pool_info.location}-c"

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      node_pool_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name=DAG_ID,
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      problematic_node_pool_info = node_pool.copy_node_pool_info_with_override(
          info=node_pool_info,
          node_pool_name=generate_problematic_node_pool_name(node_pool_info),
          node_locations=generate_problematic_node_location(node_pool_info),
      )

      task_id = "create_node_pool"
      create_node_pool = node_pool.create.override(
          task_id=task_id,
          owner=test_owner.YUNA_T,
      )(
          node_pool=node_pool_info,
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
          task_id=task_id,
          owner=test_owner.YUNA_T,
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

      chain(
          node_pool_info,
          problematic_node_pool_info,
          create_node_pool,
          wait_for_provisioning,
          wait_for_running,
          delete_node,
          wait_for_repair,
          wait_for_recovered,
          delete_node_pool,
          wait_for_stopping,
          cleanup_node_pool,
      )

      chain(
          create_problematic_node_pool_info,
          wait_for_error,
          cleanup_wrong_node_pool,
      )
