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
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.common.scheduling_helper.scheduling_helper import (
    SchedulingHelper,
    get_dag_timeout,
)
from dags.common.task_group_with_timeout import TaskGroupWithTimeout
from dags.tpu_observability.configs.common import (
    GCS_CONFIG_PATH,
    MachineConfigMap,
)
from dags.tpu_observability.utils import node_pool_util as node_pool

DAG_ID = "gke_node_pool_label_update"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

PRE_TEST_TIMEOUT = datetime.timedelta(minutes=10)
POST_TEST_TIMEOUT = datetime.timedelta(minutes=10)
TEST_TIMEOUT = DAGRUN_TIMEOUT - PRE_TEST_TIMEOUT - POST_TEST_TIMEOUT

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
    labels_to_update = (
        {"env": "prod"} if composer_env.is_prod_env() else {"env": "dev"}
    )

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      node_pool_info = node_pool.build_node_pool_info_from_gcs_yaml(
          gcs_path=GCS_CONFIG_PATH,
          dag_name=DAG_ID,
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      with TaskGroupWithTimeout(
          group_id="pre_test",
          timeout=PRE_TEST_TIMEOUT,
      ) as pre_test:
        create_node_pool = node_pool.create.override(
            task_id="create_node_pool",
            owner=test_owner.YUNA_T,
        )(
            node_pool=node_pool_info,
        )

        wait_for_availability = node_pool.wait_for_availability.override(
            task_id="wait_for_initial_availability"
        )(node_pool=node_pool_info, availability=True)

        chain(create_node_pool, wait_for_availability)

      with TaskGroupWithTimeout(
          group_id="test",
          timeout=TEST_TIMEOUT,
      ) as test:
        update_node_pool_label = node_pool.update.override(
            task_id="update_node_pool_label"
        )(
            node_pool=node_pool_info,
            spec=node_pool.NodePoolUpdateSpec.Label(delta=labels_to_update),
        )

        wait_for_unavailable = node_pool.wait_for_availability.override(
            task_id="wait_for_unavailability_after_update"
        )(node_pool=node_pool_info, availability=False)

        wait_node_pool_recovered = node_pool.wait_for_availability.override(
            task_id="wait_for_recovery"
        )(node_pool=node_pool_info, availability=True)

        chain(
            update_node_pool_label,
            wait_for_unavailable,
            wait_node_pool_recovered,
        )

      with TaskGroupWithTimeout(
          group_id="post_test",
          timeout=POST_TEST_TIMEOUT,
          is_teardown=True,
      ) as post_test:
        cleanup_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
        )(node_pool=node_pool_info)

      chain(
          pre_test,
          test,
          post_test,
      )
