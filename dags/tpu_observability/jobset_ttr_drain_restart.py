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

"""A DAG to test jobset time-to-recover metric after a node-pool drained."""

import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from airflow.decorators import task

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.node_pool_util import Info
from dags.tpu_observability.utils.node_pool_util import NodeOperationSpec
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout


DAG_ID = "jobset_ttr_drain_restart"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)


@task
def check_nodes_number(
    pool: Info,
    drained_node_number: int,
) -> bool:
  """Checks whether the current node number match the expected number after
  node draining.

  Args:
    pool: An instance of the Info class that encapsulates the
    configuration and metadata of a GKE node pool.
    drained_node_number: The number of nodes expected to be drained.

  Returns:
    A boolean indicating whether the current node number matches the
      expected number after draining.

  """
  original_number = pool.num_nodes
  command = (
      "kubectl get nodes -l"
      f"cloud.google.com/gke-nodepool={pool.node_pool_name}"
      " --field-selector spec.unschedulable!=true --no-headers | wc -l"
  )
  stdout = subprocess.run_exec(command)
  current_number = int(stdout.strip())
  return current_number == original_number - drained_node_number


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 8, 10),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu-observability",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests using a node drain to interrupt a jobset, then "
        "verifies if the jobset restarts and polls the time-to-recover-"
        "metric to check if it is updated."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Node Drain

      ### Description
      This DAG automates the process of creating a node-pool and launching a JobSet.
      It then performs a node drain on a node where the JobSet is running to
      trigger an interruption. The DAG monitors if the JobSet successfully
      restarts and verifies if the JobSet TTR (Time-To-Recover) metric is
      properly updated. Finally, the DAG cleans up the JobSet and node-pool.

      ### Prerequisites
      This test requires an existing GKE cluster to run.

      ### Procedures
      1. Setup: A dedicated node-pool is created to host the JobSet.
      2. Launch: A JobSet YAML is deployed and given time to reach a 'Running' state.
      3. Interruption (Drain): The DAG identifies a node hosting a JobSet Pod and
         executes a `kubectl drain` command to evict the Pod and trigger a restart.
      4. Verification: A sensor runs to detect if the JobSet has recovered and if the
         time-to-recover metric has been updated in the monitoring system.
         Success is determined by the metric update; otherwise, it will timeout and fail.
      5. Cleanup: The JobSet is deleted and the node-pool is torn down.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value
    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      selector = jobset.generate_node_pool_selector(DAG_ID)

      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name=DAG_ID,
          node_pool_selector=selector,
      )

      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml(
          gcs_path=GCS_CONFIG_PATH,
          dag_name=DAG_ID,
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
          node_pool_selector=selector,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
      )

      startup = jobset.create_jobset_startup_tasks(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      select_node = node_pool.draw_random_node.override(task_id="select_node")(
          node_pool=cluster_info
      )

      drained_node = node_pool.operate_node.override(task_id="drained_node")(
          node_pool=cluster_info,
          operation=NodeOperationSpec.Drain(),
          node_name=select_node,
      )

      check_nodes_number = check_nodes_number.override(
          task_id="check_nodes_number"
      )(
          pool=cluster_info,
          drained_node_number=1,
      )

      uncordon_node = node_pool.operate_node.override(task_id="uncordon_node")(
          node_pool=cluster_info,
          operation=NodeOperationSpec.Uncordon(),
          node_name=select_node,
      )

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_metric_upload"
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      ).as_teardown(
          setups=startup.jobset_start_time
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          selector,
          create_node_pool,
          *startup.tasks,
          select_node,
          drained_node,
          check_nodes_number,
          uncordon_node,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
