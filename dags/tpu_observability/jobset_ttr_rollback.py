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

"""A DAG to test jobset time-to-recover metric using a node-pool rollback."""

import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common.scheduling_helper.scheduling_helper import (
    SchedulingHelper,
    get_dag_timeout,
)
from dags.tpu_observability.configs.common import (
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
    MachineConfigMap,
)
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout
from dags.common.task_group_with_timeout import TaskGroupWithTimeout


DAG_ID = "jobset_rollback_ttr"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

PRE_TEST_TIMEOUT = datetime.timedelta(minutes=10)
POST_TEST_TIMEOUT = datetime.timedelta(minutes=10)
TEST_TIMEOUT = DAGRUN_TIMEOUT - PRE_TEST_TIMEOUT - POST_TEST_TIMEOUT

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
        "rollback",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the use of a node-pool rollback to interrupt a "
        "jobset, then polls the jobset time-to-recover metric to check "
        "if it is updated."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Node-Pool Rollback

      ### Description
      This DAG automates the process of creating a node-pool, launching a jobset
      then using a node-pool rollback to interrupt the node-pool, and afterwards
      monitors if the jobset TTR metric gets updated. Finally the DAG cleans up
      the jobset and node-pool which were created.

      ### Prerequisites
      This test requires an existing cluster to run.

      ### Procedures
      First the node-pool is created, a jobset yaml is then launched on the
      cluster and given a short period of time to initialize. After this a
      rollback is run on the previously created node-pool to interrupt it.
      A sensor is finally run which will either detect that the jobset
      time-to-recover metric has been updated, resulting in a success, or
      timeout, and fail.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml(
          gcs_path=GCS_CONFIG_PATH,
          dag_name=DAG_ID,
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name=DAG_ID,
      )

      selector = jobset.generate_node_pool_selector(DAG_ID)
      jobset_name = jobset.generate_jobset_name(jobset_config.dag_id_prefix)

      with TaskGroupWithTimeout(
          group_id="pre_test",
          timeout=PRE_TEST_TIMEOUT,
      ) as pre_test:
        create_node_pool = node_pool.create.override(
            task_id="create_node_pool"
        )(
            node_pool=cluster_info,
            node_pool_selector=selector,
        )

      with TaskGroupWithTimeout(
          group_id="test",
          timeout=TEST_TIMEOUT,
      ) as test:
        startup = jobset.create_jobset_startup_tasks(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            jobset_name=jobset_name,
            node_pool_selector=selector,
            workload_type=Workload.JAX_TPU_BENCHMARK,
        )

        rollback_node_pool = node_pool.rollback.override(
            task_id="rollback_node_pool"
        )(node_pool=cluster_info)

        wait_for_recovery = jobset.wait_for_jobset_recovered.override(
            task_id="wait_for_recovery"
        )(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            jobset_name=jobset_name,
        )

        verify_duration = jobset.verify_recovery_duration.override(
            task_id="verify_recovery_duration"
        )(
            start_time=rollback_node_pool,
            end_time=wait_for_recovery,
        )

        wait_for_metric_upload = (
            jobset.wait_for_jobset_ttr_to_be_found.override(
                task_id="wait_for_jobset_ttr_to_be_found",
            )(
                node_pool=cluster_info,
                jobset_name=jobset_name,
                start_time=rollback_node_pool,
            )
        )

        chain(
            *startup.tasks,
            rollback_node_pool,
            wait_for_recovery,
            verify_duration,
            wait_for_metric_upload,
        )

      with TaskGroupWithTimeout(
          group_id="post_test",
          timeout=POST_TEST_TIMEOUT,
          is_teardown=True,
      ) as post_test:
        cleanup_workload = jobset.end_workload.override(
            task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
        )(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            jobset_name=jobset_name,
        ).as_teardown(
            setups=startup.jobset_start_time
        )

        cleanup_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
        )(node_pool=cluster_info).as_teardown(
            setups=create_node_pool,
        )

        chain(
            cleanup_workload,
            cleanup_node_pool,
        )

      chain(
          selector,
          jobset_name,
          pre_test,
          test,
          post_test,
      )
