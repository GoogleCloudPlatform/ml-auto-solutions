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

"""A DAG to test jobset uptime metric."""

import datetime

from airflow import models
from airflow.decorators import task
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
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout
from dags.common.task_group_with_timeout import TaskGroupWithTimeout


DAG_ID = "jobset_uptime_validation"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

PRE_TEST_TIMEOUT = datetime.timedelta(minutes=10)
POST_TEST_TIMEOUT = datetime.timedelta(minutes=10)
TEST_TIMEOUT = DAGRUN_TIMEOUT - PRE_TEST_TIMEOUT - POST_TEST_TIMEOUT


@task
def get_current_time() -> TimeUtil:
  """Get the current time in UTC."""
  return TimeUtil.now()


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "uptime",
        "tpu-observability",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the jobset uptime metric by deploying a workload on a "
        "TPU v6e-16 node pool and verifying that "
        "the metric behaves as expected."
    ),
    doc_md="""
      # JobSet Uptime Metric Test Using TPU v6e-16 Node Pool

      ### Description
      This DAG automates the process of creating a TPU v6e-16 node pool, launching
      a jobset, and monitoring the jobset uptime metric to ensure it behaves
      correctly. It also includes a negative test case to verify metric behavior
      over invalid time ranges. Finally, the DAG cleans up all created resources.

      ### Prerequisites
      This test requires an existing GKE cluster with TPU v6e-16 quota.

      ### Procedures
      1. **Provisioning**: Creates a TPU v6e-16 node pool with a specified reservation.
      2. **Deployment**: Applies a JobSet workload and waits for Pods to become active.
      3. **Metric Validation**: Polls the jobset uptime metric to confirm
         it behaves as expected.
      4. **Negative Testing**: Attempts to verify uptime against a current (future)
         timestamp to ensure the sensor correctly handles out-of-bounds queries.
      5. **Cleanup**: Deletes both the JobSet workload and the node pool to prevent
         resource leakage.
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

        wait_for_jobset_uptime_data = (
            jobset.wait_for_jobset_uptime_data.override(
                task_id="wait_for_jobset_uptime_data"
            )(
                node_pool=cluster_info,
                jobset_name=jobset_name,
                jobset_apply_time=startup.jobset_start_time,
            )
        )

        clean_up_workload = jobset.end_workload.override(
            task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
        )(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            jobset_name=jobset_name,
        ).as_teardown(
            setups=startup.jobset_start_time
        )

        jobset_clear_time = get_current_time.override(
            task_id="get_current_time"
        )()

        ensure_no_jobset_uptime_data = (
            jobset.ensure_no_jobset_uptime_data.override(
                task_id="ensure_no_jobset_uptime_data"
            )
        )(
            node_pool=cluster_info,
            jobset_name=jobset_name,
            jobset_clear_time=jobset_clear_time,
            # Wait 5 minutes to confirm no data has been detected.
            wait_time_seconds=300,
        )

        chain(
            *startup.tasks,
            wait_for_jobset_uptime_data,
            clean_up_workload,
            jobset_clear_time,
            ensure_no_jobset_uptime_data,
        )

      with TaskGroupWithTimeout(
          group_id="post_test",
          timeout=POST_TEST_TIMEOUT,
          is_teardown=True,
      ) as post_test:
        cleanup_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
        )(node_pool=cluster_info).as_teardown(
            setups=create_node_pool,
        )

      chain(
          selector,
          jobset_name,
          pre_test,
          test,
          post_test,
      )
