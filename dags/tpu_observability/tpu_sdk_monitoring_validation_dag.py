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

"""A DAG to validate the `tpumonitoring` SDK, ensuring help() and
list_supported_metrics() are functional inside TPU worker pods."""

import datetime
import logging

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common.task_group_with_timeout import TaskGroupWithTimeout
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
from dags.tpu_observability.utils import tpu_monitoring_sdk_util as sdk
from dags.tpu_observability.utils.jobset_util import Workload

DAG_ID = "tpu_sdk_monitoring_validation"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

PRE_TEST_TIMEOUT = datetime.timedelta(minutes=10)
POST_TEST_TIMEOUT = datetime.timedelta(minutes=10)
TEST_TIMEOUT = DAGRUN_TIMEOUT - PRE_TEST_TIMEOUT - POST_TEST_TIMEOUT


@task
def validate_monitoring_sdk(info: node_pool.Info, pod_names: list[str]) -> None:
  """Validates the tpumonitoring SDK functions inside TPU worker pods.

  This task executes both help() and list_supported_metrics() via the SDK
  and verifies that the output contains the expected strings and patterns.

  Args:
    info: Cluster info for gcloud credentials.
    pod_names: List of pod names.
  """
  # A dict of script to its expected result patterns.
  validate_spec: dict[sdk.TpuMonitoringScript, list[str]] = {
      # Validates help() output. Expected format:
      # - list_supported_metrics(): List all supported functionality...
      # - get_metric(metric_name:str): Get specific metric...
      # - snapshot mode: Enable real-time monitoring...
      sdk.TpuMonitoringScript.HELP: [
          "list_supported_metrics()",
          "get_metric(metric_name:str)",
          "snapshot mode",
      ],
      # Validates list_supported_metrics() output. Expected format:
      # ['tensorcore_util', 'duty_cycle_pct', 'hbm_capacity_usage', ...]
      sdk.TpuMonitoringScript.LIST_SUPPORTED_METRICS: [
          "tensorcore_util",
          "duty_cycle_pct",
          "hbm_capacity_usage",
          "buffer_transfer_latency",
          "hlo_execution_timing",
      ],
  }

  failed_pods = []
  for pod_name in pod_names:
    logging.info("Validating tpumonitoring SDK for pod: %s", pod_name)
    try:
      for script, patterns in validate_spec.items():
        output = sdk.execute_sdk_command(info, pod_name, script)
        for pattern in patterns:
          if pattern not in output:
            raise AssertionError(
                f"Validation failed for 'tpumonitoring.{script.name.lower()}()' inside pod '{pod_name}': "
                f"Missing '{pattern}'."
            )
      logging.info("Validation succeeded for pod: %s", pod_name)
    except Exception as e:
      logging.error("Validation failed for pod: %s. Error: %s", pod_name, e)
      failed_pods.append((pod_name, e))

  if failed_pods:
    failed_pod_names = [name for name, _ in failed_pods]
    logging.error("The following pods failed validation: %s", failed_pod_names)
    raise AirflowFailException(
        f"Task failed because validation failed on pods: {failed_pod_names}"
    )


with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 1, 13),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "tpu-observability",
        "TPU",
        "v6e-16",
        "tpu-monitoring-sdk",
    ],
    description=(
        "Validates tpumonitoring SDK: help() and "
        "list_supported_metrics() inside TPU worker pods."
    ),
    doc_md="""
        ### Description
        This DAG performs an end-to-end validation of the `tpumonitoring` Python SDK
        within TPU worker pods. It ensures the SDK is correctly installed and its
        monitoring functions are accessible via `libtpu.sdk`.

        ### Validation Steps:
        1. **SDK Help Documentation Validation**:
           Executes `tpumonitoring.help()` to verify that the API documentation is
           correctly rendered and includes essential methods like `list_supported_metrics`.

        2. **Metric Catalog Validation**:
           Executes `tpumonitoring.list_supported_metrics()` and verifies that
           core TPU metrics (e.g., `tensorcore_util`, `hbm_capacity_usage`, `ici_link_health`)
           are present in the returned list.

        3. **Environment Integrity Check**:
           Ensures the `libtpu` library can correctly interface with the TPU driver
           and hardware devices inside the container.
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
        ).as_setup()

        startup = jobset.create_jobset_startup_tasks(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            jobset_name=jobset_name,
            node_pool_selector=selector,
            workload_type=Workload.JAX_TPU_BENCHMARK,
        )
        chain(create_node_pool, *startup.tasks)

      with TaskGroupWithTimeout(
          group_id="test",
          timeout=TEST_TIMEOUT,
      ) as test:
        sdk_validation = validate_monitoring_sdk.override(
            task_id="sdk_validation"
        )(
            info=cluster_info,
            pod_names=startup.running_pods,
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
        )

        cleanup_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
        )(node_pool=cluster_info)
        chain(cleanup_workload, cleanup_node_pool)

      chain(pre_test, test, post_test)
