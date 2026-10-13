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
This script uses a factory pattern to dynamically generate an Airflow DAG for
each metric verification strategy.
"""

import copy
import datetime
import logging
import os
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException, AirflowFailException
from airflow.models.baseoperator import chain
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
from dags.tpu_observability.tpu_info_metric import (
    ALL_METRIC_STRATEGIES,
    BaseMetricStrategy,
)
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.utils.node_pool_util import Info
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.jobset_util import Workload
from dags.tpu_observability.utils.tpu_info_util import parse_tpu_info_output
from dags.common.task_group_with_timeout import TaskGroupWithTimeout
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout
from xlml.utils import subprocess_utils as subprocess

DAG_ID = "tpu_info_metrics_verification"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

PRE_TEST_TIMEOUT = datetime.timedelta(minutes=10)
POST_TEST_TIMEOUT = datetime.timedelta(minutes=10)
TEST_TIMEOUT = DAGRUN_TIMEOUT - PRE_TEST_TIMEOUT - POST_TEST_TIMEOUT


def compare_metric_values(
    cmd_values: list[float],
    monitoring_values: list[float],
    pod_name: str,
    metric_display_name: str,
    tolerance_percent: float,
):
  """Compares two lists of metric values and checks if they are within a
  tolerance range."""
  if len(cmd_values) != len(monitoring_values):
    raise AirflowException(
        f"For pod {pod_name} ({metric_display_name}), data count mismatch. "
        f"TPU-Info has {len(cmd_values)} values, Monitoring has "
        f"{len(monitoring_values)}."
    )

  logging.info(
      "--- Comparison Results for pod: %s, Metric: %s ---",
      pod_name,
      metric_display_name,
  )
  logging.info(
      "%-12s%-15s%-17s%-12s%-15s%-10s",
      "Device",
      "TPU-Info Val",
      "Monitoring Val",
      "Difference",
      "Allowed Diff",
      "Result",
  )
  logging.info("-" * 85)

  all_passed = True
  for i, (log_val, mon_val) in enumerate(zip(cmd_values, monitoring_values)):
    diff = abs(log_val - mon_val)
    allowed_diff = mon_val * (tolerance_percent / 100.0)
    passed = diff <= allowed_diff
    if not passed:
      all_passed = False
    logging.info(
        "%-12s%-15.2f%-17.2f%-12.2f%-15.2f%-10s",
        f"Device {i}",
        log_val,
        mon_val,
        diff,
        allowed_diff,
        "PASS" if passed else "FAIL",
    )
  logging.info("-" * 70)

  if not all_passed:
    raise AirflowException(
        f"Overall Result for Pod {pod_name} ({metric_display_name}): FAIL - "
        f"Values do not match within {tolerance_percent}% tolerance."
    )
  logging.info(
      "Overall Result for Pod %s (%s): PASS", pod_name, metric_display_name
  )


@task
def verify_metric_for_all_pods(
    node_pool: Info,
    jobset_config: jobset.JobSet,
    job_apply_time: TimeUtil,
    metric_strategy: BaseMetricStrategy,
    pod_names: list[str],
) -> list[bool]:
  """Runs metric verification across all pods sequentially."""

  results = []
  failed_pods = []
  for pod_name in pod_names:
    logging.info(
        "Fetching tpu-info metric '%s' for pod: %s...",
        metric_strategy.tpu_info_metric_name,
        pod_name,
    )

    try:
      with tempfile.TemporaryDirectory() as tmpdir:
        kube_dir = tmpdir + "/kubeconfig"
        env = os.environ.copy()
        env["KUBECONFIG"] = kube_dir

        cmd = " && ".join([
            jobset.Command.get_credentials_command(node_pool),
            (
                f"kubectl --kubeconfig={kube_dir} "
                f"exec {pod_name} -n {jobset_config.namespace} "
                f"-- tpu-info --metric {metric_strategy.tpu_info_metric_name}"
            ),
        ])

        output = subprocess.run_exec(cmd=cmd, env=env)

      tpu_info_output = parse_tpu_info_output(output)

      logging.info(
          "Verifying metric '%s' for pod: %s...",
          metric_strategy.metric_name,
          pod_name,
      )

      start_time = job_apply_time
      end_time = job_apply_time + datetime.timedelta(minutes=10)

      time_series_data = metric_strategy.list_or_query_metric(
          project_id=node_pool.project_id,
          cluster_name=node_pool.cluster_name,
          pod_name=pod_name,
          start_time=start_time,
          end_time=end_time,
      )

      monitoring_values = metric_strategy.parse_from_monitoring(
          time_series_data
      )
      cmd_values = metric_strategy.parse_from_tpu_info(tpu_info_output)

      tolerance_for_metric = metric_strategy.tolerance_percent
      logging.info(
          "Using a tolerance of %.2f%% for metric '%s' comparison on pod %s.",
          tolerance_for_metric,
          metric_strategy.dag_id_suffix,
          pod_name,
      )

      compare_metric_values(
          cmd_values,
          monitoring_values,
          pod_name,
          metric_display_name=metric_strategy.dag_id_suffix,
          tolerance_percent=tolerance_for_metric,
      )
      results.append(True)
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

  return results


@task
def summarize_results(
    verification_results_dict: dict[str, list[bool]], active_pods: list[str]
):
  """
  Summarizes the results of metric verifications for all pods.

  """
  if not active_pods:
    raise AirflowException("No active nodes were found. Grand Result: SKIPPED")

  num_expected_pods = len(active_pods)
  overall_success = True
  failure_summary = []

  logging.info("--- Overall Verification Summary ---")
  logging.info("Total pods scheduled for verification: %s", num_expected_pods)
  logging.info("-" * 70)
  logging.info("%-35s | %-10s | %-20s", "Metric Name", "Result", "Details")
  logging.info("-" * 70)

  for metric_name, results in verification_results_dict.items():
    num_passes = len(results)  # Only successed task return result

    if num_passes < num_expected_pods:
      status = "FAIL"
      details = f"Passed {num_passes} of {num_expected_pods} pods."
      overall_success = False
      failure_summary.append(f"- {metric_name}: {details}")
    else:
      status = "PASS"
      details = f"All {num_expected_pods} pods passed."

    logging.info("%-35s | %-10s | %-20s", metric_name, status, details)

  logging.info("-" * 70)

  if not overall_success:
    error_message = (
        "Grand Result: FAILURE - One or more metric verifications failed.\n"
        "Failure Details:\n" + "\n".join(failure_summary)
    )
    raise AirflowException(error_message)

  logging.info(
      "Grand Result: SUCCESS - All metric verifications passed for all pods."
  )


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info", "TPU", "v6e-16"],
    description=(
        "Validates TPU metric consistency between tpu-info CLI and Cloud "
        "Monitoring API across dynamically provisioned GKE node pools."
    ),
    doc_md="""
      # Data Validation DAG
      # This DAG verifies the data value of the tables in the tpu-info output.

      ### Description
      This DAG automates the cross-validation of TPU performance data. It
      ensures that the metrics reported directly from the hardware are
      consistent with the data ingested into the cloud monitoring pipeline.
      The verification suite includes metrics such as **TPU Utilization,
      TensorCore Activity, Memory Usage, and Latency**, and is designed to
      automatically scale as new metric strategies are added to the validation
      library.

      ### Prerequisites
      This test requires an existing GKE cluster.
      A pre-built Docker image containing the necessary jax, libtpu, and
      tpu-info packages must also be available in a repository accessible
      by the GKE cluster.

      ### Procedures
      The DAG begins by creating temporary GKE TPU node pools for the test.
      Once the node pools are running, it schedules a Kubernetes JobSet and
      waits for the pods to become active. It then executes the tpu-info
      command within these pods to capture the raw text output. This output is
      parsed into structured tables, performs a point-by-point comparison
      between the two data sources. A task is marked as successful only if the
      values fall within a predefined **tolerance percentage**. the DAG cleans
      up all created resources, including the JobSet and the temporary node
      pools.
    """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
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

      cluster_info_2 = copy.deepcopy(cluster_info)
      cluster_info_2.node_pool_name = f"{cluster_info.node_pool_name}-2"

      with TaskGroupWithTimeout(
          group_id="pre_test",
          timeout=PRE_TEST_TIMEOUT,
      ) as pre_test:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1",
        )(
            node_pool=cluster_info,
            node_pool_selector=selector,
        ).as_setup()

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
        )(
            node_pool=cluster_info_2,
            node_pool_selector=selector,
        ).as_setup()

        startup = jobset.create_jobset_startup_tasks(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            jobset_name=jobset_name,
            node_pool_selector=selector,
            workload_type=Workload.JAX_TPU_BENCHMARK,
        )
        chain([create_first_node_pool, create_second_node_pool], *startup.tasks)

      with TaskGroupWithTimeout(
          group_id="test",
          timeout=TEST_TIMEOUT,
      ) as test:
        verification_results = {}
        all_verification_tasks = []

        for strategy in ALL_METRIC_STRATEGIES:
          verify_metric = verify_metric_for_all_pods.override(
              task_id=f"verify_{strategy.dag_id_suffix}"
          )(
              node_pool=cluster_info,
              jobset_config=jobset_config,
              job_apply_time=startup.jobset_start_time,
              metric_strategy=strategy,
              pod_names=startup.running_pods,
          )

          all_verification_tasks.append(verify_metric)
          verification_results[strategy.dag_id_suffix] = verify_metric

        summary = summarize_results.override(
            task_id="summarize_results", trigger_rule=TriggerRule.ALL_DONE
        )(
            verification_results_dict=verification_results,
            active_pods=startup.running_pods,
        )
        chain(all_verification_tasks, summary)

      with TaskGroupWithTimeout(
          group_id="post_test",
          timeout=POST_TEST_TIMEOUT,
          is_teardown=True,
      ) as post_test:
        clean_up_workload = jobset.end_workload.override(
            task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
        )(
            node_pool=cluster_info,
            jobset_config=jobset_config,
            jobset_name=jobset_name,
        )

        cleanup_first_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_1",
            trigger_rule=TriggerRule.ALL_DONE,
        )(node_pool=cluster_info)

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
        )(node_pool=cluster_info_2)

        chain(
            clean_up_workload, cleanup_first_node_pool, cleanup_second_node_pool
        )

      chain(pre_test, test, post_test)
