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

from dataclasses import replace
import datetime
import logging
import os
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env

from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.tpu_observability.tpu_info_metric import ALL_METRIC_STRATEGIES
from dags.tpu_observability.tpu_info_metric import BaseMetricStrategy
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.node_pool_util import Info
from dags.tpu_observability.utils.time_util import TimeUtil
from dags.tpu_observability.utils.jobset_util import Workload
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout


DAG_ID = "tpu_info_metrics_verification"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)


def compare_metric_values(
    cmd_values: list[float],
    monitoring_values: list[float],
    pod_name: str,
    metric_display_name: str,
    tolerance_percent: float,
):
  """Compares two lists of metric values and checks if they are within a tolerance range."""
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
def get_tpu_info_metric_from_pod(
    node_pool: node_pool.Info,
    pod_name: str,
    jobset_config: jobset,
    metric_name: str,
) -> str:
  """Executes the 'tpu-info' command in the specified pod and returns its output."""
  with tempfile.TemporaryDirectory() as tmpdir:
    kube_dir = tmpdir + "/kubeconfig"
    env = os.environ.copy()
    env["KUBECONFIG"] = kube_dir

    cmd = " && ".join([
        jobset.Command.get_credentials_command(node_pool),
        (
            f"kubectl --kubeconfig={kube_dir} "
            f"exec {pod_name} -n {jobset_config.namespace} "
            f"-- tpu-info --metric {metric_name}"
        ),
    ])

    return subprocess.run_exec(cmd=cmd, env=env)


@task
def run_metric_verification(
    node_pool: Info,
    job_apply_time: TimeUtil,
    metric_strategy: BaseMetricStrategy,
    comparison_data: tuple[str, list[tpu_info.Table]],
):
  """A generic task that uses a strategy object to verify a metric."""
  pod_name, tpu_info_output = comparison_data
  metric_name = metric_strategy.metric_name
  logging.info("Verifying metric '%s' for pod: %s...", metric_name, pod_name)

  start_time = job_apply_time
  end_time = job_apply_time + datetime.timedelta(minutes=10)

  time_series_data = metric_strategy.list_or_query_metric(
      project_id=node_pool.project_id,
      cluster_name=node_pool.cluster_name,
      pod_name=pod_name,
      start_time=start_time,
      end_time=end_time,
  )

  monitoring_values = metric_strategy.parse_from_monitoring(time_series_data)
  cmd_values = metric_strategy.parse_from_tpu_info(tpu_info_output)

  tolerance_for_metric = metric_strategy.tolerance_percent
  logging.info(
      "Using a tolerance of %.2f%% for metric '%s' comparison.",
      tolerance_for_metric,
      metric_strategy.dag_id_suffix,
  )

  compare_metric_values(
      cmd_values,
      monitoring_values,
      pod_name,
      metric_display_name=metric_strategy.dag_id_suffix,
      tolerance_percent=tolerance_for_metric,
  )

  return True


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
    schedule=SCHEDULE,
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

    @task
    def generate_second_node_pool_name(
        node_pool_info: node_pool.Info,
    ) -> str:
      """Generates a second node pool name."""
      return f"{node_pool_info.node_pool_name}-2"

    workload_script = jobset.Workload.JAX_TPU_BENCHMARK

    with TaskGroup(group_id=f"v{config.tpu_version.value}"):
      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name="tpu_info_metrics_verification",
      )

      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="tpu_info_metrics_verification",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      cluster_info_2 = node_pool.copy_node_pool_info_with_override(
          info=cluster_info,
          node_pool_name=generate_second_node_pool_name(cluster_info),
      )

      with TaskGroup(group_id="create_node_pool") as create_node_pool:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1",
        )(
            node_pool=cluster_info,
        )

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
        )(
            node_pool=cluster_info_2,
        )

        _ = [create_first_node_pool, create_second_node_pool]

      apply_time = jobset.run_workload(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      pod_names = jobset.list_pod_names.override(
          task_id="list_pod_names",
          retries=5,
          retry_delay=datetime.timedelta(seconds=10),
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=pod_names, job_apply_time=apply_time)

      verification_results = {}
      all_verification_groups = []

      for strategy in ALL_METRIC_STRATEGIES:
        group_id = f"verify_{strategy.dag_id_suffix}"

        with TaskGroup(group_id=group_id) as verification_group:
          tpu_info_metric_outputs = (
              get_tpu_info_metric_from_pod.override(
                  task_id="get_tpu_info_metric_table"
              )
              .partial(
                  node_pool=cluster_info,
                  jobset_config=jobset_config,
                  metric_name=strategy.tpu_info_metric_name,
              )
              .expand(pod_name=pod_names)
          )

          tpu_info_metric_output = (
              tpu_info.parse_tpu_info_output.override(
                  task_id="get_each_metric_table"
              )
              .partial()
              .expand(output=tpu_info_metric_outputs)
          )

          verify_metric = (
              run_metric_verification.override(task_id="run_verification")
              .partial(
                  node_pool=cluster_info,
                  job_apply_time=apply_time,
                  metric_strategy=strategy,
              )
              .expand(comparison_data=pod_names.zip(tpu_info_metric_output))
          )

        all_verification_groups.append(verification_group)

        verification_results[strategy.dag_id_suffix] = verify_metric

      summary = summarize_results.override(
          task_id="summarize_results", trigger_rule=TriggerRule.ALL_DONE
      )(
          verification_results_dict=verification_results,
          active_pods=pod_names,
      )

      clean_up_workload = jobset.end_workload.override(
          task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      ).as_teardown(
          setups=apply_time
      )

      with TaskGroup(group_id="cleanup_node_pool") as cleanup_node_pool:
        cleanup_first_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_1",
            trigger_rule=TriggerRule.ALL_DONE,
        )(node_pool=cluster_info).as_teardown(
            setups=create_node_pool,
        )

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
        )(node_pool=cluster_info_2).as_teardown(
            setups=create_node_pool,
        )

        chain(cleanup_first_node_pool, cleanup_second_node_pool)

      chain(
          jobset_config,
          cluster_info,
          cluster_info_2,
          create_node_pool,
          apply_time,
          pod_names,
          wait_for_job_start,
          all_verification_groups,
          summary,
          clean_up_workload,
          cleanup_node_pool,
      )
