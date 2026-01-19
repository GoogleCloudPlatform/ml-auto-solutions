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
A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

import datetime
import os
import re
import subprocess
import tempfile
from dataclasses import replace

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    TpuConfig,
    GCS_CONFIG_PATH,
)
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.jobset_util import JobSet, Workload


@task
def get_tpu_info_from_pod(info: node_pool.Info, pod_name: str) -> str:
  """
  Executes the `tpu-info` command in a specified pod and returns its output.

  This task uses kubectl to run the 'tpu-info' command inside the given pod
  in the 'default' namespace. The output of the command is captured and
  returned.

  Args:
    kubeconfig: The path to the kubeconfig file.
    pod_name: The name of the pod to execute the command in.

  Returns:
    The standard output from the 'tpu-info' command.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- tpu-info",
    ])

    return subprocess.run_exec(cmd, env=env)


@task
def verify_table_amount(tpu_info_output: list[tpu_info.Table]):
  """
  Verifies if all expected tables are present.
  """
  expect_table_names = {
      "TPU Chips",
      "TPU Runtime Utilization",
      "TensorCore Utilization",
      "TPU Buffer Transfer Latency",
  }

  found_names = {table.name for table in tpu_info_output}

  missing_names = expect_table_names - found_names

  if missing_names:
    raise AirflowFailException(
        "Mismatched tpu-info tables; "
        f"required: {expect_table_names}; got: {found_names}"
    )


@task
def validate_chips_table(
    tpu_info_output: list[tpu_info.Table],
    tpu_config: TpuConfig,
):
  """
  Validates the row count and content for the 'TPU Chips' table.
  """
  errors = []
  content = next(
      (table for table in tpu_info_output if table.name == "TPU Chips"),
      None,
  )

  expected_rows = 4
  if len(content.body) != expected_rows:
    raise AirflowFailException(
        f"Unexpected row count; except: {expected_rows}; got:"
        f" {len(content.body)}"
    )

  tpu_type = tpu_config.tpu_version.value

  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "Chip":
          if not re.match(r"/dev/vfio/\d+", data):
            errors.append(
                f"Unexpected {header}; except: '/dev/vfio/NNN'; got: '{data}'"
            )
        case "Type":
          if tpu_type not in data:
            errors.append(
                f"Unexpected {header}; except: string contains '{tpu_type}';"
                f" got: '{data}'"
            )
        case "PID":
          if not (data.isdigit() and int(data) > 0):
            errors.append(
                f"Unexpected {header}; except: a positive integer; got: "
                f"'{data}'"
            )

  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with {len(errors)} "
        f"error(s):\n{error_summary}\n\n"
        f"Raw table output:\n{content.raw_body}"
    )


@task
def validate_runtime_table(tpu_info_output: list[tpu_info.Table]):
  """
  Validates the row count and content of table 'TPU Runtime Utilization'
  """
  errors = []
  content = next(
      (
          table
          for table in tpu_info_output
          if table.name == "TPU Runtime Utilization"
      ),
      None,
  )

  expected_rows = 4
  if len(content.body) != expected_rows:
    raise AirflowFailException(
        f"Unexpected row count; except: {expected_rows}; got:"
        f" {len(content.body)}"
    )

  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "HBM Usage (GiB)":
          regex = re.match(r"(\d+\.\d+)\s*GiB\s*/\s*(\d+\.\d+)\s*GiB", data)
          if regex:
            used, total = float(regex.group(1)), float(regex.group(2))
            if used > total:
              errors.append(
                  f"Unexpected {header}; expect: 'used HBM <= total HBM'; got:"
                  f" '{used} GiB > {total} GiB'"
              )
          else:
            errors.append(
                f"Unexpected {header}; expect: 'N.NN GiB / N.NN GiB'; got:"
                f" '{data}'"
            )
        case "Duty cycle":
          duty_match = re.match(r"(\d+\.\d+)%", data)
          if not (duty_match and 0.0 <= float(duty_match.group(1)) <= 100.0):
            errors.append(
                f"Unexpected {header}; expect: 'a percentage between"
                f" 0.0-100.0'; got: '{data}'"
            )
  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with"
        f" {len(errors)} error(s):\n{error_summary}\n\nRaw table"
        f" output:\n{content.raw_body}"
    )


@task
def validate_tensorcore_table(tpu_info_output: list[tpu_info.Table]):
  """
  Validates the row count and content of table 'TensorCore Utilization'
  """
  errors = []
  content = next(
      (
          table
          for table in tpu_info_output
          if table.name == "TensorCore Utilization"
      ),
      None,
  )

  expected_rows = 4
  if len(content.body) != expected_rows:
    raise AirflowFailException(
        f"Unexpected row count; except: {expected_rows}; got:"
        f" {len(content.body)}"
    )
  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "TensorCore Utilization":
          util_match = re.match(r"(\d+\.\d+)%", data)
          if not (util_match and 0.0 < float(util_match.group(1)) <= 100.0):
            errors.append(
                f"Unexpected {header}; expect: 'a percentage > 0.0 and <="
                f" 100.0'; got: '{data}'"
            )
  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with"
        f" {len(errors)} error(s):\n{error_summary}\n\nRaw table"
        f" output:\n{content.raw_body}"
    )


@task
def validate_latency_table(tpu_info_output: list[tpu_info.Table]):
  """
  Validates the row count and content of table 'TPU Buffer Transfer Latency'
  """
  errors = []
  content = next(
      (
          table
          for table in tpu_info_output
          if table.name == "TPU Buffer Transfer Latency"
      ),
      None,
  )

  if content.body is None or len(content.body) == 0:
    raise AirflowFailException(
        "Unexpected row count; expects at least one data row; got: 0"
    )

  for row_dict in content.body:
    for header, data in row_dict.items():
      match header:
        case "Buffer Size":
          continue
        case "P50" | "P90" | "P95" | "P999":
          if not (data.endswith(" us") and float(data.replace(" us", "")) > 0):
            errors.append(
                f"Unexpected {header}; expect: 'a positive float ending in \""
                f" us\"'; got: '{data}'"
            )

  if errors:
    error_summary = "\n".join(errors)
    raise AirflowFailException(
        f"Validation failed for {content.name} table with"
        f" {len(errors)} error(s):\n{error_summary}\n\nRaw table"
        f" output:\n{content.raw_body}"
    )


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="tpu_info_format_validation_dag",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule="0 19 * * *" if composer_env.is_prod_env() else None,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info", "TPU", "v6e-16"],
    description=(
        "This DAG verifies the format of the tables in the tpu-info output "
        "using tpu-info CLI tool. It includes 4 tables: TPU Chips, TPU "
        "Runtime Utilization, TensorCore Utilization, and TPU Buffer Transfer "
        "Latency."
    ),
    doc_md="""
      # Format Validation DAG
      # This DAG verifies the format of the tables in the tpu-info output.

      ### Description
      This DAG automates the validation of the tpu-info command-line tool's
      output format.It verifies the structure and content of key metric tables,
      including "TPU Chips", "TPU Runtime Utilization", "TensorCore
      Utilization", and "TPU Buffer Transfer Latency", by running the tool on a
      live GKE cluster with TPU node pools.

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
      parsed into structured tables, and a series of validation tasks check
      each table for the correct structure, row counts, and data formats.
      Finally, regardless of the test outcome, the DAG cleans up all created
      resources, including the JobSet and the temporary node pools.
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

    jobset_config = JobSet(
        jobset_name="tpu-info-{{ ds_nodash }}-{{ ti.job_id }}",
        namespace="default",
        max_restarts=5,
        replicated_job_name="tpu-job-slice",
        replicas=2,
        backoff_limit=0,
        completions=4,
        parallelism=4,
        tpu_accelerator_type="tpu-v6e-slice",
        tpu_topology="4x4",
        container_name="jax-tpu-worker",
        image=(
            "asia-northeast1-docker.pkg.dev/cienet-cmcs/yuna-docker/"
            "tpu-info:v0.5.1"
        ),
        tpu_cores_per_pod=4,
    )

    workload_script = Workload.JAX_TPU_BENCHMARK

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="tpu_info_format_validation_dag",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      cluster_info_2 = node_pool.copy_node_pool_info_with_override(
          info=cluster_info,
          node_pool_name=generate_second_node_pool_name(cluster_info),
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="create_node_pool"
      ) as create_node_pool:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1",
            retries=2,
        )(
            node_pool=cluster_info,
        )

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
            retries=2,
        )(
            node_pool=cluster_info_2,
        )

      apply_time = jobset.run_workload.override(owner=test_owner.YUNA_T)(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=workload_script
          ),
          namespace=jobset_config.namespace,
      )

      pod_names = jobset.list_pod_names.override(task_id="list_pod_names")(
          node_pool=cluster_info,
          namespace=jobset_config.namespace,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=pod_names, job_apply_time=apply_time)

      outputs_of_tpu_info = (
          get_tpu_info_from_pod.override(task_id="get_tpu_info")
          .partial(info=cluster_info)
          .expand(pod_name=pod_names)
      )

      output_of_tpu_info = (
          tpu_info.parse_tpu_info_output.override(
              task_id="get_each_metric_table"
          )
          .partial()
          .expand(output=outputs_of_tpu_info)
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="verification_group"
      ) as verification_group:
        verify_table_amount_task = (
            verify_table_amount.override(task_id="verify_table_amount_task")
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_tpu_chips_metric = (
            validate_chips_table.override(task_id="validate_tpu_chips_metric")
            .partial(tpu_config=config)
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_runtime_metric = (
            validate_runtime_table.override(task_id="validate_runtime_metric")
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_tensorcore_metric = (
            validate_tensorcore_table.override(
                task_id="validate_tensorcore_metric"
            )
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

        validate_latency_metric = (
            validate_latency_table.override(task_id="validate_latency_metric")
            .partial()
            .expand(tpu_info_output=output_of_tpu_info)
        )

      clean_up_workload = jobset.end_workload.override(
          task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          namespace=jobset_config.namespace,
      ).as_teardown(
          setups=apply_time
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="cleanup_node_pool"
      ) as cleanup_node_pool:
        cleanup_first_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_1",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info).as_teardown(
            setups=create_node_pool,
        )

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info_2).as_teardown(
            setups=create_node_pool,
        )

      chain(
          verify_table_amount_task,
          [
              validate_tpu_chips_metric,
              validate_runtime_metric,
              validate_tensorcore_metric,
              validate_latency_metric,
          ],
      )

      [create_first_node_pool, create_second_node_pool]

      chain(cleanup_first_node_pool, cleanup_second_node_pool)

      chain(
          cluster_info,
          cluster_info_2,
          create_node_pool,
          apply_time,
          pod_names,
          wait_for_job_start,
          outputs_of_tpu_info,
          output_of_tpu_info,
          verification_group,
          clean_up_workload,
          cleanup_node_pool,
      )
