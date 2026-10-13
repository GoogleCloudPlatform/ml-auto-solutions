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

import copy
import datetime
import os
import re
import tempfile
import logging

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup
from dags.common.task_group_with_timeout import TaskGroupWithTimeout
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.common.scheduling_helper.scheduling_helper import (
    SchedulingHelper,
    get_dag_timeout,
)
from dags.tpu_observability.configs.common import (
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
    MachineConfigMap,
    TpuConfig,
)
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.tpu_info_util import parse_tpu_info_output
from dags.tpu_observability.utils.jobset_util import Workload
from xlml.utils import subprocess_utils as subprocess
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout

DAG_ID = "tpu_info_format_validation_dag"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)

PRE_TEST_TIMEOUT = datetime.timedelta(minutes=20)
POST_TEST_TIMEOUT = datetime.timedelta(minutes=10)
TEST_TIMEOUT = DAGRUN_TIMEOUT - PRE_TEST_TIMEOUT - POST_TEST_TIMEOUT
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)


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
def validate_tpu_info_format(
    info: node_pool.Info,
    tpu_config: TpuConfig,
    pod_names: list[str],
) -> None:
  """Executes tpu-info command and runs all validations sequentially across all pods."""

  failed_pods = []
  for pod_name in pod_names:
    logging.info(
        "Executing tpu-info and performing format validation for pod: %s",
        pod_name,
    )

    try:
      with tempfile.NamedTemporaryFile() as temp_config_file:
        env = os.environ.copy()
        env["KUBECONFIG"] = temp_config_file.name

        cmd = " && ".join([
            jobset.Command.get_credentials_command(info),
            f"kubectl exec {pod_name} -n default -- tpu-info",
        ])

        raw_output = subprocess.run_exec(cmd, env=env)

      tpu_info_output = parse_tpu_info_output(raw_output)

      verify_table_amount(tpu_info_output)
      validate_chips_table(tpu_info_output, tpu_config)
      validate_runtime_table(tpu_info_output)
      validate_tensorcore_table(tpu_info_output)
      validate_latency_table(tpu_info_output)
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
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 8, 15),
    default_args={
        "retries": 0,
        "owner": test_owner.YUNA_T,
    },
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
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

      cluster_info_2 = copy.deepcopy(cluster_info)
      cluster_info_2.node_pool_name = f"{cluster_info.node_pool_name}-2"

      with TaskGroupWithTimeout(
          group_id="pre_test",
          timeout=PRE_TEST_TIMEOUT,
      ) as pre_test:
        create_first_node_pool = node_pool.create.override(
            task_id="node_pool_1",
            retries=2,
        )(
            node_pool=cluster_info,
            node_pool_selector=selector,
        ).as_setup()

        create_second_node_pool = node_pool.create.override(
            task_id="node_pool_2",
            retries=2,
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
        validate_format = validate_tpu_info_format.override(
            task_id="validate_tpu_info_format"
        )(
            info=cluster_info,
            tpu_config=config,
            pod_names=startup.running_pods,
        )

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
            retries=2,
        )(node_pool=cluster_info)

        cleanup_second_node_pool = node_pool.delete.override(
            task_id="cleanup_node_pool_2",
            trigger_rule=TriggerRule.ALL_DONE,
            retries=2,
        )(node_pool=cluster_info_2)

        chain(
            clean_up_workload, cleanup_first_node_pool, cleanup_second_node_pool
        )

      chain(
          selector,
          jobset_name,
          pre_test,
          test,
          post_test,
      )
