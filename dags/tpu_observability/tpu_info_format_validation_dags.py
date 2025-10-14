"""A DAG orchestrates the process of verifying TensorCore utilization metrics.

This is done by comparing data from Cloud Logging and Cloud Monitoring.
"""

from dataclasses import replace
import datetime
import logging
import os
import re
import subprocess

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags.common.vm_resource import MachineVersion
from dags.common.vm_resource import Project
from dags.common.vm_resource import Region
from dags.common.vm_resource import Zone
from dags.map_reproducibility.utils import constants
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import tpu_info_util as tpu_info
from dags.tpu_observability.utils.jobset_util import JobSet
from dags.tpu_observability.utils.jobset_util import Workload

MACHINE_TYPE_TO_TPU_VERSION = {
    "ct6e-standard-4t": "v6e",
    "ct5p-hightpu-4t": "v5p",
}


@task
def get_tpu_info_from_pod(kubeconfig: str, pod_name: str) -> str:
  """Executes the 'tpu-info' command within a specified pod and returns its output.

  This task uses kubectl to run the 'tpu-info' command inside the given pod
  in the 'default' namespace. The output of the command is captured and
  returned.

  Args:
    kubeconfig: The path to the kubeconfig file.
    pod_name: The name of the pod to execute the command in.

  Returns:
    The standard output from the 'tpu-info' command.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  result = subprocess.run(
      (
          f"kubectl --kubeconfig={kubeconfig} "
          f"exec {pod_name} -n default "
          f"-- tpu-info"
      ),
      shell=True,
      env=env,
      # Since tpu-info feature still has some issues, so the command will
      # inevitably throw an error. To avoid marking the task as failed,
      # I set check to False so that the task status does not show as failed.
      check=True,
      capture_output=True,
      text=True,
  )
  logging.info("STDOUT: %s", result.stdout)
  return result.stdout


@task
def verify_table_amount(tpu_info_output: list[tpu_info.Table]):
  """Verifies if all expected tables are present in the parsed tpu_info dictionary."""
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
    tpu_info_output: list[tpu_info.Table], node_pool: node_pool.Info
):
  """Validates the row count and content for the 'TPU Chips' table."""
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

  tpu_type = MACHINE_TYPE_TO_TPU_VERSION[node_pool.machine_type]

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
  """Validates the row count and content for the 'TPU Runtime Utilization' table."""
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
          hbm_match = re.match(r"(\d+\.\d+)\s*GiB\s*/\s*(\d+\.\d+)\s*GiB", data)
          if hbm_match:
            used, total = float(hbm_match.group(1)), float(hbm_match.group(2))
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
  """Validates the row count and content for the 'TensorCore Utilization' table."""
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
  """Validates the row count and content for the TPU Buffer Transfer Latency table."""
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


with models.DAG(
    dag_id="tpu_info_format_validation_dag",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=constants.Schedule.WEEKDAY_PDT_6AM_7AM_EXCEPT_THURSDAY,
    catchup=False,
    tags=["gke", "tpu-observability", "tpu-info"],
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
      including "TPU Chips", "TPU Runtime Utilization", "TensorCore Utilization",
      and "TPU Buffer Transfer Latency", by running the tool on a live GKE
      cluster with TPU node pools.

      ### Prerequisites
      This test requires an existing GKE cluster.
      A pre-built Docker image containing the necessary jax, libtpu, and
      tpu-info packages must also be available in a repository accessible
      by the GKE cluster.

      ### Procedures
      The DAG begins by creating temporary GKE TPU node pools for the test.
      Once the node pools are running, it schedules a Kubernetes JobSet and
      waits for the pods to become active. It then executes the tpu-info command
      within these pods to capture the raw text output. This output is parsed
      into structured tables, and a series of validation tasks check each table
      for the correct structure, row counts, and data formats. Finally,
      regardless of the test outcome, the DAG cleans up all created resources,
      including the JobSet and the temporary node pools.
      """,
) as dag:
  cluster_info = node_pool.Info(
      project_id=models.Variable.get(
          "TFV_PROJECT_ID", default_var=Project.TPU_PROD_ENV_ONE_VM.value
      ),
      cluster_name=models.Variable.get(
          "TFV_CLUSTER_NAME", default_var="yuna-auto-testing"
      ),
      node_pool_name=models.Variable.get(
          "TFV_NODE_POOL_NAME", default_var="tpu-info-fromat-test-v6e"
      ),
      region=models.Variable.get(
          "TFV_REGION", default_var=Region.US_EAST5.value
      ),
      location=models.Variable.get(
          "TFV_LOCATION", default_var=Region.US_EAST5.value
      ),
      node_locations=models.Variable.get(
          "TFV_NODE_LOCATIONS", default_var=Zone.US_EAST5_B.value
      ),
      num_nodes=models.Variable.get("TFV_NUM_NODES", default_var=4),
      machine_type=models.Variable.get(
          "TFV_MACHINE_TYPE", default_var=MachineVersion.CT6E_STAND_4T.value
      ),
      tpu_topology=models.Variable.get("TFV_TPU_TOPOLOGY", default_var="4x4"),
  )
  cluster_info_2 = replace(
      cluster_info,
      node_pool_name=models.Variable.get(
          "TFV_NODE_POOL_NAME", default_var="tpu-info-format-test-v6e-2"
      ),
  )

  kubeconfig_path = "/tmp/kubeconfig"
  jobset_config = JobSet(
      jobset_name="tpu-info-v6e-workload",
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
      image="us-docker.pkg.dev/tpu-prod-env-one-vm/yuna-docker-repo/tpu-info:v0.5.1",
      tpu_cores_per_pod=4,
  )

  workload_script = Workload.JAX_TPU_BENCHMARK

  with TaskGroup(group_id="create_node_pool") as create_node_pool:
    create_first_node_pool = node_pool.create.override(
        task_id="node_pool_1",
        retries=2,
    )(
        node_pool=cluster_info,
        reservation="cloudtpu-20250131131310-2118578099",
    )

    create_second_node_pool = node_pool.create.override(
        task_id="node_pool_2",
        retries=2,
    )(
        node_pool=cluster_info_2,
        reservation="cloudtpu-20250131131310-2118578099",
    )

  apply_time = jobset.run_workload(
      node_pool=cluster_info,
      kubeconfig=kubeconfig_path,
      yaml_config=jobset_config.generate_yaml(workload_script=workload_script),
      namespace=jobset_config.namespace,
  )

  active_pods = jobset.get_active_pods.override(task_id="get_active_pod")(
      node_pool=cluster_info,
      kubeconfig=kubeconfig_path,
      namespace=jobset_config.namespace,
  )

  wait_for_job_start = jobset.wait_for_jobset_started.override(
      task_id="wait_for_job_start"
  )(cluster_info, pod_name_list=active_pods, job_apply_time=apply_time)

  tpu_info_outputs = (
      get_tpu_info_from_pod.override(task_id="get_tpu_info")
      .partial(kubeconfig=kubeconfig_path)
      .expand(pod_name=active_pods)
  )

  tpu_info_output = (
      tpu_info.parse_tpu_info_output.override(task_id="get_each_metric_table")
      .partial()
      .expand(output=tpu_info_outputs)
  )

  with TaskGroup(group_id="verification_group") as verification_group:
    verify_table_amount_task = (
        verify_table_amount.override(task_id="verify_table_amount_task")
        .partial()
        .expand(tpu_info_output=tpu_info_output)
    )

    validate_tpu_chips_metric = (
        validate_chips_table.override(task_id="validate_tpu_chips_metric")
        .partial(node_pool=cluster_info)
        .expand(tpu_info_output=tpu_info_output)
    )

    validate_runtime_metric = (
        validate_runtime_table.override(task_id="validate_runtime_metric")
        .partial()
        .expand(tpu_info_output=tpu_info_output)
    )

    validate_tensorcore_metric = (
        validate_tensorcore_table.override(task_id="validate_tensorcore_metric")
        .partial()
        .expand(tpu_info_output=tpu_info_output)
    )

    validate_latency_metric = (
        validate_latency_table.override(task_id="validate_latency_metric")
        .partial()
        .expand(tpu_info_output=tpu_info_output)
    )

  clean_up_workload = jobset.end_workload.override(
      task_id="clean_up_workload", trigger_rule=TriggerRule.ALL_DONE
  )(
      node_pool=cluster_info,
      kubeconfig=kubeconfig_path,
      jobset_name=jobset_config.jobset_name,
      namespace=jobset_config.namespace,
  ).as_teardown(
      setups=apply_time
  )

  with TaskGroup(group_id="cleanup_node_pool") as cleanup_node_pool:
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

  (
      verify_table_amount_task
      >> [
          validate_tpu_chips_metric,
          validate_runtime_metric,
          validate_tensorcore_metric,
          validate_latency_metric,
      ]
  )

  [create_first_node_pool, create_second_node_pool]
  (cleanup_first_node_pool >> cleanup_second_node_pool)

  (
      create_node_pool
      >> apply_time
      >> active_pods
      >> wait_for_job_start
      >> tpu_info_outputs
      >> tpu_info_output
      >> verification_group
      >> clean_up_workload
      >> cleanup_node_pool
  )
