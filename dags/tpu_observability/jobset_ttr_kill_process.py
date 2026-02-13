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
A DAG to test jobset time-to-recover metric by
killing the main process inside a worker Pod.
"""

import datetime
import logging
import tempfile
import os

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup


from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import (
    MachineConfigMap,
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
)
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper, get_dag_timeout


DAG_ID = "jobset_ttr_kill_process"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)


@task
def kill_tpu_pod_workload(info: node_pool.Info, pod_name: str) -> None:
  """
  Kills the python process on a single pod.

  This task retrieves cluster credentials, then attempts to kill the JAX
  python process inside the specified pod. It ignores errors if the pod
  has already been deleted to ensure pipeline continuity.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        f"kubectl exec {pod_name} -n default -- pkill -9 -f python",
    ])

    try:
      subprocess.run_exec(cmd, env=env)
    except subprocess.ProcessKilledException:
      logging.info("Process was terminated with SIGKILL")
    except Exception as e:
      raise e


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
        "kill-main-process",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the use of killing the main process inside a jobset "
        "pod to interrupt a jobset, then polls the jobset time-to-recover "
        "metric to check if it is updated."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test by Killing Main Process

      ### Description
      This DAG validates the **Time-To-Recover (TTR)** metric by simulating a software-level failure.
      It provisions a TPU node pool, launches a JobSet workload, and then intentionally
      terminates the main Python process inside the worker Pods to trigger a recovery event.

      ### Prerequisites
      * Access to a GKE cluster with TPU support.
      * The `tpu-info` container image must be accessible by the cluster.
      * GCS configuration must be present at the defined `GCS_CONFIG_PATH`.

      ### Procedures
      1.  **Environment Setup**: Dynamically builds node pool info and creates a dedicated TPU node pool.
      2.  **Workload Launch**: Applies a JobSet YAML configured for JAX TPU benchmarks.
      3.  **Fault Injection**: Once the job is started, the DAG executes `pkill -9 -f python`
          inside the worker Pods via `kubectl exec`. This simulates a crash of the main training process.
      4.  **Metric Monitoring**: A sensor waits for the system to detect the failure, restart the
          workload, and successfully publish the `time-to-recover` metric.
      5.  **Cleanup**: Automatically tears down the JobSet and deletes the TPU node pool to
          ensure no resource leakage, regardless of whether the test passed or failed.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      selector = jobset.generate_node_pool_selector("jobset-ttr-kill-process")

      jobset_config = jobset.build_jobset_from_gcs_yaml(
          gcs_path=GCS_JOBSET_CONFIG_PATH,
          dag_name=DAG_ID,
          node_pool_selector=selector,
      )

      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
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

      apply_time = jobset.run_workload.override(task_id="run_workload")(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      pod_names = jobset.list_pod_names.override(task_id="list_pod_names")(
          node_pool=cluster_info,
          jobset_config=jobset_config,
      )

      wait_for_job_start = jobset.wait_for_jobset_started.override(
          task_id="wait_for_job_start"
      )(cluster_info, pod_name_list=pod_names, job_apply_time=apply_time)

      kill_tasks = (
          kill_tpu_pod_workload.override(task_id="kill_tpu_pod_workload")
          .partial(info=cluster_info)
          .expand(pod_name=pod_names)
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
          setups=apply_time
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          selector,
          jobset_config,
          cluster_info,
          create_node_pool,
          apply_time,
          pod_names,
          wait_for_job_start,
          kill_tasks,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
