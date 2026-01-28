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

"""A DAG to test jobset time-to-recover metric using a jobset pod delete."""

import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap, GCS_CONFIG_PATH

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_ttr_pod_delete",
    start_date=datetime.datetime(2026, 1, 8),
    schedule="0 19 * * *" if composer_env.is_prod_env() else None,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu-observability",
        "delete_pod",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the JobSet time-to-recover metric by deleting a random "
        "pod to trigger a recovery, then polls the metric to check if it is updated."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Random Pod Deletion

      ### Description
      This DAG verifies that JobSet can recover from a single pod failure.
      It launches a JobSet, deletes one running pod, and then uses a sensor
      to confirm that the JobSet controller triggers a restart.

      ### Prerequisites
      This test requires an existing cluster to run.

      ### Procedures
      First the node-pool is created, a jobset yaml is then launched on the
      cluster and given a short period of time to initialize. After this a
      random pod deletion is triggered to interrupt the jobset. A sensor is
      finally run which will poll Cloud Monitoring to detect that the jobset
      time-to-recover (TTR) metric has been updated, resulting in a success,
      or timeout, and fail.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    jobset_config = JobSet(
        jobset_name="ttr-delete-v6e-workload",
        namespace="default",
        max_restarts=5,
        replicated_job_name="tpu-job-slice",
        replicas=1,
        backoff_limit=0,
        completions=4,
        parallelism=4,
        tpu_accelerator_type="tpu-v6e-slice",
        tpu_topology="4x4",
        container_name="jax-tpu-worker",
        image="python:3.11",
        tpu_cores_per_pod=4,
    )

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=GCS_CONFIG_PATH,
          dag_name="jobset_ttr_pod_delete",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
      )

      start_workload = jobset.run_workload.override(task_id="start_workload")(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=Workload.JAX_TPU_BENCHMARK
          ),
          namespace=jobset_config.namespace,
      )

      ensure_all_pods_running = jobset.wait_for_all_pods_running.override(
          task_id="ensure_all_pods_running"
      )(
          num_pods=(jobset_config.replicas * jobset_config.parallelism),
          node_pool=cluster_info,
      )

      delete_random_pod = jobset.delete_one_random_pod.override(
          task_id="delete_random_pod"
      )(node_pool=cluster_info, namespace=jobset_config.namespace)

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_jobset_ttr_to_be_found"
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_name=jobset_config.jobset_name,
          namespace=jobset_config.namespace,
      ).as_teardown(
          setups=start_workload
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          create_node_pool,
          start_workload,
          ensure_all_pods_running,
          delete_random_pod,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
