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
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.common.vm_resource import Region, Zone
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_rollback_ttr",
    start_date=datetime.datetime(2025, 8, 10),
    schedule="00 02 * * *",
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
  cluster_name = "tpu-observability-automation"
  cluster_name += "-prod" if composer_env.is_prod_env() else "-dev"

  for machine in MachineConfigMap:
    config = machine.value
    cluster_info = node_pool.Info(
        project_id=models.Variable.get("PROJECT_ID", default_var="cienet-cmcs"),
        cluster_name=models.Variable.get(
            "CLUSTER_NAME", default_var=cluster_name
        ),
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME", default_var="jobset-ttr-rollback-v6e"
        ),
        region=models.Variable.get(
            "REGION", default_var=Region.US_CENTRAL1.value
        ),
        location=models.Variable.get(
            "LOCATION", default_var=Region.US_CENTRAL1.value
        ),
        node_locations=models.Variable.get(
            "LOCATIONS", default_var=Zone.US_CENTRAL1_B.value
        ),
        num_nodes=models.Variable.get("NUM_NODES", default_var=4),
        machine_type=config.machine_version.value,
        tpu_topology=config.tpu_topology,
    )

    jobset_config = JobSet(
        jobset_name="ttr-rollback-v6e-workload",
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
      create_node_pool = node_pool.create(
          node_pool=cluster_info,
          reservation="cloudtpu-20251107233000-1246578561",
      )

      start_workload = jobset.run_workload(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=Workload.JAX_TPU_BENCHMARK
          ),
          namespace=jobset_config.namespace,
      )

      ensure_all_pods_running = jobset.wait_for_all_pods_running(
          num_pods=(jobset_config.replicas * jobset_config.parallelism),
          node_pool=cluster_info,
      )

      rollback_node_pool = node_pool.rollback(node_pool=cluster_info)

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found(
          node_pool=cluster_info
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

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          create_node_pool
          >> start_workload
          >> ensure_all_pods_running
          >> rollback_node_pool
          >> wait_for_metric_upload
          >> cleanup_workload
          >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
