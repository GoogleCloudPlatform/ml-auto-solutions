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

"""A DAG to test "Jobset Ready Healthiness" metric."""

import datetime

from airflow.decorators import task
from airflow import models
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags.common.vm_resource import Region, Zone
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload
from dags.tpu_observability.configs.common import MachineConfigMap
from dags import composer_env

# Can be moved to jobset_util.py
@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def validate_replica_number(
    node_pool: node_pool.Info,
    jobset_config: JobSet,
    replica_type: str,
    replica_num: int
):
  found_replia_num = jobset.get_replica_num(
      replica_type=replica_type,
      job_name=jobset_config.replicated_job_name,
      node_pool=node_pool
  )

  return (found_replia_num == replica_num)


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="jobset_healthiness_ready",
    start_date=datetime.datetime(2025, 8, 10),
    schedule="00 03 * * *",
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "healthiness",
        "tpu-obervability",
        "TPU",
        "v6e-16",
    ],
    description=(
        "This DAG tests the 'Ready' status of jobset healthiness by "
        "comparing the number of 'Ready' replicas before and after "
        "a jobset is running."
    ),
    doc_md="""
      # JobSet Healthiness Test For the "Ready" Status

      ### Description
      This DAG automates the process of creating node-pools, ensuring the
      correct number of "Ready" replicas appear, then launching a jobset on
      multiple replicas to ensure the correct number begin running.

      ### Prerequisites
      This test requires an existing cluster to run.

      ### Procedures
      First two node-pools are created. The validation test is then run to
      check if the number of "Ready" replicas is 0. A jobset is then launched
      which uses 2 replicas. Once the jobset is running the jobs should
      quickly enter the "Ready" state. The number of found replicas is
      tested against the number of replicas which should be "Ready". If they
      match the DAG is a success.
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
            "NODE_POOL_NAME", default_var="jobset-healthiness-ready-v6e"
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

    cluster_info_2 = node_pool.Info(
        project_id=models.Variable.get("PROJECT_ID", default_var="cienet-cmcs"),
        cluster_name=models.Variable.get(
            "CLUSTER_NAME", default_var=cluster_name
        ),
        node_pool_name=models.Variable.get(
            "NODE_POOL_NAME", default_var="jobset-healthiness-ready-v6e-2"
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
        jobset_name="jobset-healthiness-ready",
        namespace="default",
        max_restarts=0,
        replicated_job_name="tpu-job-slice",
        replicas=2,
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

      create_node_pool_2 = node_pool.create(
          node_pool=cluster_info_2,
          reservation="cloudtpu-20251107233000-1246578561",
      )

      validate_zero_replicas = validate_replica_number(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          replica_type="ready",
          replica_num=0
      )

      start_workload = jobset.run_workload(
          node_pool=cluster_info,
          yaml_config=jobset_config.generate_yaml(
              workload_script=Workload.READY_TPU
          ),
          namespace=jobset_config.namespace,
      )

      validate_ready_replicas = validate_replica_number(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          replica_type="ready",
          replica_num=jobset_config.replicas
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

      cleanup_node_pool_2 = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info_2).as_teardown(
          setups=create_node_pool_2,
      )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          create_node_pool
          >> create_node_pool_2
          >> validate_zero_replicas
          >> start_workload
          >> validate_ready_replicas
          >> cleanup_workload
          >> cleanup_node_pool
          >> cleanup_node_pool_2

      )
      # pylint: enable=pointless-statement
