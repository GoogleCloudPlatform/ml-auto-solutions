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

"""A DAG to test jobset uptime metric."""

import datetime

from airflow import models
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.tpu_observability.configs.common import MachineConfigMap, VM_GCS_CONFIG_PATH
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import JobSet, Workload


# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id="chips_capacity_validation_dag",
    start_date=datetime.datetime(2025, 8, 15),
    default_args={"retries": 0},
    schedule=None,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "chips_capacity",
        "tpu-observability",
        "TPU",
        "v6e-16",
    ],
    description=(" "),
    doc_md="""

      """,
) as dag:
  jobset_config = JobSet(
      jobset_name="chips-capacity-validation-v6e-workload",
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
      image="asia-northeast1-docker.pkg.dev/cienet-cmcs/"
      "yuna-docker/tpu-info:v0.5.1",
      tpu_cores_per_pod=4,
  )
  workload_script = Workload.JAX_TPU_BENCHMARK

  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroup(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}"
    ):
      cluster_info = node_pool.build_node_pool_info_from_gcs_yaml.override(
          task_id="build_node_pool_info_from_gcs_yaml"
      )(
          gcs_path=VM_GCS_CONFIG_PATH,
          dag_name="chips_capacity_validation_dag",
          is_prod=composer_env.is_prod_env(),
          machine_type=config.machine_version.value,
          tpu_topology=config.tpu_topology,
      )

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
      )

      apply_time = jobset.run_workload.override(task_id="run_workload")(
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

      instance_ids = jobset.list_instance_ids_by_pod_names.override(
          task_id="list_instance_ids_by_pod_names"
      )(
          node_pool=cluster_info,
          namespace=jobset_config.namespace,
          jobset_name=jobset_config.jobset_name,
      )

      # Keyword arguments are generated dynamically at runtime (pylint does not
      # know this signature).
      with TaskGroup(  # pylint: disable=unexpected-keyword-arg
          group_id="verification_group"
      ) as verification_group:
        scheduled_chips = (
            jobset.wait_for_tpu_scheduled_chips.override(
                task_id="wait_for_tpu_scheduled_chips"
            )
            .partial(node_pool=cluster_info, job_apply_time=apply_time)
            .expand(
                instance_id=instance_ids,
            )
        )

        tpu_active_chips = (
            jobset.wait_for_tpu_active_chips.override(
                task_id="wait_for_tpu_active_chips"
            )
            .partial(node_pool=cluster_info, job_apply_time=apply_time)
            .expand(
                instance_id=instance_ids,
            )
        )

        tpu_utilized_chips = (
            jobset.wait_for_tpu_utilized_chips.override(
                task_id="wait_for_tpu_utilized_chips"
            )
            .partial(node_pool=cluster_info, job_apply_time=apply_time)
            .expand(
                instance_id=instance_ids,
            )
        )

        tpu_chips_status = (
            jobset.wait_for_tpu_chip_state.override(
                task_id="wait_for_tpu_chip_state"
            )
            .partial(node_pool=cluster_info, job_apply_time=apply_time)
            .expand(
                instance_id=instance_ids,
            )
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

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      # Airflow uses >> for task chaining, which is pointless for pylint.
      # pylint: disable=pointless-statement
      (
          cluster_info
          >> create_node_pool
          >> apply_time
          >> pod_names
          >> wait_for_job_start
          >> instance_ids
          >> verification_group
          >> clean_up_workload
          >> cleanup_node_pool
      )
      # pylint: enable=pointless-statement
