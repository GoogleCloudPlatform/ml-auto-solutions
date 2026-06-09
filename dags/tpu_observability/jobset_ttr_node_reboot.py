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

"""A DAG to test JobSet Time-To-Recover (TTR) metric by triggering a node
reboot."""

import datetime
from datetime import timedelta

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common.scheduling_helper.scheduling_helper import (
    SchedulingHelper,
    get_dag_timeout,
)
from dags.common.task_group_with_timeout import TaskGroupWithTimeout
from dags.tpu_observability.configs.common import (
    GCS_CONFIG_PATH,
    GCS_JOBSET_CONFIG_PATH,
    MachineConfigMap,
)
from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.jobset_util import Workload

DAG_ID = "jobset_ttr_node_reboot"
DAGRUN_TIMEOUT = get_dag_timeout(DAG_ID)
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

# Keyword arguments are generated dynamically at runtime (pylint does not
# know this signature).
with models.DAG(  # pylint: disable=unexpected-keyword-arg
    dag_id=DAG_ID,
    start_date=datetime.datetime(2026, 1, 21),
    schedule=SCHEDULE if composer_env.is_prod_env() else None,
    dagrun_timeout=DAGRUN_TIMEOUT,
    catchup=False,
    tags=[
        "cloud-ml-auto-solutions",
        "jobset",
        "time-to-recover",
        "tpu-observability",
        "node_reboot",
        "TPU",
        "v6e-16",
    ],
    description=(
        "Tests JobSet TTR metric by rebooting a random TPU node to trigger "
        "recovery, then polls the metric for updates."
    ),
    doc_md="""
      # JobSet Time-To-Recover (TTR) Test Using Random Node Reboot

      ### Description
      This DAG verifies that a TPU JobSet can recover from a hardware-level failure.
      It launches a JobSet, executes a `reboot` command on a random node via a
      privileged container (using nsenter), and uses a sensor to confirm that
      the TTR (Time-To-Recover) metric is recorded.

      ### Prerequisites
      This test requires an existing cluster to run.
      GKE Cluster with TPU v6e support.
      The JobSet must be configured with
      privileged: True to allow node-level interaction.

      ### Procedures
      First the node-pool is created, a jobset yaml is then launched on the cluster.
      After initialization, a random node reboot is triggered by executing
      `nsenter` through a privileged pod. This simulates a hardware failure
      by taking one of the TPU nodes offline. A sensor finally polls Cloud
      Monitoring to confirm the jobset TTR metric is updated.
      """,
) as dag:
  for machine in MachineConfigMap:
    config = machine.value

    # Keyword arguments are generated dynamically at runtime (pylint does not
    # know this signature).
    with TaskGroupWithTimeout(  # pylint: disable=unexpected-keyword-arg
        group_id=f"v{config.tpu_version.value}",
        timeout=timedelta(minutes=90),
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
          privileged=True,
      )

      selector = jobset.generate_node_pool_selector(DAG_ID)
      jobset_name = jobset.generate_jobset_name(jobset_config.dag_id_prefix)

      create_node_pool = node_pool.create.override(task_id="create_node_pool")(
          node_pool=cluster_info,
          node_pool_selector=selector,
      )

      startup = jobset.create_jobset_startup_tasks(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
          node_pool_selector=selector,
          workload_type=Workload.JAX_TPU_BENCHMARK,
      )

      target_pod = jobset.draw_random_pod(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      )

      reboot_node = jobset.operate_pod.override(task_id="reboot_node")(
          node_pool=cluster_info,
          operation=jobset.PodOperationSpec.reboot(),
          pod_name=target_pod,
          namespace="default",
      )

      wait_for_metric_upload = jobset.wait_for_jobset_ttr_to_be_found.override(
          task_id="wait_for_jobset_ttr_to_be_found"
      )(
          node_pool=cluster_info,
          jobset_name=jobset_name,
      )

      cleanup_workload = jobset.end_workload.override(
          task_id="cleanup_workload", trigger_rule=TriggerRule.ALL_DONE
      )(
          node_pool=cluster_info,
          jobset_config=jobset_config,
          jobset_name=jobset_name,
      ).as_teardown(
          setups=startup.jobset_start_time
      )

      cleanup_node_pool = node_pool.delete.override(
          task_id="cleanup_node_pool", trigger_rule=TriggerRule.ALL_DONE
      )(node_pool=cluster_info).as_teardown(
          setups=create_node_pool,
      )

      chain(
          selector,
          jobset_name,
          create_node_pool,
          *startup.tasks,
          target_pod,
          reboot_node,
          wait_for_metric_upload,
          cleanup_workload,
          cleanup_node_pool,
      )
