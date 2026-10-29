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

"""DAG definition for running MaxText Pathways Elastic benchmarks on GKE."""

import datetime
from absl import logging

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.common import test_owner
from dags.common.goodput_utils import check_workload_goodput
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from dags.maxtext_pathways.configs.goodput_utils import (
    GOODPUT_LOG_LIST,
    check_goodput_logname,
    phase1_validate,
    phase2_validate,
    phase3_validate,
)
from dags.maxtext_pathways.configs.utils import (
    get_dag_parameters,
    generate_install_dependencies_commands,
    generate_derived_parameters,
    interrupt_worker_pod,
    check_gcp_logs_exist,
    check_logs_stream,
    worker_pod_interruption,
    COLOCATED_PYTHON_IMAGE,
)
from xlml.utils import gcluster, gke, kpo

ELASTIC_TYPE = ["Pause-resume", "Replica-resize"]

# Pause resume configuration
elastic_params = ui_params.PARAMETERS.copy()
elastic_params.update({
    "colocated_python_image": ui_params.Param(
        COLOCATED_PYTHON_IMAGE,
        type="string",
        title="Colocated Python Image",
        description="Colocated Python image for pathways.",
    ),
    "elastic_type": ui_params.Param(
        ELASTIC_TYPE[0],
        type="string",
        title="Elastic Type",
        description="Pause-resume/Replica-resize",
        enum=ELASTIC_TYPE,
    ),
})

# Replica resize configuration
replica_params = elastic_params.copy()
replica_params.update({
    "elastic_type": ui_params.Param(
        ELASTIC_TYPE[1],
        type="string",
        title="Elastic Type",
        description="Pause-resume/Replica-resize",
        enum=ELASTIC_TYPE,
    ),
    "num_slices_list": ui_params.Param(
        2,
        type="integer",
        title="Number Slices",
        description="Number of slices",
    ),
})


# TODO(b/576036761): Remove the temporary branch checkout once
# feat/benchmarks-elastic-goodput-ckpt-flags is merged into maxtext.
@task
def generate_commands(
    dag_params: dict, derived_params: dict, recipe_instance: recipe_cfg.Recipe
) -> str:
  """Generates a command string using config and derived parameters.

  Runtime modifications are made to the recipe command to enable elastic
  training and colocated Python data input.
  """
  # Initialization command.
  env_cmds = generate_install_dependencies_commands(
      cluster_name=dag_params["cluster_name"],
      location=derived_params["region"],
      project_id=dag_params["project"],
  )
  checkout_branch_cmd = (
      "git clone --depth=1 --branch feat/benchmarks-elastic-goodput-ckpt-flags "
      "https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext && "
      "cp -r /tmp/maxtext/benchmarks /deps/"
  )
  recipe_cmd = recipe_instance.run_command

  # Combine parameters to further generate the final command.
  all_params = {**dag_params, **derived_params}
  for key, value in all_params.items():
    if key in recipe_cfg.RECIPE_FLAG:
      if isinstance(value, int):
        recipe_cmd += f" --{key}={value}"
      else:
        recipe_cmd += f" --{key}='{value}'"
  # Override the default benchmark_steps too bigger.
  recipe_cmd += " --benchmark_steps=1000"
  # Specify the image for the colocated Python container.
  recipe_cmd += (
      f" --colocated_python_image='{dag_params['colocated_python_image']}'"
  )
  # Add proxy_flags to enable elastic training and colocated Python data input.
  recipe_cmd += (
      f" --proxy_flags='--virtual_slices={derived_params['topology']} "
      f"--num_elastic_slices={derived_params['num_elastic_slices']} "
      " --sidecar_name=external'"
  )
  recipe_cmd += (
      " --worker_flags='--cloud_pathways_sidecar_shm_directory=/tmp/sidecar "
      "--enable_metrics_collection=false "
      " --xla_tpu_use_enhanced_launch_barrier=true'"
  )
  # Enable elastic training.
  recipe_cmd += " --elastic_enabled=True"
  recipe_cmd += " --enable_single_controller=True"
  recipe_cmd += (
      f" --elastic_min_slice_count={derived_params['elastic_min_slice_count']}"
  )
  # Enable goodput setting.
  recipe_cmd += " --enable_pathways_goodput=True"
  recipe_cmd += " --enable_goodput_recording=True"
  recipe_cmd += " --goodput_upload_interval_seconds=30"
  recipe_cmd += " --monitor_goodput=True"
  # Enable checkpointing for elastic training.
  recipe_cmd += " --enable_checkpointing=True"
  recipe_cmd += " --async_checkpointing=True"
  recipe_cmd += " --enable_checkpoint_cloud_logger=True"
  recipe_cmd += " --checkpoint_period=100"
  # Enable colocated python data input and grain dataset.
  recipe_cmd += " --colocated_python_data_input=True"
  recipe_cmd += (
      " --tokenizer_path='src/maxtext/assets/tokenizers/tokenizer.llama2'"
  )
  recipe_cmd += " --dataset_type='grain'"
  recipe_cmd += (
      " --grain_train_files='gs://tess-tpu-dataloading-"
      "us-central1/array-record/c4/en/3.0.1/c4-train.array_record*'"
  )
  recipe_cmd += " --maxtext_override='grain_use_elastic_iterator=True'"
  recipe_cmd += " --maxtext_override='packing=False'"
  recipe_cmd += " --maxtext_override='elastic_timeout_seconds=3600'"
  recipe_cmd += " --maxtext_override='grain_worker_count=0'"
  # Add the skip-validation flag in the recipe to bypass xpk checks.
  recipe_cmd += " --skip-validation"
  formatted_cmds = recipe_cmd.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")
  commands = " && ".join([
      env_cmds,
      checkout_branch_cmd,
      recipe_cmd,
  ])

  return commands


def worker_pod_interruption_with_se(
    project_id: str = "",
    region: str = "",
    cluster_name: str = "",
    workload_id: str = "",
    times: int = 3,
    entry_log_pattern: str = "completed step:",
    elastic_log_pattern: str = "Elastic attempt",
    end_log_pattern: str = "Sufficient slices active:",
) -> DAGNode:
  """Run a test job with worker pod interruption."""
  with TaskGroup(group_id="worker_pod_interruption") as group:
    last_timestamp = None
    for i in range(1, times + 1):
      wait_for_step = check_logs_stream.override(
          task_id=f"wait_for_step_starts_{i}",
          retries=5,
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=entry_log_pattern,
          since_time=last_timestamp,
      )
      phase1_metrics = phase1_validate(
          project_id=project_id,
          workload_id=workload_id,
          start_time=wait_for_step,
          times=i,
      )

      wait_for_step >> phase1_metrics

      trigger_interrupt = interrupt_worker_pod.override(
          task_id=f"interrupt_worker_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
      )

      wait_for_elastic_attempt = check_logs_stream.override(
          task_id=f"wait_for_elastic_attempt_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=elastic_log_pattern,
          since_time=trigger_interrupt,
      )

      phase2_metrics = phase2_validate(
          project_id=project_id,
          workload_id=workload_id,
          slices_phase1=phase1_metrics,
          start_time=trigger_interrupt,
          times=i,
      )

      wait_for_elastic_attempt >> phase2_metrics

      wait_for_slices_active = check_logs_stream.override(
          task_id=f"wait_for_slices_active_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=end_log_pattern,
          since_time=wait_for_elastic_attempt,
      )

      phase3_metrics = phase3_validate(
          project_id=project_id,
          workload_id=workload_id,
          start_time=wait_for_slices_active,
          times=i,
      )

      wait_for_slices_active >> phase3_metrics
      chain(
          wait_for_step,
          trigger_interrupt,
          wait_for_elastic_attempt,
          wait_for_slices_active,
      )

      last_timestamp = wait_for_slices_active

    return group


RECIPE_INSTANCE = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
RECIPE_NAME = RECIPE_INSTANCE.value.lower()


def create_elastic_goodput_dag(
    dag_id: str,
    description: str,
    params: dict,
    slice_efficiency: bool = False,
    entry_log_pattern: str = "completed step:",
    end_log_pattern: str = "Sufficient slices active: 1 >= 1",
) -> models.DAG:
  schedule = SchedulingHelper.arrange_schedule_time(dag_id)
  with models.DAG(
      dag_id=dag_id,
      start_date=datetime.datetime(2025, 1, 1),
      schedule_interval=schedule if composer_env.is_prod_env() else None,
      catchup=False,
      default_args={
          "retries": 0,
      },
      tags=[
          "maxtext",
          "pathways",
          "mcjax",
          "benchmark",
          "nightly",
          "TPU",
          "v6e",
      ],
      description=description,
      params=params,
      doc_md=f"""
      # A DAG to run a MaxText {RECIPE_NAME} with elastic training on GKE.

      ### Description
      Specify different models and number of slices to test the MaxText
      {RECIPE_NAME} on different clusters. The DAG first generates recipe
      command through UI parameters, then runs the workload, waits and monitors
      the workload logs, and finally cleans up the workload.

      ### Prerequisites
      - This test requires an existing cluster.
      - If you're using a service account to pull an image from a different
        project, you need to grant the service account the
        `Artifact Registry Reader` role in that project.

      ### Procedures
      An Airflow Composer environment must be created, and the required DAG code
      must be deployed to the associated GCS bucket. To initiate the recipe, the
      user must access the Airflow UI, locate the specific DAG, and trigger it.

      ### Model Configuration
      If you want to add other TPU type models, you need to manually modify
      `/ml-auto-solutions/dags/maxtext_pathways/configs/model_configs.py`.
      """,
  ) as dag:
    # Define task dependencies by instantiating and linking tasks.
    fetched_params = get_dag_parameters()
    calculated_params = generate_derived_parameters(fetched_params, dag_id)
    generated_cmds = generate_commands(
        fetched_params, calculated_params, RECIPE_INSTANCE
    )

    formatted_goodput_logs = [
        log.format(workload_id=calculated_params["workload_id"])
        for log in GOODPUT_LOG_LIST
    ]

    start_recipe = kpo.run_command_in_kpo(
        start_cli_command=generated_cmds,
        workload_id="start_recipe",
        task_owner=test_owner.DORA_H,
        provisioning_timeout=datetime.timedelta(minutes=5),
        workload_run_timeout=datetime.timedelta(minutes=15),
        image_full_url=fetched_params["runner"],
    )

    if slice_efficiency:
      interruption_task = worker_pod_interruption_with_se(
          project_id=fetched_params["project"],
          region=calculated_params["region"],
          cluster_name=fetched_params["cluster_name"],
          workload_id=calculated_params["workload_id"],
          entry_log_pattern=entry_log_pattern,
          end_log_pattern=end_log_pattern,
      )
    else:
      interruption_task = worker_pod_interruption(
          project_id=fetched_params["project"],
          region=calculated_params["region"],
          cluster_name=fetched_params["cluster_name"],
          workload_id=calculated_params["workload_id"],
          entry_log_pattern=entry_log_pattern,
          end_log_pattern=end_log_pattern,
      )

    wait_for_workload_complete = gke.wait_for_workload_completion.override(
        task_id="wait_for_workload_complete",
        timeout=3600,
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
    )

    check_goodput_logs = check_gcp_logs_exist.override(
        task_id="check_goodput_logs",
        timeout=180,
    )(
        project_id=fetched_params["project"],
        location=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
        workload_id=calculated_params["workload_id"],
        expect_log_contains=formatted_goodput_logs,
    )

    goodput_logname = check_goodput_logname.override(
        timeout=180,
    )(
        project_id=fetched_params["project"],
        workload_id=calculated_params["workload_id"],
    )

    workload_goodput = check_workload_goodput.override(
        task_id="check_workload_goodput",
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        using_pathways=True,
    )

    clean_up_recipe = gcluster.clean_up_workload.override(
        task_id="clean_up_recipe"
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        zone=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
    ).as_teardown()

    chain(
        fetched_params,
        calculated_params,
        generated_cmds,
        start_recipe,
        interruption_task,
        wait_for_workload_complete,
        workload_goodput,
        goodput_logname,
        check_goodput_logs,
        clean_up_recipe,
    )

    return dag


# Instantiate the Goodput DAG
dag_elastic = create_elastic_goodput_dag(
    dag_id="pw_elastic_goodput",
    description=(
        f"A DAG to run a MaxText {RECIPE_NAME} with elastic training on GKE."
    ),
    params=elastic_params,
)

# Instantiate the Replica Resize Goodput DAG
dag_replica = create_elastic_goodput_dag(
    dag_id="pw_elastic_goodput_replica",
    description=(
        f"A DAG to run a MaxText {RECIPE_NAME} with goodput "
        "setting and replica resize on GKE."
    ),
    params=replica_params,
    end_log_pattern="Sufficient slices active: 2 >= 1",
)


# Instantiate the Goodput DAG
dag_elastic = create_elastic_goodput_dag(
    dag_id="pw_elastic_goodput_se",
    description=(
        f"A DAG to run a MaxText {RECIPE_NAME} with elastic training on GKE."
    ),
    params=elastic_params,
    slice_efficiency=True,
)

# Instantiate the Replica Resize Goodput DAG
dag_replica = create_elastic_goodput_dag(
    dag_id="pw_elastic_goodput_replica_se",
    description=(
        f"A DAG to run a MaxText {RECIPE_NAME} with goodput "
        "setting and replica resize on GKE."
    ),
    params=replica_params,
    slice_efficiency=True,
    end_log_pattern="Sufficient slices active: 2 >= 1",
)
