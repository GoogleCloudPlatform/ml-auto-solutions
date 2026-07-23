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

import os
import datetime
from absl import logging

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from google.cloud import logging as gcp_logging
from ml_goodput_measurement import goodput

from dags import composer_env
from dags.common import test_owner
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from dags.maxtext_pathways.configs.utils import (
    get_dag_parameters,
    generate_install_dependencies_commands,
    generate_derived_parameters,
    check_gcp_logs_exist,
    COLOCATED_PYTHON_IMAGE,
)
from xlml.utils import kpo, xpk

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

GOODPUT_LOG_LIST = [
    "Cumulative goodput monitoring process started for job: {workload_id}",
    "Started Goodput upload to Tensorboard & GCM in the background!",
    "Sent Goodput metrics to GCM Monitoring.",
    "Final goodput query and upload for job: {workload_id}",
    "Flushed final metrics and safe exited from Goodput monitoring.",
]


@task.sensor(poke_interval=10, timeout=3600, mode="reschedule")
def check_goodput_logname(
    project_id: str,
    workload_id: str,
) -> bool:
  """
  Counts occurrences of a string pattern in GCP
  Cloud Logging for a specific workload.
  """
  # Initialize the GCP Logging Client
  client = gcp_logging.Client(project=project_id)

  log_filter = f'logName="projects/{project_id}/logs/goodput_{workload_id}" '

  logging.info(f"Querying GCP Logging with filter: {log_filter}")

  # Fetch the entries. (Adjust page_size based on log volume to optimize speed)
  entries = client.list_entries(filter_=log_filter, page_size=500)

  # Consolidate all log payloads into a single text body
  log_lines = []
  for entry in entries:
    payload = entry.payload

    if payload:
      if isinstance(payload, str):
        log_lines.append(payload)
      elif isinstance(payload, dict):
        message = payload.get("message") or payload.get("textPayload")
        if message:
          log_lines.append(str(message))
        else:
          log_lines.append(str(payload))

  full_logs_text = "\n".join(log_lines)

  if not full_logs_text:
    logging.info("No logs found yet in Cloud Logging for filter.")
    return False
  logging.info(f"Full Logs Text:\n{full_logs_text}")

  return True


@task
def check_workload_goodput(
    workload_id: str,
    project_id: str,
) -> bool:
  """
  Query and log Goodput/Badput metrics for the MaxText XPK workload.
  """
  goodput_logger_name = f"goodput_{workload_id}"
  os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
  goodput_calculator = goodput.GoodputCalculator(
      job_name=workload_id,
      logger_name=goodput_logger_name,
      using_pathways=True,
  )
  (
      current_goodput,
      badput_breakdown,
      last_step,
  ) = goodput_calculator.get_job_goodput(include_badput_breakdown=True)

  logging.info(f"Last step recorded: {last_step}")
  logging.info(f"Goodput (%): {current_goodput:.2f}%")
  logging.info("\n--- Badput Breakdown ---")

  for badput_type, percentage in badput_breakdown.items():
    if badput_type == goodput.BadputType.CUSTOM_BADPUT_EVENTS:
      logging.info(f"Badput due to {badput_type}:")
      custom_events = percentage
      if isinstance(custom_events, dict):
        for event_name, event_percentage in custom_events.items():
          logging.info(f"  - {event_name}: {event_percentage:.2f}%")
    else:
      # Access the name attribute of the enum member
      logging.info(f"Badput due to {badput_type.name}: {percentage:.2f}%")


@task
def generate_commands(
    dag_params: dict, derived_params: dict, recipe_instance: recipe_cfg.Recipe
) -> str:
  """Generates a command string using config and derived parameters.

  Runtime modifications are made to the recipe command to enable elastic
  training and colocated Python data input.
  """
  # Initialization command.
  env_cmds = generate_install_dependencies_commands()
  recipe_cmd = recipe_instance.run_command

  # Patch command for enabling elastic training, good put and
  # colocated Python data input.
  patch_cmd_runner = (
      r'sed -i "/python3 -m maxtext.trainers.pre_train.train/a '
      # enable elastice training
      r"          f\"elastic_enabled=True\",\n"
      r"          f\"enable_single_controller=True\",\n"
      f'          \\"elastic_min_slice_count='
      f'{derived_params["elastic_min_slice_count"]}\\",\\n'
      # enable goodput setting
      r"          f\"enable_pathways_goodput=True\",\n"
      r"          f\"enable_goodput_recording=True\",\n"
      r"          f\"goodput_upload_interval_seconds=30\",\n"
      r"          f\"monitor_goodput=True\",\n"
      # enanble checkpointing for elastic training
      r"          f\"async_checkpointing=True\",\n"
      r"          f\"enable_checkpoint_cloud_logger=True\",\n"
      r"          f\"checkpoint_period=10\","
      # enable colocated python data input
      r"          f\"colocated_python_data_input=True\",\n"
      r"          f\"tokenizer_path="
      r'"src/maxtext/assets/tokenizers/tokenizer.llama2"\"," '
      r"benchmarks/maxtext_xpk_runner.py"
  )

  model_configs_checkpointing = (
      r'sed -i "/model_name=\"default-basic-1\"/,/xla_flags/ { '
      r"s/\"enable_checkpointing\": False/\"enable_checkpointing\": True/; "
      r"/\"profiler\":/d; "
      r'}" benchmarks/maxtext_trillium_model_configs.py'
  )

  # Patch command to modify dataset configuration for default_basic_1.
  model_configs_grain = (
      r'sed -i "/model_name=\"default-basic-1\"/,/)/ { '
      r"s/\"dataset_type\": \"synthetic\"/\"dataset_type\": \"grain\"/; "
      r"s/\"dataset_path\":/# \"dataset_path\":/; "
      r'}" benchmarks/maxtext_trillium_model_configs.py'
  )
  # Text to be inserted after the line matching '"dataset_type": "grain"'
  insert_line = (
      r"            \"grain_train_files\": \"gs://tess-tpu-dataloading-"
      r"us-central1/array-record/c4/en/3.0.1/c4-train.array_record*\","
  )
  # Patch command to modify dataset configuration: append grain_train_files.
  model_configs_grain_sub = (
      'sed -i $\'/"dataset_type": "grain"/a \\\n'
      + insert_line
      + "' benchmarks/maxtext_trillium_model_configs.py"
  )

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
  # Add the skip-validation flag in the recipe to bypass xpk checks.
  recipe_cmd += " --skip-validation"
  formatted_cmds = recipe_cmd.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")
  commands = " && ".join([
      env_cmds,
      patch_cmd_runner,
      model_configs_checkpointing,
      model_configs_grain,
      model_configs_grain_sub,
      recipe_cmd,
  ])

  return commands


def worker_pod_interruption(
    project_id: str = "",
    region: str = "",
    cluster_name: str = "",
    workload_id: str = "",
) -> DAGNode:
  """Run a test job with worker pod interruption."""
  with TaskGroup(group_id="worker_pod_interruption") as group:
    previous_cycle_tail = None
    for i in range(1, 2):
      wait_for_step = xpk.check_last_logs.override(
          task_id=f"wait_for_step_starts_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains="completed step:",
      )

      trigger_interrupt = xpk.interrupt_worker_pod.override(
          task_id=f"interrupt_worker_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
      )

      # TODO(cienet): refine validation
      #   1. more precise log content and order
      #   2. use kubectl instead of CoreV1Api
      #   (since it doesn't support "since_time")
      #   3. cache a timestamp, to skip the old logs

      wait_for_elastic_attempt = xpk.check_logs_exist.override(
          task_id=f"wait_for_elastic_attempt_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=f"Elastic attempt {i+1}",
      )

      wait_for_slices_active = xpk.check_logs_exist.override(
          task_id=f"wait_for_slices_active_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains="Sufficient slices active:",
          expected_count=i + 1,
      )

      chain(
          wait_for_step,
          trigger_interrupt,
          wait_for_elastic_attempt,
          wait_for_slices_active,
      )

      if previous_cycle_tail:
        chain(previous_cycle_tail, wait_for_step)
      previous_cycle_tail = wait_for_slices_active
    return group


RECIPE_INSTANCE = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
RECIPE_NAME = RECIPE_INSTANCE.value.lower()


def create_elastic_goodput_dag(
    dag_id: str, description: str, params: dict
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

    interruption_task = worker_pod_interruption(
        project_id=fetched_params["project"],
        region=calculated_params["region"],
        cluster_name=fetched_params["cluster_name"],
        workload_id=calculated_params["workload_id"],
    )

    wait_for_workload_complete = xpk.wait_for_workload_completion.override(
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
    )

    clean_up_recipe = xpk.clean_up_workload.override(
        task_id="clean_up_recipe", trigger_rule=TriggerRule.ALL_DONE
    )(
        workload_id=calculated_params["workload_id"],
        project_id=fetched_params["project"],
        zone=fetched_params["zone"],
        cluster_name=fetched_params["cluster_name"],
    )

    chain(
        fetched_params,
        calculated_params,
        generated_cmds,
        start_recipe,
        interruption_task,
        wait_for_workload_complete,
        goodput_logname,
        check_goodput_logs,
        workload_goodput,
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
)
