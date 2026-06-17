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
import time
from absl import logging

from airflow import models
from airflow.decorators import task
from airflow.utils.trigger_rule import TriggerRule
from airflow.models.taskmixin import DAGNode
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.common import test_owner
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from xlml.utils import kpo, gke, xpk

ELASTIC_TYPE = ["Pause-resume", "Replica-resize"]
elastic_params = ui_params.PARAMETERS.copy()
elastic_params.update({
    "cluster_name": ui_params.Param(
        "pw-v6e-16x4",
        type="string",
        title="Cluster Name",
        description="GCP cluster name for training model.",
    ),
    "zone": ui_params.Param(
        "us-east4-b",
        type="string",
        title="Zone",
        description="Cluster zone.",
    ),
    "core_count": ui_params.Param(
        16,
        type="integer",
        title="Core Count",
        description='Device core count for the cluster. ex: v6e-"64"',
    ),
    "num_slices_list": ui_params.Param(
        1,
        type="integer",
        title="Number Slices",
        description="Number of slices",
    ),
    "colocated_python_image": ui_params.Param(
        "gcr.io/tpu-prod-env-multipod/lidanny_maxtext-colocated-python:latest",
        type="string",
        title="Colocated Python Image",
        description="Colocated Python image for pathways.",
    ),
    "server_image": ui_params.Param(
        # Reference to:
        # https://g3doc.corp.google.com/cloud/tpu/g3doc/fas/pathways-on-cloud/index.md?cl=head
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/"
        "server:20260521_RC00-jax_0.10.0",
        type="string",
        title="Server Image",
        description="Server image for pathways.",
    ),
    "proxy_image": ui_params.Param(
        # Reference to:
        # https://g3doc.corp.google.com/cloud/tpu/g3doc/fas/pathways-on-cloud/index.md?cl=head
        "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/"
        "proxy_server:20260521_RC00-jax_0.10.0",
        type="string",
        title="Proxy Image",
        description="Proxy image for pathways.",
    ),
    "elastic_type": ui_params.Param(
        ELASTIC_TYPE[0],
        type="string",
        title="Elastic Type",
        description="Pause-resume/Replica-resize",
        enum=ELASTIC_TYPE,
    ),
})


@task.python(multiple_outputs=True)
def get_dag_parameters(**context) -> dict:
  """Fetches and returns the DAG run's configuration parameters."""
  dag_params = context.get("params", {})

  return dag_params


@task.python(multiple_outputs=True)
def generate_derived_parameters(dag_params: dict) -> dict:
  """Generates new parameters based on the initial DAG parameters."""
  derived_params = {}

  # Generate recipe workload_id.
  name = generate_recipe_workload_id()
  derived_params["workload_id"] = name

  # Generate region by zone
  derived_params["region"] = gke.zone_to_region(dag_params["zone"])

  # Generate device_type.
  device_type = (
      dag_params["device_version"] + "-" + str(dag_params["core_count"])
  )
  derived_params["device_type"] = device_type

  # Confirm whether to use customized_model_name.
  if dag_params["selected_model_names"] == "customized_model_name":
    derived_params["selected_model_names"] = dag_params["customized_model_name"]

  # TODO(cienet): Refine the parameter parsing logic
  core_calc = dag_params["core_count"] // 4
  if dag_params["elastic_type"] == "Pause-resume":
    derived_params["elastic_min_slice_count"] = -1
    derived_params["topology"] = f"tpuv6e:4x{core_calc}"
    derived_params["num_elastic_slices"] = 1
  else:
    derived_params["elastic_min_slice_count"] = 1
    derived_params["topology"] = ",".join([f"tpuv6e:4x{core_calc}"] * 2)
    derived_params["num_elastic_slices"] = 2

  return derived_params


# TODO(cienet): Remove the temporary local code once these changes have been
# merged into the maxtext repository.
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

  # Patch command for enabling elastic training & colocated Python data input.
  patch_cmd_runner = (
      r'sed -i "/python3 -m maxtext.trainers.pre_train.train/a '
      r"          f\"elastic_enabled=True\",\n"
      r"          f\"enable_single_controller=True\",\n"
      f'          \\"elastic_min_slice_count='
      f'{derived_params["elastic_min_slice_count"]}\\",\\n'
      r"          f\"colocated_python_data_input=True\",\n"
      r"          f\"tokenizer_path="
      r'"src/maxtext/assets/tokenizers/tokenizer.llama2"\"," '
      r"benchmarks/maxtext_xpk_runner.py"
  )

  # Patch command to modify dataset configuration for default_basic_1.
  patch_cmd_model_configs_sub = (
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
  patch_cmd_model_configs_add = (
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
      patch_cmd_model_configs_sub,
      patch_cmd_model_configs_add,
      recipe_cmd,
  ])

  return commands


# TODO(cienet): Remove the temporary local code once these changes have been
# merged into the maxtext repository.
@task
def generate_commands_replica(
    dag_params: dict, derived_params: dict, recipe_instance: recipe_cfg.Recipe
) -> str:
  """Generates a command string using config and derived parameters.

  Runtime modifications are made to the recipe command to enable elastic
  training and colocated Python data input.
  """
  # Initialization command.
  env_cmds = generate_install_dependencies_commands()
  recipe_cmd = recipe_instance.run_command

  # Patch command for enabling elastic training & colocated Python data input.
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
      r"          f\"checkpoint_period=10\",\n"
      r'" benchmarks/maxtext_xpk_runner.py'
  )

  # changing to synthetic and disabling colocated_python:
  # https://b.corp.google.com/issues/511164291#comment26
  patch_cmd_model_configs_sub = (
      r'sed -i "/model_name=\"default-basic-1\"/,/xla_flags/ { '
      r"s/\"enable_checkpointing\": False/\"enable_checkpointing\": True/; "
      r"/\"profiler\":/d; "
      r'}" benchmarks/maxtext_trillium_model_configs.py'
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
  recipe_cmd += " --benchmark_steps=3000"
  # Add proxy_flags to enable elastic training and colocated Python data input.
  recipe_cmd += (
      f" --proxy_flags='--virtual_slices={derived_params['topology']} "
      f"--num_elastic_slices={derived_params['num_elastic_slices']}'"
  )
  # Add the skip-validation flag in the recipe to bypass xpk checks.
  recipe_cmd += " --skip-validation"
  formatted_cmds = recipe_cmd.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")
  commands = " && ".join([
      env_cmds,
      patch_cmd_runner,
      patch_cmd_model_configs_sub,
      recipe_cmd,
  ])

  return commands


def generate_install_dependencies_commands() -> str:
  """Generate shell commands to install necessary dependencies in the Pod."""
  # fmt: off
  return " && ".join([
      # Update apt package list
      "sudo apt-get update",

      # Install kubectl
      "sudo apt-get install -y kubectl",

      # Install GKE auth plugin for cluster authentication
      "sudo apt-get install google-cloud-sdk-gke-gcloud-auth-plugin -y",

      # Install xpk
      *xpk.get_xpk_setup_cmd("/root", xpk.MAIN_BRANCH),

      # Install dependencies for maxtext
      "pip install omegaconf",

      # Prepare environment for further pip installs
      "cd /deps",
      "export USER=root",
  ])
  # fmt: on


def generate_recipe_workload_id() -> tuple[str, str]:
  """Generate a workload_id following the standard naming convention."""
  time.localtime()
  timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
  dag_id = RECIPE_NAME
  name = f"{dag_id[:10]}-{timestamp[:10]}"
  name = name[:40].replace("_", "-")

  return name


def worker_pod_interruption(
    project_id: str = "",
    region: str = "",
    cluster_name: str = "",
    workload_id: str = "",
    entry_log_pattern: str = "completed step:",
    elastic_log_pattern: str = "Elastic attempt",
    end_log_pattern: str = "Sufficient slices active:",
) -> DAGNode:
  """Run a test job with worker pod interruption."""
  with TaskGroup(group_id="worker_pod_interruption") as group:
    previous_cycle_tail = None
    for i in range(1, 4):
      wait_for_step = xpk.check_last_logs.override(
          task_id=f"wait_for_step_starts_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=entry_log_pattern,
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
          expect_log_contains=elastic_log_pattern,
      )

      wait_for_slices_active = xpk.check_logs_exist.override(
          task_id=f"wait_for_slices_active_{i}"
      )(
          project_id=project_id,
          region=region,
          cluster_name=cluster_name,
          workload_id=workload_id,
          expect_log_contains=end_log_pattern,
          expected_count=i + 1,
      )

      # TODO(cienet): Refine the mechanism to chain tasks
      _ = (
          wait_for_step
          >> trigger_interrupt
          >> wait_for_elastic_attempt
          >> wait_for_slices_active
      )

      if previous_cycle_tail:
        previous_cycle_tail >> wait_for_step
      previous_cycle_tail = wait_for_slices_active
    return group


RECIPE_INSTANCE = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
RECIPE_NAME = RECIPE_INSTANCE.value.lower()
DAG_ID = f"{RECIPE_NAME}_elastic"
SCHEDULE = SchedulingHelper.arrange_schedule_time(DAG_ID)

with models.DAG(
    dag_id=DAG_ID,
    start_date=datetime.datetime(2025, 1, 1),
    schedule_interval=SCHEDULE if composer_env.is_prod_env() else None,
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
    description=(
        f"A DAG to run a MaxText {RECIPE_NAME} with elastic training on GKE."
    ),
    params=elastic_params,
    doc_md=f"""
    # A DAG to run a MaxText {RECIPE_NAME} with elastic training on GKE.

    ### Description
    Pause-resume refers to the process of halting the training execution,
    saving its state (typically to a checkpoint), and later restarting
    the training, loading the state from the checkpoint to continue.
    Stop the training process when slices become unavailable, and starts it
    again later on the new set inherently. This mechanism is crucial for
    fault tolerance and elasticity. Resuming can occur on the same
    set of resources or a different set.

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
  recipe_runtime = (
      RECIPE_NAME.replace("_", "-") + '-{{ execution_date.strftime("%H%M%S") }}'
  )

  # Define task dependencies by instantiating and linking tasks.
  fetched_params = get_dag_parameters()
  calculated_params = generate_derived_parameters(fetched_params)
  generated_cmds = generate_commands(
      fetched_params, calculated_params, RECIPE_INSTANCE
  )

  start_recipe = kpo.run_command_in_kpo(
      start_cli_command=generated_cmds,
      workload_id="start_recipe",
      task_owner=test_owner.DORA_H,
      provisioning_timeout=datetime.timedelta(minutes=5),
      workload_run_timeout=datetime.timedelta(minutes=15),
      image_full_url=fetched_params["runner"],
  )

  # TODO(cienet): Add comments or documentation to explain the expected log
  # patterns.
  interruption_task = worker_pod_interruption(
      project_id=fetched_params["project"],
      region=calculated_params["region"],
      cluster_name=fetched_params["cluster_name"],
      workload_id=calculated_params["workload_id"],
      entry_log_pattern="completed step:",
      elastic_log_pattern="Elastic attempt",
      end_log_pattern="Sufficient slices active: 1 >= 1",
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

  clean_up_recipe = xpk.clean_up_workload.override(
      task_id="clean_up_recipe", trigger_rule=TriggerRule.ALL_DONE
  )(
      workload_id=calculated_params["workload_id"],
      project_id=fetched_params["project"],
      zone=fetched_params["zone"],
      cluster_name=fetched_params["cluster_name"],
  )

  (
      fetched_params
      >> calculated_params
      >> generated_cmds
      >> start_recipe
      >> interruption_task
      >> wait_for_workload_complete
      >> clean_up_recipe
  )

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
    "runner": ui_params.Param(
        # TODO(cienet): Replace with an official or production-ready image
        # TODO(cienet): Use an image tag instead of the full SHA hash
        "gcr.io/cloud-tpu-multipod-dev/lidanny_rr_dag@sha256:08dfca25461e3f2fb736e7313a742e1ceea6123858c0aec5dfe43d0c51d14e38",
        type="string",
        title="Runner Image",
        description="Runner image for the cluster",
    ),
})

DAG_ID_RESIZE = f"{RECIPE_NAME}_elastic_{ELASTIC_TYPE[1]}"
SCHEDULE_RESIZE = SchedulingHelper.arrange_schedule_time(DAG_ID_RESIZE)

with models.DAG(
    dag_id=DAG_ID_RESIZE,
    start_date=datetime.datetime(2025, 1, 1),
    schedule_interval=SCHEDULE_RESIZE if composer_env.is_prod_env() else None,
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
    description=(
        f"A DAG to run a MaxText {RECIPE_NAME}"
        "with elastic replica resize on GKE."
    ),
    params=replica_params,
    doc_md=f"""
    # A DAG to run a MaxText {RECIPE_NAME} with elastic replica resize on GKE.

    ### Description
    Replica-resize refers to the ability of the training job to dynamically
    adjust the number of active TPU slices (replicas) it uses during execution.
    Expected Behavior:
    - A change in slice availability (failure or addition)
    triggers an event. Often, a slice failure results in an error.
    - The elastic training framework detects this change.
    - Training on the previous configuration halts, and try to identify
    the new set of healthy, available slice.
    - The training job is automatically relaunched, loading the model
    state from the most recent checkpoint. The relaunched job now runs on
    the new set of available slices.

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
  recipe_runtime = (
      RECIPE_NAME.replace("_", "-") + '-{{ execution_date.strftime("%H%M%S") }}'
  )

  # Define task dependencies by instantiating and linking tasks.
  fetched_params = get_dag_parameters()
  calculated_params = generate_derived_parameters(fetched_params)
  generated_cmds = generate_commands_replica(
      fetched_params, calculated_params, RECIPE_INSTANCE
  )

  start_recipe = kpo.run_command_in_kpo(
      start_cli_command=generated_cmds,
      workload_id="start_recipe",
      task_owner=test_owner.DORA_H,
      provisioning_timeout=datetime.timedelta(minutes=5),
      workload_run_timeout=datetime.timedelta(minutes=15),
      image_full_url=fetched_params["runner"],
  )

  # TODO(cienet): Add comments or documentation to explain the expected log
  # patterns.
  interruption_task = worker_pod_interruption(
      project_id=fetched_params["project"],
      region=calculated_params["region"],
      cluster_name=fetched_params["cluster_name"],
      workload_id=calculated_params["workload_id"],
      entry_log_pattern="live slice count: 2",
      elastic_log_pattern="Elastic attempt",
      end_log_pattern="Sufficient slices active: 2 >= 1",
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

  clean_up_recipe = xpk.clean_up_workload.override(
      task_id="clean_up_recipe", trigger_rule=TriggerRule.ALL_DONE
  )(
      workload_id=calculated_params["workload_id"],
      project_id=fetched_params["project"],
      zone=fetched_params["zone"],
      cluster_name=fetched_params["cluster_name"],
  )

  (
      fetched_params
      >> calculated_params
      >> generated_cmds
      >> start_recipe
      >> interruption_task
      >> wait_for_workload_complete
      >> clean_up_recipe
  )
