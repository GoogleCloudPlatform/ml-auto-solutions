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

import datetime
import time
from absl import logging

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.hooks.subprocess import SubprocessHook
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from xlml.utils import kpo, xpk, gke


@task.python(multiple_outputs=True)
def get_dag_parameters(**context) -> dict:
  """
  Fetches and returns the DAG run's configuration parameters.
  """
  dag_params = context.get("params", {})

  return dag_params


@task.python(multiple_outputs=True)
def generate_derived_parameters(dag_params: dict) -> dict:
  """
  Generates new parameters based on the initial DAG parameters.
  """
  derived_params = {}

  # Generate recipe workload_id.
  name = generate_recipe_workload_id(dag_params)
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

  return derived_params


@task
def generate_commands(
    dag_params: dict, derived_params: dict, recipe_instance: recipe_cfg.Recipe
) -> str:
  """
  Generates a command string using the initial DAG parameters and derived parameters.
  """
  # Initialization command.
  env_cmds = generate_install_dependencies_commands()
  recipe_cmd = recipe_instance.run_command

  # Combine parameters to further generate the final command.
  all_params = {**dag_params, **derived_params}
  for key, value in all_params.items():
    if key in recipe_cfg.RECIPE_FLAG:
      if isinstance(value, int):
        recipe_cmd += f" --{key}={value}"
      else:
        recipe_cmd += f" --{key}='{value}'"
  # Add the skip-validation flag in the recipe to bypass xpk system dependency check.
  recipe_cmd += " --skip-validation"
  formatted_cmds = recipe_cmd.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")

  commands = " && ".join([env_cmds, recipe_cmd])

  return commands


def generate_install_dependencies_commands() -> str:
  """
  Generate the shell commands to install necessary dependencies in the Pod.
  """
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


def generate_recipe_workload_id(params: dict) -> tuple[str, str]:
  """
  Generate a workload_id following the convention: {dag_id[:10]}-{timestamp[:10]}.
  This is intended to facilitate later deletion of the workload.
  """
  time.localtime()
  timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
  dag_id = RECIPE_NAME
  name = f"{dag_id[:10]}-{timestamp[:10]}"
  name = name[:40].replace("_", "-")

  return name


@task
def clean_up_pod(
    cluster_name: str, region: str, project: str, airflow_runtime: str
) -> None:
  """
  Use SubprocessHook to execute shell commands to delete Pods.
  """
  hook = SubprocessHook()

  commands = [
      "set -xue",
      "export KUBECONFIG=/tmp/kubeconfig",  # Change KUBECONFIG from /home/airflow to /tmp to avoid permission issue.
      f"gcloud container clusters get-credentials {cluster_name} --region={region} --project={project}",
      f"kubectl delete pod -l airflow-runtime={airflow_runtime} --namespace=default --force --grace-period=0",
  ]

  result = hook.run_command(
      ["bash", "-c", ";".join(commands)],
  )

  assert (
      result.exit_code == 0
  ), f"kubectl clean-up failed with code {result.exit_code}"


RECIPE_INSTANCE = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
RECIPE_NAME = RECIPE_INSTANCE.value.lower()

with models.DAG(
    dag_id=RECIPE_NAME,
    start_date=datetime.datetime(2025, 1, 1),
    schedule_interval="0 21 * * *" if composer_env.is_prod_env() else None,
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
        "v6e-64",
    ],
    description=f"A DAG to run a MaxText {RECIPE_NAME} on GKE.",
    params=ui_params.PARAMETERS,
    doc_md=f"""
    # A DAG to run a MaxText {RECIPE_NAME} on GKE.

    ### Description
    Specify different models and number of slices to test the MaxText {RECIPE_NAME} on different clusters.
    The DAG first generates recipe command through UI parameters, then runs the workload, waits and monitors the workload logs, and finally cleans up the workload.

    ### Prerequisites
    - This test requires an existing cluster.
    - If you're using a service account to pull an image from a different project, you need to grant the service account the `Artifact Registry Reader` role in that project.

    ### Procedures
    An Airflow Composer environment must be created, and the required DAG code must be deployed to the associated GCS bucket.
    To initiate the recipe, the user must access the Airflow UI, locate the specific DAG, and trigger its execution.

    ### Model Configuration
    If you want to add other TPU type models, you need to manually modify `/ml-auto-solutions/dags/maxtext_pathways/configs/model_configs.py`.
    """,
) as dag:
  recipe_runtime = (
      RECIPE_NAME.replace("_", "-") + '-{{ execution_date.strftime("%H%M%S") }}'
  )

  # Define task dependencies by instantiating and linking tasks.
  dag_params = get_dag_parameters()
  derived_params = generate_derived_parameters(dag_params)
  commands = generate_commands(dag_params, derived_params, RECIPE_INSTANCE)

  start_recipe = kpo.run_command_in_kpo(
      start_cli_command=commands,
      workload_id="start_recipe",
      task_owner=test_owner.DORA_H,
      provisioning_timeout=datetime.timedelta(minutes=5),
      workload_run_timeout=datetime.timedelta(minutes=15),
      image_full_url=dag_params["runner"],
  )

  wait_for_workload_complete = xpk.wait_for_workload_completion.override(
      task_id="wait_for_workload_complete",
  )(
      workload_id=derived_params["workload_id"],
      project_id=dag_params["project"],
      region=derived_params["region"],
      cluster_name=dag_params["cluster_name"],
  )

  clean_up_recipe = xpk.clean_up_workload.override(
      task_id="clean_up_recipe", trigger_rule=TriggerRule.ALL_DONE
  )(
      workload_id=derived_params["workload_id"],
      project_id=dag_params["project"],
      zone=dag_params["zone"],
      cluster_name=dag_params["cluster_name"],
  )

  (
      dag_params
      >> derived_params
      >> commands
      >> start_recipe
      >> wait_for_workload_complete
      >> clean_up_recipe
  )
  start_recipe >> wait_for_workload_complete
