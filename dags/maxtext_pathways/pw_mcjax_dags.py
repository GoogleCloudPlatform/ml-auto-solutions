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
import random
import string
import time
from absl import logging

from airflow import models
from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.hooks.subprocess import SubprocessHook
from airflow.utils.trigger_rule import TriggerRule

from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from kubernetes.client import models as k8s

from dags.common.vm_resource import DockerImage
from dags.maxtext_pathways.configs import commands as cmds
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from xlml.utils import xpk, gke


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

  # Generate recipe workload_id and temp_key.
  name, temp_post_fix = generate_recipe_workload_id(dag_params)
  derived_params["temp_key"] = temp_post_fix
  derived_params["recipe_workload_id"] = name

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
  env_cmds = generate_install_dependencies_commands(
      service_account=dag_params["service_account"]
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

  formatted_cmds = recipe_cmd.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")

  commands = " && ".join([env_cmds, recipe_cmd])

  return commands


# TODO(b/455427519): DockerImage.MAXTEXT_TPU_JAX_NIGHTLY cannot directly execute benchmark recipes.
# TODO(b/455412930): Build a Pathways-specific image. This feature will be changed to install dependencies directly from the image.
def generate_install_dependencies_commands(service_account: str) -> str:
  """
  Generate the shell commands to install necessary dependencies in the Pod.
  """
  env_cmds_list = (
      cmds.UPDATE_APT
      + cmds.INSTALL_MAKE
      + cmds.INSTALL_KUBECTL
      + cmds.INSTALL_DOCKER
      + cmds.SWITCH_SERVICE_ACCOUNT
      + cmds.INSTALL_KUBECTL_KJOB
      + cmds.INSTALL_KUBECTL_KUEUE
      + cmds.INSTALL_XPK
      + cmds.BACK_MAXTEXT
  )

  env_cmds = " && ".join(env_cmds_list)
  env_cmds = env_cmds.format(service_account=service_account)

  return env_cmds


# TODO(b/455415420): Extract workload_id from start_recipe log to avoid being out of sync with the MaxText repo.
def generate_recipe_workload_id(params: dict) -> tuple[str, str]:
  """
  Generate a random value in advance to fix the workload_id so that the workload can be deleted later.
  Please refer to the `generate_xpk_workload_cmd` function in the `/maxtext/benchmarks/maxtext_xpk_runner.py` file.
  """
  # Confirm whether to use customized_model_name.
  params = params.copy()
  if params["selected_model_names"] == "customized_model_name":
    params["selected_model_names"] = params["customized_model_name"]

  time.localtime()
  length_of_random_str = 3
  temp_post_fix = "".join(
      random.choice(string.ascii_lowercase + string.digits)
      for _ in range(length_of_random_str)
  )

  truncate_model_name = 10
  truncate_prefix = 3
  post_fix = f'-{params["num_slices_list"]}-{time.strftime("%m%d%H", time.localtime())}-{temp_post_fix}'
  common_prefix = params["user"]

  pw_prefix = "pw-"

  if params["selected_model_framework"] == "pathways":
    post_fix = f'-{params["num_slices_list"]}-{temp_post_fix}'
    name = f'{pw_prefix}{params["selected_model_names"].replace("_", "-")[:truncate_model_name - len(pw_prefix)]}'
  else:
    name = f'{params["selected_model_names"].replace("_", "-")[:truncate_model_name]}'

  name = f"{common_prefix[:truncate_prefix]}-{name}{post_fix}"

  return name, temp_post_fix


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


@task
def wait_workload_complete(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    benchmark_steps: int,
    override_timeout_in_min: int = 10,
    poke_interval_in_second: int = 60,
) -> bool:
  """
  Checks for the completion of an workload by repeatedly executing its core logic.
  The core logic uses workload logs to report the detailed status of successful or failed completion.
  """
  # Extract and reuse the logic in the` xpk` module to wait for a workload to be completed.
  impl = getattr(
      xpk.wait_for_workload_completion, "python_callable", None
  ) or getattr(xpk.wait_for_workload_completion, "__wrapped__", None)

  if impl is None:
    raise AirflowException(
        f"Cannot extract core callable from {xpk.wait_for_workload_completion}."
        "It might not be a valid task/sensor or is wrapped too deeply."
    )

  # Dynamically sets the Airflow task timeout based on a custom flag or calculates it using a benchmark step count.
  if override_timeout_in_min:
    timeout_in_min = override_timeout_in_min
  else:
    max_step_min = 5  # Average time required for each step in lama3-1-405b.
    timeout_in_min = benchmark_steps * max_step_min

  timeout_in_sec = timeout_in_min * 60

  logging.info(
      f"The timeout for this task is {timeout_in_min} minutes ({timeout_in_sec} seconds).\n"
      f"Check if completed every {poke_interval_in_second} seconds."
  )

  deadline = datetime.datetime.now() + datetime.timedelta(
      seconds=timeout_in_sec
  )
  while datetime.datetime.now() < deadline:
    # Call the core function to check if the workload is complete.
    if impl(workload_id, project_id, region, cluster_name):
      return True

    time.sleep(poke_interval_in_second)

  raise AirflowException(
      f"Timed out after {timeout_in_min}min({timeout_in_sec}s). Please adjust `Override Timeout In Minutes` in UI input."
  )


RECIPE_INSTANCE = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE
RECIPE_NAME = RECIPE_INSTANCE.value.lower()

with models.DAG(
    dag_id=RECIPE_NAME,
    start_date=datetime.datetime(2025, 1, 1),
    schedule_interval="0 10 * * *",
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
        "v5e-32",
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
    - This test requires that a dataset with the same name as the UI parameter "[BigQuery Database Dataset]".
    - Create a service account named `one-click` with the following roles: `Artifact Registry Reader`, `Kubernetes Engine Admin`, `Monitoring Viewer`.
        - Generate a new service account key and download the JSON file to retrieve its contents.
        Next, create a secret manager named `one-click-key` and store the key contents there for use when switching service accounts.
        - Make sure the default service account has the `Secret Manager Secret Accessor` role.
        ex: [PROJECT_NUMBER]-compute@developer.gserviceaccount.com
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

  start_recipe = GKEStartPodOperator(
      task_id="start_recipe",
      name=RECIPE_NAME.replace("_", "-"),
      project_id=dag_params["project"],
      cluster_name=dag_params["cluster_name"],
      location=derived_params["region"],
      namespace="default",
      hostnetwork=True,
      image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
      # TODO(b/452777428): Apply this once the "apache-airflow-providers-google" in
      # prod composer is upgraded to "16.0.0".
      # on_finish_action=OnFinishAction.DELETE_POD.value,
      get_logs=True,
      cmds=["/bin/bash", "-cxue", commands],
      container_security_context=k8s.V1SecurityContext(privileged=True),
      labels={"airflow-runtime": recipe_runtime},
  )

  # TODO(b/452777428): Remove this once the "apache-airflow-providers-google" in prod
  # composer is upgraded to "16.0.0".
  # Explicitly clean up the pod since the `on_finish_action` of
  # `GKEStartPodOperator` is not functioning.
  clean_up_start_recipe_pod = clean_up_pod.override(
      trigger_rule=TriggerRule.ALL_DONE
  )(
      cluster_name=dag_params["cluster_name"],
      region=derived_params["region"],
      project=dag_params["project"],
      airflow_runtime=recipe_runtime,
  )

  check_recipe_log = wait_workload_complete.override(
      task_id="check_recipe_log",
  )(
      workload_id=derived_params["recipe_workload_id"],
      project_id=dag_params["project"],
      region=derived_params["region"],
      cluster_name=dag_params["cluster_name"],
      benchmark_steps=dag_params["benchmark_steps"],
      override_timeout_in_min=dag_params["override_timeout_in_min"],
      poke_interval_in_second=30,
  )

  clean_up_recipe = xpk.clean_up_workload.override(
      task_id="clean_up_recipe", trigger_rule=TriggerRule.ALL_DONE
  )(
      workload_id=derived_params["recipe_workload_id"],
      project_id=dag_params["project"],
      zone=dag_params["zone"],
      cluster_name=dag_params["cluster_name"],
  )

  # TODO: Add an EmptyOperator to detect the overall state.

  (
      dag_params
      >> derived_params
      >> commands
      >> start_recipe
      >> check_recipe_log
      >> clean_up_recipe
  )
  start_recipe >> clean_up_start_recipe_pod
