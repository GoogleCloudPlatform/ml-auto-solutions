import datetime
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.google.cloud.operators.kubernetes_engine import GKEStartPodOperator
from kubernetes.client import models as k8s

from dags.common.vm_resource import DockerImage
from dags.maxtext_pathways.configs.parameters import PARAMETERS
from dags.maxtext_pathways.configs.commands import COMMAND_DELETE_POD_STR
from dags.maxtext_pathways.utils.tasks import get_parameters, set_sensor_timeout
from xlml.utils.xpk import wait_for_workload_completion, clean_up_workload


RECIPE = "pw_mcjax_benchmark_recipe"
with DAG(
    dag_id=RECIPE,
    start_date=datetime.datetime(2025, 1, 1),
    schedule_interval=None,
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
    ],
    description=f"A DAG to run a MaxText {RECIPE} on GKE.",
    params=PARAMETERS,
    doc_md=f"""
  # A DAG to run a MaxText {RECIPE} on GKE.

  ### Description
  Specify different models and number of slices to test the MaxText {RECIPE} on different clusters.
  The DAG first generates recipe command through UI parameters, then runs the workload, waits and monitors the workload logs, and finally cleans up the workload.

  ### Prerequisites
  - This test requires an existing cluster.
  - Create a service account with the following roles: `Artifact Registry Reader`, `Kubernetes Engine Admin`, `Monitoring Viewer`.
    - Generate a new service account key and download the JSON file to retrieve its contents. Next, create a secret manager named `one-click-key` and store the key contents there for use when switching service accounts.
    - Make sure the default service account has the `Secret Manager Secret Accessor` role.
  - If you're using a service account to pull an image from a different project, you need to grant the service account the `Artifact Registry Reader` role in that project.

  ### Procedures
  An Airflow Composer environment must be created, and the required DAG code must be deployed to the associated GCS bucket.
  To initiate the recipe, the user must access the Airflow UI, locate the specific DAG, and trigger its execution.
  """,
) as dag:
  # Define task dependencies by instantiating and linking tasks.
  params = get_parameters()

  recipe_runtime = (
      RECIPE.replace("_", "-") + '-{{ execution_date.strftime("%H%M%S") }}'
  )

  start_recipe = GKEStartPodOperator(
      task_id="start_recipe",
      name=RECIPE.replace("_", "-"),
      project_id=params["project"],
      cluster_name=params["cluster_name"],
      location=params["region"],
      namespace="default",
      hostnetwork=True,
      image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
      # on_finish_action=OnFinishAction.DELETE_POD.value,  # TODO: Unable to delete the pod, may need to update the Airflow version.
      get_logs=True,
      cmds=["/bin/bash", "-cxue", params["commands"]],
      container_security_context=k8s.V1SecurityContext(privileged=True),
      labels={"airflow-runtime": recipe_runtime},
  )

  clean_up_pod = BashOperator(
      task_id="clean_up_pod",
      bash_command=(
          COMMAND_DELETE_POD_STR.format(
              cluster_name=params["cluster_name"],
              region=params["region"],
              project=params["project"],
              airflow_runtime=recipe_runtime,
          )
      ),
  )

  check_recipe_log = wait_for_workload_completion.override(
      task_id="check_recipe_log", on_execute_callback=[set_sensor_timeout]
  )(
      workload_id=params["recipe_workload_id"],
      project_id=params["project"],
      region=params["region"],
      cluster_name=params["cluster_name"],
  )

  clean_up_recipe = clean_up_workload.override(task_id="clean_up_recipe")(
      workload_id=params["recipe_workload_id"],
      project_id=params["project"],
      zone=params["zone"],
      cluster_name=params["cluster_name"],
  ).as_teardown(setups=start_recipe)

  # Set the execution order.
  (
      params
      >> start_recipe
      >> clean_up_pod
      >> check_recipe_log
      >> clean_up_recipe
  )
