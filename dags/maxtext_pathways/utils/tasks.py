import os
import tempfile
from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from airflow.hooks.subprocess import SubprocessHook
from airflow.operators.python import get_current_context
from dags.common.vm_resource import DockerImage
from dags.maxtext_pathways.configs.commands import ENV_COMMAND, RECIPE_COMMAND
from xlml.utils.gke import zone_to_region
from xlml.utils.xpk import _get_core_api_client, _list_workload_pods, _get_batch_api_client, _get_workload_job, LOGGING_URL_FORMAT, MAIN_BRANCH


@task
def build_recipe_command():
  """Combine preset commands and UI parameters."""
  # Get the DAG run"s configuration and parameters.
  context = get_current_context()
  params = context.get("params", {})

  # Initialization command.
  recipe_cmds = RECIPE_COMMAND

  # Combine command.
  device_type = params["device_version"] + "-" + str(params["core_count"])
  recipe_cmds += f" --device_type={device_type}"

  for key, value in params.items():
    if key not in ["time_out_in_min", "device_version", "core_count"]:
      if isinstance(value, int):
        recipe_cmds += f" --{key}={value}"
      else:
        recipe_cmds += f" --{key}='{value}'"
        
  logging.info("\n" + recipe_cmds.replace(" --", " \n  --"))
  
  return recipe_cmds

@task
def run_workload(recipe_cmds):
  """Run workload through xpk tool."""

  context = get_current_context()
  params = context.get("params", {})

  container_workload_cmds = " && ".join([ENV_COMMAND, recipe_cmds])

  with tempfile.TemporaryDirectory() as tmpdir:
    xpk_cmds = [
        "set -xue",
        f"git clone --branch {MAIN_BRANCH} https://github.com/AI-Hypercomputer/xpk"
        f" {tmpdir}/xpk",
        "pip install ruamel.yaml docker",
    ]

    workload_create_cmds = [
        f"python {tmpdir}/xpk/xpk.py workload create"
        f" --cluster={params['cluster_name']} --workload={params['user']}-workload"
        f" --command='{container_workload_cmds}' --device-type={params['device_version']}-{str(params['core_count'])}"
        f" --num-slices=1 --docker-image={DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value}"
        f" --project={params['project']} --zone={params['zone']}"
        " --restart-on-user-code-failure",
    ]

    cmds = []
    cmds.extend(xpk_cmds)
    cmds.extend(workload_create_cmds)

    logging.info(f"{cmds = }")
    logging.info(f"{xpk_cmds = }")
    logging.info(f"{workload_create_cmds = }")
    logging.info(f"{container_workload_cmds = }")
    logging.info(f"{ENV_COMMAND = }")
    logging.info(f"{recipe_cmds = }")

    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={**os.environ, "KUBECONFIG": os.path.join(tmpdir, "xpk.conf")},
    )
    assert (
        result.exit_code == 0
    ), f"XPK command failed with code {result.exit_code}"

@task.sensor(poke_interval=60, timeout=3600, mode="reschedule")
def wait_for_workload_completion() -> bool:
  """Check the workload status."""

  context = get_current_context()
  params = context.get("params", {})

  workload_id = f"{params['user']}-workload"
  project_id = params["project"]
  region = zone_to_region(params["zone"])
  cluster_name = params["cluster_name"]

  context["task"].timeout = params["time_out_in_min"] * 60 # change to second

  core_api = _get_core_api_client(project_id, region, cluster_name)
  pods = _list_workload_pods(core_api, workload_id)

  if not pods.items:
    logging.info(f"No pods found for workload selector: {workload_id}.")

    # Pathways jobs delete all pods on failure so we must also check if the job
    # is complete
    batch_api = _get_batch_api_client(project_id, region, cluster_name)
    job = _get_workload_job(batch_api, workload_id)
    if job is None:
      logging.info(
          f"No pods or jobs were found for workload selector: {workload_id}"
      )
      return False

    if any(condition.type == "Failed" for condition in job.status.conditions):
      # Don"t keep retrying if the job has failed
      raise AirflowFailException("Job has condition type: 'Failed'")

    if any(condition.type == "Complete" for condition in job.status.conditions):
      logging.info(
          "No pods found but job is complete for workload selector:"
          f" {workload_id}"
      )
      return True

    return False

  if any(pod.status.phase in ["Pending", "Running"] for pod in pods.items):
    logging.info("At least one pod has yet to complete.")
    return False

  try:
    for pod in pods.items:
      if pod.status.phase == "Failed":
        # Don"t keep retrying if the pod has failed
        raise AirflowFailException(f"Bad pod phase: {pod.status.phase}")
      elif pod.status.phase in ["Unknown"]:
        raise RuntimeError(f"Bad pod phase: {pod.status.phase}")
  finally:
    # TODO(jonbolin): log printing for GPUs, which have multiple containers
    if len(pod.spec.containers) == 1:
      # Print the logs of the last pod checked - either the first failed pod or
      # the last successful one.
      logs = core_api.read_namespaced_pod_log(
          name=pod.metadata.name, namespace=pod.metadata.namespace
      )
      logging.info(f"Logs for pod {pod.metadata.name}:")
      for line in logs.split("\n"):
        logging.info(line)
    url = LOGGING_URL_FORMAT.format(
        project=project_id,
        region=region,
        cluster=cluster_name,
        workload_id=workload_id,
    )
    logging.info(f"Link to logs: {url}")

  logging.info("All pod(s) phase are succeeded.")
  return True

@task(trigger_rule="all_done")
def clean_up_workload():
  """Delete workload."""

  context = get_current_context()
  params = context.get("params", {})

  with tempfile.TemporaryDirectory() as tmpdir:
    xpk_cmds = [
        "set -xue",
        f"git clone --branch {MAIN_BRANCH} https://github.com/AI-Hypercomputer/xpk"
        f" {tmpdir}/xpk",
        "pip install ruamel.yaml docker",
    ]

    workload_delete_cmd = [
        f"python {tmpdir}/xpk/xpk.py workload delete"
        f" --cluster={params['cluster_name']} --workload={params['user']}-workload"
        f" --project={params['project']} --zone={params['zone']}",
    ]

    cmds = []
    cmds.extend(xpk_cmds)
    cmds.extend(workload_delete_cmd)

    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={**os.environ, "KUBECONFIG": os.path.join(tmpdir, "xpk.conf")},
    )
    assert (
        result.exit_code == 0
    ), f"XPK clean-up failed with code {result.exit_code}"
