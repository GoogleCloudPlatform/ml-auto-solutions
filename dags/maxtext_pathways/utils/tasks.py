import datetime
from absl import logging
from airflow.decorators import task
from airflow.operators.python import get_current_context

from dags.maxtext_pathways.configs.commands import ENV_COMMAND, RECIPE_COMMAND
from xlml.utils.gke import zone_to_region


@task.python(multiple_outputs=True)
def get_parameters():
  """Combine preset commands and UI parameters."""
  # Get the DAG run"s configuration and parameters.
  context = get_current_context()
  params = context.get("params", {})

  # Initialization command.
  recipe_cmds = RECIPE_COMMAND

  # Combine command.
  device_type = params["device_version"] + "-" + str(params["core_count"])
  params["device_type"] = device_type

  for key, value in params.items():
    if key not in ["time_out_in_min", "device_version", "core_count", "service_account", "gcs_service_account_key_path"]:
      if isinstance(value, int):
        recipe_cmds += f" --{key}={value}"
      else:
        recipe_cmds += f" --{key}='{value}'"

  formatted_cmds = recipe_cmds.replace(' --', ' \n  --')
  logging.info(f"\n {formatted_cmds}")

  env_cmds = ENV_COMMAND.format(service_account = params["service_account"])

  # Add parameters.
  params["commands"] = " && ".join([env_cmds, recipe_cmds])
  params["region"] = zone_to_region(params["zone"])
  params["time_out_in_sec"] = datetime.timedelta(minutes=params["time_out_in_min"]).total_seconds() # TODO: add it to wait_for_workload_completion

  return params
