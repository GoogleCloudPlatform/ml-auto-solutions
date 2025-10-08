import datetime
import random
import string
import time
from absl import logging
from airflow.decorators import task

from dags.maxtext_pathways.configs.commands import COMMAND_ENV, COMMAND_RECIPE
from xlml.utils.gke import zone_to_region


@task.python(multiple_outputs=True)
def get_parameters(**context):
  """Generate parameteres form DAG UI input."""
  # Get the DAG run"s configuration and parameters.
  params = context.get("params", {})

  # Initialization command.
  recipe_cmds = COMMAND_RECIPE

  # Generate recipe workload_id and temp_key.
  name, temp_post_fix = generate_recipe_workload_id(params)
  params["temp_key"] = temp_post_fix
  params["recipe_workload_id"] = name

  # Combine command.
  device_type = params["device_version"] + "-" + str(params["core_count"])
  params["device_type"] = device_type

  for key, value in params.items():
    if key not in [
        "time_out_in_min",
        "device_version",
        "core_count",
        "service_account",
        "recipe_workload_id",
    ]:
      if isinstance(value, int):
        recipe_cmds += f" --{key}={value}"
      else:
        recipe_cmds += f" --{key}='{value}'"

  formatted_cmds = recipe_cmds.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")

  env_cmds = COMMAND_ENV.format(service_account=params["service_account"])

  # Add parameters.
  params["commands"] = " && ".join([env_cmds, recipe_cmds])
  params["region"] = zone_to_region(params["zone"])

  return params


def generate_recipe_workload_id(params):
  """Please refer to the `generate_xpk_workload_cmd` function in the `/maxtext/benchmarks/maxtext_xpk_runner.py` file."""
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


def set_sensor_timeout(context):
  context["ti"].task.timeout = context["params"]["time_out_in_min"] * 60
