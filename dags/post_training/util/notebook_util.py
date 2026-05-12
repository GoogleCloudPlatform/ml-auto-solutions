"""Utility functions for automating Jupyter notebooks in Airflow."""

import datetime
import inspect
import logging
import textwrap
import airflow
import json
import re

from airflow.decorators import task as airflow_task
from airflow.models.taskmixin import DAGNode
from airflow.models.baseoperator import chain
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.task_group import TaskGroup

from dags.common.vm_resource import (
    Project,
    RuntimeVersion,
    TpuVersion,
    V6E_GCE_NETWORK,
    V6E_GCE_SUBNETWORK,
    Zone,
)
from dags.post_training.util import test_config_util
from xlml.apis import gcp_config, gcs, metric_config, task, test_config

NOTEBOOK_CONFIG_GCS_PATH = (
    "gs://ml-auto-solutions-dag-configs/post-training/notebook_dag_configs.yaml"
)


def build_maxtext_setup_script() -> str:
  """Builds the shell script for setting up the MaxText environment on TPU VM.

  Returns:
      A shell script string that clones MaxText, installs dependencies, and
      sets up the virtual environment.
  """
  return textwrap.dedent(
      """
      set -e
      set -x

      # =======================================================================
      # Environment Setup
      # =======================================================================

      if [ ! -d "maxtext" ]; then
        git clone https://github.com/AI-Hypercomputer/maxtext.git
      fi
      cd maxtext

      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$PATH"

      uv venv --python 3.12 --seed --clear maxtext_venv
      source maxtext_venv/bin/activate

      # =======================================================================
      # MaxText Installation
      # =======================================================================

      uv pip install -e .[tpu-post-train] --resolution=lowest
      install_tpu_post_train_extra_deps

      # =======================================================================
      # Notebook Automation Tools
      # =======================================================================

      uv pip install nbconvert ipykernel papermill

      echo "Environment setup completed"
      """
  )


def _run_parameter_injection(
    notebook_path, output_path, parameters, env_params
):
  """
  Injects literal values or environment lookups into a notebook's code cells.

  This function searches for lines matching `KEY = VALUE` in the notebook and
  replaces the assignment with either a literal repr of the provided value
  or an `os.getenv` call.

  Args:
      notebook_path: Path to the source .ipynb file.
      output_path: Path where the modified .ipynb will be saved.
      parameters: Dict of {key: value} for literal injection.
      env_params: List of keys to be injected as `os.getenv("KEY")`.
  """
  with open(notebook_path, encoding="utf-8") as f:
    nb = json.load(f)

  all_keys_to_match = set(parameters.keys()) | set(env_params)
  found_keys = set()

  for cell in (c for c in nb["cells"] if c["cell_type"] == "code"):
    source = cell.get("source", [])
    if isinstance(source, str):
      lines = source.splitlines(keepends=True)
    else:
      lines = source

    new_lines = []
    for line in lines:
      # Match KEY=VALUE assignments with leading spaces and trailing comments.
      # Allow empty values and don't anchor to $ for robustness.
      match = re.match(r"^(\s*)(\w+)(\s*=\s*)([^#\n]*)(.*)", line)
      if match and (key := match.group(2)) in all_keys_to_match:
        found_keys.add(key)
        val = (
            repr(parameters[key])
            if key in parameters
            else f"os.getenv({key!r})"
        )
        # Preserve original indentation and comments
        line = f"{match.group(1)}{key}{match.group(3)}{val}{match.group(5)}\n"
        new_lines.append(line)
        continue

      new_lines.append(line)
    cell["source"] = new_lines

  injected_source = []
  if env_params:
    injected_source.append("import os\n")

  if missing := all_keys_to_match - found_keys:
    if injected_source:
      injected_source.append("\n")
    injected_source.append("# Injected missing parameters (fallback)\n")
    for key in sorted(list(missing)):
      val = (
          repr(parameters[key]) if key in parameters else f"os.getenv({key!r})"
      )
      injected_source.append(f"{key} = {val}\n")

  if injected_source:
    nb["cells"].insert(
        0,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"tags": ["injected-parameters"]},
            "outputs": [],
            "source": injected_source,
        },
    )

  with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

  print(f"Prepared notebook: {output_path}")


def build_notebook_execution_command(
    notebook_path: str,
    parameters: dict,
    maxtext_path: str = "$(pwd)",
    venv_path: str | None = None,
    env_params: dict[str, any] | None = None,
) -> str:
  """
  Builds a shell command to execute a notebook with injected parameters.

  Args:
      notebook_path: Path to the input notebook file on the TPU VM.
      parameters: Parameters to inject literally (e.g., {"BATCH_SIZE": 32}).
      maxtext_path: Root directory for execution (defaults to current dir).
      venv_path: Path to a virtualenv to activate (relative to maxtext_path).
      env_params: Parameters to pass as environment variables. The notebook
          will be modified to read these via `os.getenv`.

  Returns:
      A shell command string that sets up the env and runs the notebook.
  """
  env_params = env_params or {}

  # Construct the shell command for environment setup and notebook run
  exports = " && ".join(f"export {k}={v}" for k, v in env_params.items())
  export_prefix = f"{exports} && " if exports else ""

  venv_cmd = f"source {venv_path}/bin/activate" if venv_path else "true"
  output_nb = "/tmp/notebook_with_params.ipynb"

  env_setup_script = textwrap.dedent(
      f"""
      cd {maxtext_path}
      {venv_cmd}
      """
  )

  # Bash heredoc containing Python code to inject notebook parameters
  func_body = textwrap.dedent(inspect.getsource(_run_parameter_injection))
  call_func = textwrap.dedent(
      f"""
      _run_parameter_injection(
          {notebook_path!r},
          {output_nb!r},
          {parameters!r},
          {list(env_params.keys())!r}
      )
      """
  )

  injection_script = f"""python << 'PYEOF'\n{func_body}\n{call_func}\nPYEOF"""

  notebook_run_script = (
      f"{export_prefix}papermill {output_nb} {output_nb} --log-output"
  )

  # Verify the success message exists in the notebook's output.
  # We 'grep -v "print("' to ignore the Python source code line
  # and ensure we are matching an actual execution result.
  expected_completed_message = "Training Completed Successfully!"
  verification_script = textwrap.dedent(
      f"""
      set -o pipefail
      if ! grep "{expected_completed_message}" {output_nb} | grep -vq "print("; then
        echo "Error: Notebook did not report '{expected_completed_message}'."
        exit 1
      fi
      """
  )

  template = textwrap.dedent(
      """
      set -ex
      set -o pipefail

      # 1. Environment Setup
      {env_setup_script}

      # 2. Parameter Injection
      {injection_script}

      # 3. Notebook Execution
      {notebook_run_script}

      # 4. Success Verification
      {verification_script}

      echo "Notebook execution completed successfully"
      """
  )

  return template.format(
      env_setup_script=env_setup_script,
      injection_script=injection_script,
      notebook_run_script=notebook_run_script,
      verification_script=verification_script,
  )


def initialize_notebook_test(
    test_name: str,
    dag_name: str,
    notebook_path: str,
    set_up_script: str,
    parameters: dict[str, any],
    task_owner: str,
    tpu_version: TpuVersion,
) -> test_config.TpuVmTest:
  """Creates a TpuVmTest configuration for notebook execution."""
  notebook_execution = build_notebook_execution_command(
      notebook_path=notebook_path,
      parameters=parameters,
      maxtext_path="maxtext",
      venv_path="maxtext_venv",
  )
  return test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=8,
          runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
          reserved=False,
          network=V6E_GCE_NETWORK,
          subnetwork=V6E_GCE_SUBNETWORK,
      ),
      test_name=test_name,
      set_up_cmds=[set_up_script],
      run_model_cmds=[notebook_execution],
      timeout=datetime.timedelta(minutes=180),
      task_owner=task_owner,
      num_slices=1,
      gcs_subfolder=f"{test_config_util.DEFAULT_BUCKET}/{dag_name}",
  )


class NotebookConfig:
  """A simple container holding dynamic XComArg fields."""

  def __init__(self, config_arg: airflow.XComArg) -> None:
    self.zone = config_arg["zone"]
    self.tpu_version = config_arg["tpu_version"]


@airflow_task(multiple_outputs=True)
def load_notebook_config_from_gcs_yaml(
    gcs_path: str, dag_name: str
) -> dict[str, str]:
  """Loads and parses TPU version and zone configs from GCS yaml config."""
  config = gcs.load_yaml_from_gcs(gcs_path)
  dag_cfg = config.get("dag", {}).get(dag_name, {})

  tpu_version = dag_cfg.get("tpu_version")
  zone = dag_cfg.get("zone")

  # Validate that GCS config values correspond to valid Enum values
  def assert_is_valid_enum(value: str, enum_class) -> None:
    try:
      enum_class(value)
    except ValueError as e:
      raise ValueError(f"Config Validation Error: {e}") from e

  assert_is_valid_enum(zone, Zone)
  assert_is_valid_enum(tpu_version, TpuVersion)

  logging.info(
      f"Loaded configuration: tpu_version='{tpu_version}', zone='{zone}'."
  )

  return {"tpu_version": tpu_version, "zone": zone}


def run_training(
    config: test_config.TpuVmTest, hf_token: str, zone: str | airflow.XComArg
) -> DAGNode:
  return task.run_queued_resource_test(
      task_test_config=config,
      task_gcp_config=gcp_config.GCPConfig(
          project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
          zone=zone,
          dataset_name=metric_config.DatasetOption.XLML_DATASET,
      ),
      skip_post_process=True,
      custom_env={"HF_TOKEN": hf_token},
  )


def run_notebook_tests(
    dag_name: str,
    task_id_prefix: str,
    notebook_path: str,
    set_up_script: str,
    parameters: dict[str, any],
    task_owner: str,
    hf_token: str,
    config: NotebookConfig,
    previous_task: DAGNode | None = None,
) -> TaskGroup:
  """Creates and chains branched notebook tests for all TPU versions.

  Args:
    dag_name: Name of the DAG.
    task_id_prefix: Prefix for task and operator IDs (e.g. "rl_grpo" or "sft").
    notebook_path: Path to the notebook to run.
    set_up_script: Setup script for MaxText environment.
    parameters: Dict of parameters to inject in the notebook.
    task_owner: Owner of the task.
    hf_token: HuggingFace access token.
    config: A `NotebookConfig` wrapper containing zone and tpu_version.
    previous_task: Optional task/DAGNode to chain *before* the branches.

  Returns:
    A TaskGroup representing the entire branched notebook test workflow.
  """
  with TaskGroup(
      group_id=f"{task_id_prefix}_tests", prefix_group_id=False
  ) as group:
    # 1. Initialize and create the V5E test and task group
    notebook_test_v5e = initialize_notebook_test(
        test_name=f"{dag_name}_{task_id_prefix}",
        dag_name=dag_name,
        notebook_path=notebook_path,
        set_up_script=set_up_script,
        parameters=parameters,
        task_owner=task_owner,
        tpu_version=TpuVersion.V5E,
    )
    run_task_v5e = run_training(notebook_test_v5e, hf_token, zone=config.zone)

    # 2. Initialize and create the TRILLIUM/V6E test and task group
    notebook_test_v6e = initialize_notebook_test(
        test_name=f"{dag_name}_{task_id_prefix}",
        dag_name=dag_name,
        notebook_path=notebook_path,
        set_up_script=set_up_script,
        parameters=parameters,
        task_owner=task_owner,
        tpu_version=TpuVersion.TRILLIUM,
    )
    run_task_v6e = run_training(notebook_test_v6e, hf_token, zone=config.zone)

    # 3. Create skipped fallback empty operator task
    skipped = EmptyOperator(task_id=f"skipped_{task_id_prefix}")

    # 4. Define central Task-decorated Branch Operator accepting dynamic parameters
    @airflow_task.branch(
        task_id=f"task_path_decider_{task_id_prefix}",
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )
    def task_path_decider(tpu_version: str) -> str:
      logging.info(f"Configured active TPU version: '{tpu_version}'")
      active_tpu_version = TpuVersion(tpu_version)

      match active_tpu_version:
        case TpuVersion.V5E:
          decided_task_id = run_task_v5e.group_id
        case TpuVersion.TRILLIUM:
          decided_task_id = run_task_v6e.group_id
        case _:
          decided_task_id = skipped.task_id

      logging.info(f"running task_id: {decided_task_id}")
      return decided_task_id

    # 5. Instantiate branch decider task, passing XComArg directly for Airflow auto-resolution
    task_decider = task_path_decider(tpu_version=config.tpu_version)

    # 6. Chain previous tasks to the decider if exist
    if previous_task:
      chain(previous_task, task_decider)

    # 7. Connect branch decider to target branches
    chain(task_decider, [run_task_v5e, run_task_v6e, skipped])

  return group
