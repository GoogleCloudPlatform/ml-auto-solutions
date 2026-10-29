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

import enum
import datetime

from absl import logging
from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain

from dags import composer_env
from dags.common import test_owner
from dags.common.scheduling_helper.scheduling_helper import SchedulingHelper
from dags.maxtext_pathways.configs import parameters as ui_params
from dags.maxtext_pathways.configs import recipe_config as recipe_cfg
from dags.maxtext_pathways.configs.utils import (
    get_dag_parameters,
    generate_install_dependencies_commands,
    generate_derived_parameters,
    worker_pod_interruption,
    COLOCATED_PYTHON_IMAGE,
)
from xlml.utils import gcluster, gke, kpo


@enum.unique
class ElasticType(enum.Enum):
  """Type of elastic scaling mode."""

  PAUSE_RESUME = "Pause-resume"
  REPLICA_RESIZE = "Replica-resize"


# Base elastic parameters
elastic_params = ui_params.PARAMETERS.copy()
elastic_params.update({
    "colocated_python_image": ui_params.Param(
        COLOCATED_PYTHON_IMAGE,
        type="string",
        title="Colocated Python Image",
        description="Colocated Python image for pathways.",
    ),
    "elastic_type": ui_params.Param(
        ElasticType.PAUSE_RESUME.value,
        type="string",
        title="Elastic Type",
        description="Pause-resume/Replica-resize",
        enum=[elastic_type.value for elastic_type in ElasticType],
    ),
})

# Replica resize parameter overrides
replica_params = elastic_params.copy()
replica_params.update({
    "elastic_type": ui_params.Param(
        ElasticType.REPLICA_RESIZE.value,
        type="string",
        title="Elastic Type",
        description="Pause-resume/Replica-resize",
        enum=[elastic_type.value for elastic_type in ElasticType],
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
    dag_params: dict,
    derived_params: dict,
    recipe_instance: recipe_cfg.Recipe,
    elastic_mode: ElasticType,
) -> str:
  """Generates a command string using config and derived parameters.

  Runtime modifications are made to the recipe command depending on the
  elastic mode (Pause-resume vs Replica-resize).
  """
  env_cmds = generate_install_dependencies_commands(
      cluster_name=dag_params["cluster_name"],
      location=derived_params["region"],
      project_id=dag_params["project"],
  )
  checkout_branch_cmd = (
      "git clone --depth=1 https://github.com/AI-Hypercomputer/maxtext.git "
      "/tmp/maxtext && "
      "git -C /tmp/maxtext fetch --depth=2 origin "
      "feat/benchmarks-elastic-goodput-ckpt-flags && "
      "git -C /tmp/maxtext cherry-pick --no-commit FETCH_HEAD && "
      "cp -r /tmp/maxtext/benchmarks /deps/"
  )
  recipe_cmd = recipe_instance.run_command

  # Append recipe flags
  all_params = {**dag_params, **derived_params}
  for key, value in all_params.items():
    if key in recipe_cfg.RECIPE_FLAG:
      if isinstance(value, int):
        recipe_cmd += f" --{key}={value}"
      else:
        recipe_cmd += f" --{key}='{value}'"

  # Elastic-mode specific flags
  match elastic_mode:
    case ElasticType.PAUSE_RESUME:
      recipe_cmd += " --benchmark_steps=1000"
      recipe_cmd += (
          f" --colocated_python_image='{dag_params['colocated_python_image']}'"
      )
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
      recipe_cmd += " --elastic_enabled=True"
      recipe_cmd += " --enable_checkpointing=True"
      recipe_cmd += " --async_checkpointing=True"
      recipe_cmd += " --enable_checkpoint_cloud_logger=True"
      recipe_cmd += " --checkpoint_period=100"
      recipe_cmd += " --enable_single_controller=True"
      recipe_cmd += (
          " --elastic_min_slice_count="
          f"{derived_params['elastic_min_slice_count']}"
      )
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

      recipe_cmd += " --maxtext_override='colocated_python_checkpointing=True'"

    case ElasticType.REPLICA_RESIZE:
      recipe_cmd += " --benchmark_steps=3000"
      recipe_cmd += (
          f" --proxy_flags='--virtual_slices={derived_params['topology']} "
          f"--num_elastic_slices={derived_params['num_elastic_slices']}'"
      )
      recipe_cmd += " --elastic_enabled=True"
      recipe_cmd += " --enable_single_controller=True"
      recipe_cmd += (
          " --elastic_min_slice_count="
          f"{derived_params['elastic_min_slice_count']}"
      )
      recipe_cmd += " --enable_pathways_goodput=True"
      recipe_cmd += " --enable_goodput_recording=True"
      recipe_cmd += " --goodput_upload_interval_seconds=30"
      recipe_cmd += " --monitor_goodput=True"
      recipe_cmd += " --enable_checkpointing=True"
      recipe_cmd += " --async_checkpointing=True"
      recipe_cmd += " --enable_checkpoint_cloud_logger=True"
      recipe_cmd += " --checkpoint_period=100"
      recipe_cmd += " --maxtext_override='profiler='"

  recipe_cmd += " --skip-validation"
  formatted_cmds = recipe_cmd.replace(" --", " \n  --")
  logging.info(f"\n {formatted_cmds}")

  command_steps = [env_cmds, checkout_branch_cmd, recipe_cmd]

  return " && ".join(command_steps)


def create_elastic_dag(
    dag_id: str,
    params: dict,
    doc_md: str,
    elastic_mode: ElasticType,
    recipe_instance: recipe_cfg.Recipe = recipe_cfg.Recipe.PW_MCJAX_BENCHMARK_RECIPE,
    entry_log_pattern: str | None = None,
    end_log_pattern: str | None = None,
) -> models.DAG:
  """Factory function to build and configure elastic benchmark DAGs."""
  recipe_name = recipe_instance.value.lower()
  schedule = SchedulingHelper.arrange_schedule_time(dag_id)

  with models.DAG(
      dag_id=dag_id,
      start_date=datetime.datetime(2025, 1, 1),
      schedule_interval=schedule if composer_env.is_prod_env() else None,
      catchup=False,
      default_args={"retries": 0},
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
          f"A DAG to run a MaxText {recipe_name} with elastic "
          f"{elastic_mode.value} on GKE."
      ),
      params=params,
      doc_md=doc_md,
  ) as dag:
    fetched_params = get_dag_parameters()
    calculated_params = generate_derived_parameters(fetched_params, dag_id)

    generated_cmds = generate_commands(
        fetched_params, calculated_params, recipe_instance, elastic_mode
    )

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
        clean_up_recipe,
    )

  return dag


DOC_MD_PAUSE_RESUME = """
# A DAG to run a MaxText pw_mcjax_benchmark_recipe with elastic training on GKE.

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
"""

DOC_MD_REPLICA_RESIZE = """
# A DAG to run a MaxText pw_mcjax_benchmark_recipe with elastic replica resize on GKE.

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
"""

dag_pause_resume = create_elastic_dag(
    dag_id="pw_elastic_pause_resume",
    elastic_mode=ElasticType.PAUSE_RESUME,
    params=elastic_params,
    doc_md=DOC_MD_PAUSE_RESUME,
    entry_log_pattern="live slice count: 1",
    end_log_pattern="Sufficient slices active: 1 >= 1",
)

dag_replica_resize = create_elastic_dag(
    dag_id="pw_elastic_replica_resize",
    elastic_mode=ElasticType.REPLICA_RESIZE,
    params=replica_params,
    doc_md=DOC_MD_REPLICA_RESIZE,
    entry_log_pattern="live slice count: 2",
    end_log_pattern="Sufficient slices active: 2 >= 1",
)
