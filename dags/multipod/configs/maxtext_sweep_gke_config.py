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

"""Utilities to construct job sweep GKE configs for maxtext DAG."""

import datetime
from xlml.apis import gcp_config, metric_config, task, test_config
from dags.common.vm_resource import TpuVersion, XpkClusterConfig
import itertools
from typing import List, Iterable, Dict, Any


def update_run_name_from_quantization_config(run_name, configs):
  """Updates the run name based on quantization configuration.

  Args:
    run_name: The original run name.
    configs: A dictionary of configurations.

  Returns:
    The updated run name.
  """
  quantization_value = configs.get("M_QUANTIZATION")
  if quantization_value is not None:
    run_name += "_" + (
        "bf16" if quantization_value == "" else quantization_value
    )
  return run_name


def get_maxtext_sweep_gke_config(
    test_owner: str,
    cluster: XpkClusterConfig,
    num_slices: List[int],
    sweep_params: Dict[str, List[Any]],
    time_out_in_min: int,
    run_name_prefix: str,
    docker_image: str,
    base_output_directory: str,
    base_run_model_cmds: Iterable[str],
    dataset_name: metric_config.DatasetOption = metric_config.DatasetOption.BENCHMARK_DATASET,
    metric_aggregation_strategy: metric_config.AggregationStrategy = metric_config.AggregationStrategy.MEDIAN,
    dataset_project: str = None,
    composer_project: str = None,
    enable_profile_config: bool = False,
) -> List[task.XpkTask]:
  if not dataset_project:
    dataset_project = cluster.project
  if not composer_project:
    composer_project = cluster.project

  job_gcp_config = gcp_config.GCPConfig(
      project_name=cluster.project,
      zone=cluster.zone,
      dataset_name=dataset_name,
      dataset_project=dataset_project,
      composer_project=composer_project,
  )

  # Add num_slices as a sweep param
  sweep_params["NUM_SLICES"] = num_slices

  # Convert sweep_params to a list of lists to generate sweep param combinations
  sweep_params_list = []
  for param, values in sweep_params.items():
    sweep_params_list.append([(param, val) for val in values])

  # Generate all combinations of sweep param configurations and create a XpkTask for each one
  xpk_task_list = []
  for idx, config in enumerate(itertools.product(*sweep_params_list)):
    config_dict = {key: value for (key, value) in config}

    # Remove num_slices as a sweep param after combinations have been generated
    curr_num_slices = config_dict["NUM_SLICES"]
    del config_dict["NUM_SLICES"]

    # Export sweep params as env variables for MaxText to read
    run_model_cmds = [
        f"export {key}={value}" for (key, value) in config_dict.items()
    ]
    for cmd in base_run_model_cmds:
      run_model_cmds.append(cmd)

    updated_run_name_prefix = update_run_name_from_quantization_config(
        run_name_prefix, config_dict
    )
    job_test_config = test_config.TpuGkeTest(
        test_config.Tpu(
            version=cluster.device_version,
            cores=cluster.core_count,
        ),
        test_name=f"{updated_run_name_prefix}-{idx}",
        set_up_cmds=None,
        run_model_cmds=run_model_cmds,
        timeout=datetime.timedelta(minutes=time_out_in_min),
        task_owner=test_owner,
        num_slices=curr_num_slices,
        cluster_name=cluster.name,
        docker_image=docker_image,
    )

    job_metric_config = metric_config.MetricConfig(
        tensorboard_summary=metric_config.SummaryConfig(
            file_location=base_output_directory,
            aggregation_strategy=metric_aggregation_strategy,
            use_regex_file_location=True,
        ),
    )
    if enable_profile_config:
      job_metric_config.profile = metric_config.ProfileConfig(
          file_location=base_output_directory,
      )

    xpk_task = task.XpkTask(
        task_test_config=job_test_config,
        task_gcp_config=job_gcp_config,
        task_metric_config=job_metric_config,
    )
    xpk_task_list.append(xpk_task)

  return xpk_task_list
