# Copyright 2023 Google LLC
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

"""Utilities to construct configs for multipodteam_pytorch_llama2_nightly DAG."""

import os
from typing import Tuple, Optional
from apis import gcp_config, metric_config, task, test_config
from configs import vm_resource, test_owner
from configs.xlml.pytorch import common


def _get_run_clm_command(
    config_name: str,
    logging_dir: str,
    per_slice_batch_size: int,
    block_size: int = 2048,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_slices: int = 1,
    save_strategy: str = "no",
    chkpt_manager_path: Optional[str] = None,
    max_steps: Optional[int] = None,
    xla_execution_time_step: Optional[int] = None,
):
  command = (
      "python3 /tmp/transformers/examples/pytorch/language-modeling/run_clm.py "
      "--tokenizer_name hf-internal-testing/llama-tokenizer "
      f"--dataset_name {dataset_name} "
      f"--dataset_config_name {dataset_config} "
      # TODO(jonbolin): SPMD requires integration with Accelerate for
      # distributed dataloading. per_device_train_batch_size currently accepts
      # the global batch size.
      f"--per_device_train_batch_size {per_slice_batch_size * num_slices} "
      "--do_train "
      "--output_dir /tmp/output "
      f"--config_name /configs/llama/{config_name}.json "
      "--remove_unused_columns no "
      "--report_to tensorboard "
      f"--logging_dir {logging_dir} "
      "--spmd_defer_init "
      "--spmd_2d_sharding 1 "
      f"--spmd_dcn_parallelism {num_slices} "
      "--optim adafactor "
      "--torch_dtype float32 "
      "--dataloader_drop_last yes "
      "--spmd_grad_chkpt "
      f"--block_size {block_size} "
  )
  if chkpt_manager_path:
    command = ' '.join((command, f'--checkpoint_manager_path {chkpt_manager_path}'))
  if max_steps:
    command = ' '.join((command, f'--max_steps {max_steps}'))
  if xla_execution_time_step is not None:
    command = ' '.join((command, f'--xla_execution_time_step {xla_execution_time_step}'))
  return command


def get_pytorch_llama2_perf_config(
  tpu_version: str,
  tpu_cores: int,
  tpu_zone: str,
  cluster_name: str,
  cluster_project_name: str,
  gcs_bucket_prefix: str,
  per_slice_batch_size: int = 16,
  config_name: str = "2B",
  num_slices: int = 1,
  time_out_in_min: int = 60,
):
  job_gcp_config = gcp_config.GCPConfig(
    project_name=cluster_project_name,
    zone=tpu_zone,
    dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  gcs_bucket_path = os.path.join(gcs_bucket_prefix, f"{num_slices}xv{tpu_version}-{tpu_cores}")

  hf_setup_commands = common.set_up_hugging_face_transformers_llama2_fork()
  train_command = _get_run_clm_command(config_name=config_name, logging_dir=gcs_bucket_path, per_slice_batch_size=per_slice_batch_size, max_steps=10, xla_execution_time_step=5)
  # TODO(jonbolin): set_up_cmds isn't used for GKE workloads, put setup in
  # run_model_cmds
  run_model_cmds = (*hf_setup_commands, train_command)

  job_test_config = test_config.TpuGkeTest(
    test_config.Tpu(
      version=tpu_version,
      cores=tpu_cores,
    ),
    test_name=f"pytorch_spmd_llama2_{config_name}_perf",
    cluster_name=cluster_name,
    docker_image=vm_resource.DockerImage.PYTORCH_MULTISLICE_BASELINE.value,
    run_model_cmds=run_model_cmds,
    set_up_cmds=None,
    time_out_in_min=time_out_in_min,
    num_slices=num_slices,
  )

  job_metric_config = metric_config.MetricConfig(
    tensorboard_summary=metric_config.SummaryConfig(
      file_location=gcs_bucket_path,
      aggregation_strategy=metric_config.AggregationStrategy.LAST,
      include_tag_patterns=['step_wall_time', 'tracing_time'],
    )
  )

  return task.TpuXpkTask(
    task_test_config=job_test_config,
    task_gcp_config=job_gcp_config,
    task_metric_config=job_metric_config,
  )

