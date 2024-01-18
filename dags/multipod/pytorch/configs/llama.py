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
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import vm_resource, test_owner
from urllib.parse import urlparse


XLA_CACHE_MOUNT_LOC = "/tmp/compilation_cache"


def _get_cache_mount_command(gcs_bucket_prefix: str):
  gcs_cache_path = os.path.join(gcs_bucket_prefix, "cache")
  parse_result = urlparse(gcs_cache_path)
  bucket_name = parse_result.netloc
  subpath = parse_result.path
  return f"mkdir -p {XLA_CACHE_MOUNT_LOC}; gcsfuse --only-dir {subpath} {bucket_name} {XLA_CACHE_MOUNT_LOC}"


def _get_run_clm_command(
    config_name: vm_resource.TpuVersion,
    per_slice_batch_size: int,
    block_size: int = 2048,
    tp_axis: int = 1,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    num_slices: int = 1,
    max_steps: Optional[int] = None,
    xla_execution_time_step: Optional[int] = None,
):
  command = (
      'LIBTPU_INIT_ARGS="--xla_enable_async_collective_permute=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_rwb_fusion=false --xla_jf_rematerialization_percent_shared_memory_limit=10000 --xla_tpu_enable_net_router_in_all_gather=false --xla_tpu_scheduler_percent_shared_memory_limit=95 --xla_jf_spmd_threshold_for_windowed_einsum_mib=0" '
      "python3 /tmp/transformers/examples/pytorch/language-modeling/run_clm.py "
      "--tokenizer_name hf-internal-testing/llama-tokenizer "
      f"--dataset_name {dataset_name} "
      f"--dataset_config_name {dataset_config} "
      # TODO(jonbolin): SPMD requires integration with Accelerate for
      # distributed dataloading. per_device_train_batch_size currently accepts
      # the global batch size.
      f"--per_device_train_batch_size {per_slice_batch_size * num_slices} "
      f"--save_strategy no "
      "--do_train "
      "--output_dir /tmp/output "
      f"--config_name /configs/llama/{config_name}.json "
      "--remove_unused_columns no "
      "--report_to tensorboard "
      f"--logging_dir /tmp/tensorboard "
      "--torch_dtype bfloat16 "
      "--spmd_defer_init "
      f"--spmd_2d_sharding {tp_axis} "
      f"--spmd_dcn_parallelism {num_slices} "
      "--optim adafactor "
      "--dataloader_drop_last yes "
      "--spmd_grad_chkpt "
      f"--block_size {block_size} "
      "--xla_autocast "
      f"--xla_cache_path {XLA_CACHE_MOUNT_LOC} "
      "--xla_cache_single_writer yes "
  )
  if max_steps:
    command += f" --max_steps {max_steps}"
  if xla_execution_time_step is not None:
    command += f" --xla_execution_time_step {xla_execution_time_step}"
  return command


def _move_metrics_command(gcs_path):
  # Only copy the results from worker 0 to GCS.
  return (
      "if [[ $TPU_WORKER_ID -eq 0 ]]; then "
      "gsutil cp /tmp/tensorboard/events.out.tfevents.* "
      f"{gcs_path}/events.out.tfevents; fi"
  )


def get_pytorch_llama2_perf_config(
    tpu_version: vm_resource.TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    cluster_name: str,
    cluster_project_name: str,
    docker_image: str,
    gcs_bucket_prefix: str,
    tp_axis: int = 1,
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

  # Generate an output bucket path for the specific test config.
  gcs_bucket_path = os.path.join(
      gcs_bucket_prefix, f"perf-{num_slices}xv{tpu_version.value}-{tpu_cores}"
  )

  cache_mount_command = _get_cache_mount_command(gcs_bucket_prefix)
  train_command = _get_run_clm_command(
      config_name=config_name,
      tp_axis=tp_axis,
      per_slice_batch_size=per_slice_batch_size,
      num_slices=num_slices,
      max_steps=10,
      xla_execution_time_step=5,
  )
  metrics_command = _move_metrics_command(gcs_bucket_path)
  run_model_cmds = ("set -xe", cache_mount_command, train_command, metrics_command)

  job_test_config = test_config.TpuGkeTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
      ),
      test_name=f"mp-pt-llama-{config_name.lower()}-perf",
      cluster_name=cluster_name,
      docker_image=docker_image,
      run_model_cmds=run_model_cmds,
      set_up_cmds=None,
      time_out_in_min=time_out_in_min,
      num_slices=num_slices,
      task_owner=test_owner.JON_B,
  )

  job_metric_config = metric_config.MetricConfig(
      tensorboard_summary=metric_config.SummaryConfig(
          file_location=f"{gcs_bucket_path}/events.out.tfevents",
          aggregation_strategy=metric_config.AggregationStrategy.LAST,
          include_tag_patterns=["train/step_wall_time", "train/tracing_time"],
      )
  )

  return task.TpuXpkTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      task_metric_config=job_metric_config,
  )
