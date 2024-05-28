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

"""Utilities to construct configs for solutionsteam_flax_latest_supported DAG."""

import datetime
from typing import Tuple
import uuid
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket, test_owner
from dags.solutions_team.configs.flax import common
from dags.vm_resource import TpuVersion, Project, RuntimeVersion
import os


PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value
RUN_DATE = datetime.datetime.now().strftime("%Y_%m_%d")
GCS_SUBFOLDER_PREFIX = test_owner.Team.SOLUTIONS_TEAM.value


def get_flax_resnet_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    extraFlags: str = "",
    is_tpu_reserved: bool = True,
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_flax()

  work_dir = generate_unique_dir("/tmp/imagenet")
  run_model_cmds = (
      (
          f"export TFDS_DATA_DIR={data_dir} &&"
          " JAX_PLATFORM_NAME=TPU python3 /tmp/flax/examples/imagenet/main.py"
          " --config=/tmp/flax/examples/imagenet/configs/tpu.py"
          f" --workdir={work_dir} --config.num_epochs=1 {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name="flax_resnet_imagenet",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.SHIVA_S,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def get_flax_wmt_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    num_train_steps: int,
    data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    extraFlags: str = "",
    is_tpu_reserved: bool = True,
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=PROJECT_NAME,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_google_flax() + (
      "pip install tf-nightly-cpu",
      "pip install tensorflow-datasets",
      "pip install tensorflow-text-nightly",
      "pip install sentencepiece",
  )

  work_dir = generate_unique_dir("/tmp/wmt")
  run_model_cmds = (
      (
          f"export TFDS_DATA_DIR={data_dir} &&"
          " JAX_PLATFORM_NAME=TPU python3 /tmp/flax/examples/wmt/main.py"
          " --config=/tmp/flax/examples/wmt/configs/default.py"
          f" --workdir={work_dir} --config.num_train_steps={num_train_steps}"
          f" --config.per_device_batch_size=16 {extraFlags}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=RUNTIME_IMAGE,
          reserved=is_tpu_reserved,
      ),
      test_name="flax_wmt_translate",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.SHIVA_S,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def generate_unique_dir(prefix: str) -> str:
  """Generate a unique dir based on prefix to avoid skipping runs during retry."""
  short_id = str(uuid.uuid4())[:8]
  return f"{prefix}_{short_id}"
