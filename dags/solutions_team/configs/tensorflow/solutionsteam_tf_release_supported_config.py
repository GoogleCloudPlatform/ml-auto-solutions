# Copyright 2024 Google LLC
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

"""Utilities to construct configs for solutionsteam_tf_nightly_supported DAG."""

from __future__ import annotations

import datetime
from datetime import date
import time
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket, test_owner
from dags.solutions_team.configs.tensorflow import common
from airflow.models import Variable
from dags.vm_resource import TpuVersion, Project, RuntimeVersion


MAJOR_VERSION = "2"
MINOR_VERSION = "17"
PATCH_VERSION = "0"
LIBTPU_VERSION = "1.11.0"
KERAS_VERSION = "2.17.0rc0"
MODELS_BRANCH = "r2.17.0"

GS_VERSION_STR = f"tf-{MAJOR_VERSION}-{MINOR_VERSION}-{PATCH_VERSION}"


def get_tf_resnet_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    runtime_version: str,
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    network: str = "default",
    subnetwork: str = "default",
    is_pod: bool = False,
    is_pjrt: bool = True,
    imagenet_dir: str = gcs_bucket.IMAGENET_DIR,
    tfds_data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    global_batch_size: int = 2048,
    train_steps: int = 320,
    validation_interval: int = 320,
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_tensorflow_models(MODELS_BRANCH, KERAS_VERSION)
  set_up_cmds += common.install_tf(
      MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION, LIBTPU_VERSION
  )
  if is_pod:
    if not is_pjrt:
      set_up_cmds += common.set_up_se(
          MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION
      )
    else:
      set_up_cmds += common.set_up_pjrt(
          MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION
      )

  global_batch_size = 128 * tpu_cores
  params_override = {
      "runtime": {"distribution_strategy": "tpu"},
      "task": {
          "train_data": {
              "input_path": imagenet_dir + "/train*",
              "tfds_data_dir": tfds_data_dir,
              "global_batch_size": global_batch_size,
          },
          "validation_data": {
              "input_path": imagenet_dir + "/valid*",
              "tfds_data_dir": tfds_data_dir,
              "global_batch_size": global_batch_size,
          },
      },
      "trainer": {
          "train_steps": train_steps,
          "validation_interval": validation_interval,
      },
  }

  test_name = f"tf_{MAJOR_VERSION}_{MINOR_VERSION}_resnet_imagenet"
  benchmark_id = f"{test_name}-v{tpu_version.value}-{tpu_cores}"
  # Add default_var to pass DAG check
  # TODO(ranran): replace Variable.get() to XCOM when it applies
  tpu_name = Variable.get(benchmark_id, default_var=None) if is_pod else "local"
  env_variable = common.export_env_variables(tpu_name, is_pod, is_pjrt)
  run_model_cmds = (
      "sudo chmod -R 777 /tmp/",
      (
          f"cd /usr/share/tpu/models && {env_variable} &&"
          " python3 official/vision/train.py"
          f" --experiment=resnet_imagenet"
          f" --mode=train_and_eval --model_dir=/tmp"
          f" --params_override='{params_override}'"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.CHANDRA_D,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      tpu_name_env_var=is_pod,
      all_workers=not is_pod,
  )


def get_tf_dlrm_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    bottom_mlp: list[int],
    embedding_dim: int,
    train_steps: int,
    runtime_version: str,
    extraFlags: str = "",
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    is_pod: bool = False,
    is_pjrt: bool = True,
    criteo_dir: str = gcs_bucket.CRITEO_DIR,
    network: str = "default",
    subnetwork: str = "default",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  # Add default_var to pass DAG check
  # TODO(ranran): replace Variable.get() to XCOM when it applies
  test_name = "tf_dlrm_criteo"
  benchmark_id = f"{test_name}-v{tpu_version.value}-{tpu_cores}"
  tpu_name = Variable.get(benchmark_id, default_var=None) if is_pod else "local"
  is_v5p = tpu_version == TpuVersion.V5P
  env_variable = common.export_env_variables(
      tpu_name, is_pod, is_pjrt, is_v5p_sc=is_v5p
  )

  set_up_cmds = common.set_up_tensorflow_models(MODELS_BRANCH, KERAS_VERSION)
  set_up_cmds += common.install_tf(
      MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION, LIBTPU_VERSION
  )
  if is_pod:
    if not is_pjrt:
      set_up_cmds += common.set_up_se(
          MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION
      )
    else:
      set_up_cmds += common.set_up_pjrt(
          MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION
      )

  params_override = {
      "runtime": {"distribution_strategy": "tpu"},
      "task": {
          "train_data": {
              "input_path": criteo_dir + "/train/*",
              "global_batch_size": 16384,
          },
          "validation_data": {
              "input_path": criteo_dir + "/eval/*",
              "global_batch_size": 16384,
          },
          "model": {
              "interaction": "dot",
              "num_dense_features": 13,
              "bottom_mlp": bottom_mlp,
              "embedding_dim": embedding_dim,
              "top_mlp": [1024, 1024, 512, 256, 1],
              "vocab_sizes": [
                  39884406,
                  39043,
                  17289,
                  7420,
                  20263,
                  3,
                  7120,
                  1543,
                  63,
                  38532951,
                  2953546,
                  403346,
                  10,
                  2208,
                  11938,
                  155,
                  4,
                  976,
                  14,
                  39979771,
                  25641295,
                  39664984,
                  585935,
                  12972,
                  108,
                  36,
              ],
          },
      },
      "trainer": {
          "use_orbit": "true",
          "validation_interval": 90000,
          "checkpoint_interval": 270000,
          "validation_steps": 5440,
          "train_steps": train_steps,
          "optimizer_config": {
              "embedding_optimizer": "SGD",
              "lr_config": {
                  "decay_exp": 1.6,
                  "decay_start_steps": 150000,
                  "decay_steps": 136054,
                  "learning_rate": 30,
                  "warmup_steps": 8000,
              },
          },
      },
  }

  model_dir = "/tmp"

  params_override["trainer"]["pipeline_sparse_and_dense_execution"] = "true"
  tpu_id = Variable.get(benchmark_id, default_var=None)
  # TODO (ericlefort): Replace the model_dir with this line when the var is available
  # model_dir = metric_config.SshEnvVars.GCS_OUTPUT.value + f"/dlrm/v5p/{benchmark_id}"
  model_dir = f"{gcs_bucket.BASE_OUTPUT_DIR}/{test_owner.Team.SOLUTIONS_TEAM.value}/dlrm/v5p/{benchmark_id}"

  # Clean out the prior checkpoint if it exists
  run_model_cmds = (
      (
          f"cd /usr/share/tpu/models && {env_variable} &&"
          " python3 official/recommendation/ranking/train.py"
          f" --model_dir={model_dir} {extraFlags}"
          f" --params_override='{params_override}'"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.CHANDRA_D,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      tpu_name_env_var=is_pod,
      all_workers=not is_pod,
  )
