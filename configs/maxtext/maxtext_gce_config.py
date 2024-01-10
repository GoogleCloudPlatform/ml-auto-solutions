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

"""Utilities to construct configs for maxtext DAG."""

from typing import Tuple
import uuid
from apis import gcp_config, metric_config, task, test_config
from configs import gcs_bucket, test_owner
from configs.maxtext import common
from configs.vm_resource import TpuVersion, Project, RuntimeVersion

PROJECT_NAME = Project.CLOUD_ML_AUTO_SOLUTIONS.value
RUNTIME_IMAGE = RuntimeVersion.TPU_UBUNTU2204_BASE.value


def get_maxtext_nightly_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    num_train_epochs: int,
    project_name: str = PROJECT_NAME,
    runtime_version: str = RUNTIME_IMAGE,
    network: str = "default",
    subnetwork: str = "default",
    resolution: int = 512,
    extraFlags: str = "",
    is_tpu_reserved: bool = True,
) -> task.TpuQueuedResourceTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.download_maxtext()
  run_model_cmds = (
      (
          "cd /tmp/maxtext"
          " bash setup.sh MODE=nightly; \
            DATETIME=$(date +%Y-%m-%d-%H-%M-%S) \
            RUN_NAME=maxtext-nightly-${DATETIME} \
            JAX_PLATFORM_NAME=TPU \
            XLA_FLAGS=\"--xla_dump_to=/tmp/xla_dump/\" \
            python3 MaxText/train.py MaxText/configs/base.yml run_name=$RUN_NAME \
            base_output_directory=gs://tonyjohnchen-mxla-debug/ dataset_path=gs://max-datasets-rogue \
            dataset_type=synthetic \
            per_device_batch_size=6 reuse_example_batch=1 \
            global_parameter_scale=1 \
            metrics_file='metrics.txt' \
            steps=50 enable_checkpointing=false enable_profiler=true gcs_metrics=false;"
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
      test_name="maxtext_nightly",
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      time_out_in_min=time_out_in_min,
      task_owner=test_owner.Tony_C,
  )

  return task.TpuQueuedResourceTask(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
