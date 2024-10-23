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

"""Utilities to construct configs for simple DAG."""

import datetime
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner
import dags.vm_resource as resource
from dags.vm_resource import  Zone, Project

GCS_SUBFOLDER_PREFIX = test_owner.Team.REPRODUCIBILITY.value


def get_simple_config(
    machine_type: resource.MachineVersion,
    image_family: resource.ImageFamily,
    accelerator_type: resource.GpuVersion,
    count: int,
    gpu_zone: resource.Zone,
    time_out_in_min: int,
    project_name: resource.Project = resource.Project.SUPERCOMPUTER_TESTING,
) -> task.GpuGkeTask:
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name.value,
      zone=gpu_zone.value,
      dataset_name=metric_config.DatasetOption.BENCHMARK_DATASET,
  )
  set_up_cmds = (
      "set +x",
      "pip install -U pip",
  )

  # installs gsutil for uploading results into GCS
  install_gsutil = (
      "curl -s -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-472.0.0-linux-x86_64.tar.gz",
      "tar -xf google-cloud-cli-472.0.0-linux-x86_64.tar.gz",
      "./google-cloud-sdk/install.sh -q",
      "source /google-cloud-sdk/path.bash.inc",
      "which gsutil",
  )

  set_lib_path = (
      "export PATH=/usr/local/nvidia/bin${PATH:+:${PATH}}",
      "export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}",
      "nvidia-smi",
  )

  command_script = ["bash", "-cxue"]
  command_script.append(
        "\n".join(
            install_gsutil + set_lib_path + set_up_cmds
        )
    )
  job_test_config = test_config.GpuGkeTest(
      accelerator=test_config.Gpu(
          machine_type=machine_type.value,
          image_family=image_family.value,
          count=count,
          accelerator_type=accelerator_type.value,
      ),
      test_name="reproducibility_simple",
      entrypoint_script=command_script,
      test_command="",
      timeout=datetime.timedelta(minutes=time_out_in_min),
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/gpt3",
  )
  project_name = Project.CLOUD_ML_AUTO_SOLUTIONS.value
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=Zone.US_CENTRAL2_B.value,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
      dataset_project=project_name,
      composer_project=project_name,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )
