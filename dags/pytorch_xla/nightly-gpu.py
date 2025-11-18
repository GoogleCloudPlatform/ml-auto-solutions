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

import datetime
from airflow.decorators import task_group
from airflow import models
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import composer_env
from dags.common.vm_resource import Project, Zone, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, V6E_SUBNETWORKS


# Run once a day at 2 pm UTC (6 am PST)
SCHEDULED_TIME = "0 16 * * *" if composer_env.is_prod_env() else None

US_CENTRAL1 = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    # HACK: use region in place of zone, since clusters are regional
    zone="us-central1",
    dataset_name=metric_config.DatasetOption.XLML_DATASET,
)


@task_group(prefix_group_id=False)
def torchvision():
  resnet_v100_2x2 = task.GpuGkeTask(
      test_config.GpuGkeTest.from_pytorch(
          "pt-nightly-resnet50-mp-fake-v100-x2x2"
      ),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()

  resnet_v100_2x2_spmd = task.GpuGkeTask(
      test_config.GpuGkeTest.from_pytorch(
          "pt-nightly-resnet50-spmd-batch-fake-v100-x2x2"
      ),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()

  resnet_v100_2x2 >> resnet_v100_2x2_spmd


with models.DAG(
    dag_id="pytorchxla-nightly-gpu",
    schedule=SCHEDULED_TIME,
    tags=[
        "pytorchxla",
        "latest",
        "supported",
        "xlml",
        "gpu",
        "nvidia-tesla-v100",
        "n1-standard-16",
    ],
    start_date=datetime.datetime(2023, 7, 12),
    catchup=False,
):
  torchvision()
