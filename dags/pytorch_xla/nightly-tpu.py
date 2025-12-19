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

from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import composer_env
from dags.common.vm_resource import Project, Zone, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, V6E_SUBNETWORKS


# Run once a day at 2 pm UTC (6 am PST)
SCHEDULED_TIME = "15 9 * * *" if composer_env.is_prod_env() else None

US_CENTRAL1_C = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    Zone.US_CENTRAL1_C.value,
    metric_config.DatasetOption.XLML_DATASET,
)

US_CENTRAL2_B = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    Zone.US_CENTRAL2_B.value,
    metric_config.DatasetOption.XLML_DATASET,
)

US_EAST1_D = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    Zone.US_EAST1_D.value,
    metric_config.DatasetOption.XLML_DATASET,
)

US_EAST5_A_TPU_PROD_ENV_AUTOMATED = gcp_config.GCPConfig(
    Project.TPU_PROD_ENV_AUTOMATED.value,
    Zone.US_EAST5_A.value,
    metric_config.DatasetOption.XLML_DATASET,
)

US_CENTRAL1 = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    # HACK: use region in place of zone, since clusters are regional
    zone="us-central1",
    dataset_name=metric_config.DatasetOption.XLML_DATASET,
)

US_EAST1_C = gcp_config.GCPConfig(
    project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
    zone=Zone.US_EAST1_C.value,
    dataset_name=metric_config.DatasetOption.XLML_DATASET,
)

US_CENTRAL2_B_TPU_PROD_ENV = gcp_config.GCPConfig(
    project_name=Project.TPU_PROD_ENV_AUTOMATED.value,
    zone=Zone.US_CENTRAL2_B.value,
    dataset_name=metric_config.DatasetOption.XLML_DATASET,
)


@task_group(prefix_group_id=False)
def torchvision():
  mnist_v2_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-mnist-pjrt-func-v2-8-1vm",
          test_owner.BHAVYA_B,
      ),
      US_CENTRAL1_C,
  )
  resnet_v2_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v2-8-1vm",
          test_owner.BHAVYA_B,
          reserved=True,
      ),
      US_CENTRAL1_C,
  )
  resnet_v3_8_tests = [
      task.run_queued_resource_test(
          test_config.JSonnetTpuVmTest.from_pytorch(
              test, test_owner.BHAVYA_B, reserved=True
          ),
          US_EAST1_D,
      )
      for test in (
          "pt-nightly-resnet50-pjrt-fake-v3-8-1vm",
          "pt-nightly-resnet50-pjrt-ddp-fake-v3-8-1vm",
      )
  ]
  resnet_v4_8_tests = [
      task.run_queued_resource_test(
          test_config.JSonnetTpuVmTest.from_pytorch(
              test,
              test_owner.BHAVYA_B,
          ),
          US_CENTRAL2_B,
      )
      for test in (
          "pt-nightly-resnet50-pjrt-fake-v4-8-1vm",
          "pt-nightly-resnet50-pjrt-ddp-fake-v4-8-1vm",
          "pt-nightly-resnet50-spmd-batch-fake-v4-8-1vm",
          "pt-nightly-resnet50-spmd-spatial-fake-v4-8-1vm",
      )
  ]
  resnet_v4_32 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v4-32-1vm",
          test_owner.BHAVYA_B,
      ),
      US_CENTRAL2_B,
  )
  resnet_v5lp_4 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-resnet50-pjrt-fake-v5litepod-4-1vm",
          test_owner.BHAVYA_B,
          network=V5_NETWORKS,
          subnetwork=V5E_SUBNETWORKS,
          reserved=True,
      ),
      US_EAST1_C,
  )

  mnist_v2_8 >> (resnet_v2_8, *resnet_v4_8_tests, resnet_v4_32, resnet_v5lp_4)
  resnet_v2_8 >> resnet_v3_8_tests


@task_group(prefix_group_id=False)
def huggingface():
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-accelerate-smoke-v2-8-1vm",
          test_owner.BHAVYA_B,
          reserved=True,
      ),
      US_CENTRAL1_C,
  )
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-accelerate-smoke-v4-8-1vm",
          test_owner.BHAVYA_B,
      ),
      US_CENTRAL2_B,
  )

  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-hf-bert-pjrt-func-v4-8-1vm",
          test_owner.BHAVYA_B,
      ),
      US_CENTRAL2_B,
  )

  # Stable Diffusion 2
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-stable-diffusion-2-train-func-v6e-4-1vm",
          test_owner.BHAVYA_B,
          network=V5_NETWORKS,
          subnetwork=V6E_SUBNETWORKS,
      ),
      US_CENTRAL2_B_TPU_PROD_ENV,
  )
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-stable-diffusion-2-train-func-v5p-8-1vm",
          test_owner.BHAVYA_B,
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-stable-diffusion-2-train-func-v4-8-1vm",
          test_owner.BHAVYA_B,
      ),
      US_CENTRAL2_B,
  )


@task_group(prefix_group_id=False)
def llama():
  llama_inference_v4_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama2-infer-func-v4-8-1vm",
          test_owner.BHAVYA_B,
      ),
      US_CENTRAL2_B,
  )
  llama_train_v4_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama2-train-spmd-func-v4-8-1vm",
          test_owner.BHAVYA_B,
      ),
      US_CENTRAL2_B,
  )
  llama_3_train_trillium = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama3-train-func-v6e-4-1vm",
          test_owner.BHAVYA_B,
          network=V5_NETWORKS,
          subnetwork=V6E_SUBNETWORKS,
      ),
      US_CENTRAL2_B_TPU_PROD_ENV,
  )
  llama_3_train_v5p_2_slices = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama3-train-2-slice-func-v5p-8-1vm",
          test_owner.BHAVYA_B,
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
          num_slices=2,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )
  llama_3_train_v5p_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-llama3-train-func-v5p-8-1vm",
          test_owner.BHAVYA_B,
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )


with models.DAG(
    dag_id="pytorchxla-nightly-tpu",
    schedule=SCHEDULED_TIME,
    tags=[
        "pytorchxla",
        "latest",
        "supported",
        "xlml",
        "TPU",
        "v2-8",
        "v3-8",
        "v4-8",
        "v5e-4",
        "v5p-8",
        "v6e-4",
    ],
    start_date=datetime.datetime(2023, 7, 12),
    catchup=False,
):
  torchvision()
  huggingface()
  llama()

  ci_v5lp_4 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-ci-func-v5litepod-4-1vm",
          test_owner.BHAVYA_B,
          network=V5_NETWORKS,
          subnetwork=V5E_SUBNETWORKS,
          reserved=True,
      ),
      US_EAST1_C,
  )

  ci_trillium_4 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-ci-func-v6e-4-1vm",
          test_owner.BHAVYA_B,
          network=V5_NETWORKS,
          subnetwork=V6E_SUBNETWORKS,
      ),
      US_CENTRAL2_B_TPU_PROD_ENV,
  )

  ci_v5p_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-nightly-ci-func-v5p-8-1vm",
          test_owner.BHAVYA_B,
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )
