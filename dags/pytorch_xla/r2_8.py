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

import datetime
from airflow.decorators import task_group
from airflow import models
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import composer_env
from dags.common.vm_resource import Project, Zone, V5_NETWORKS, V5E_SUBNETWORKS, V5P_SUBNETWORKS, V6E_SUBNETWORKS, BM_NETWORKS, V5P_BM_SUBNETWORKS


# Run once a day at 2 pm UTC (6 am PST)
SCHEDULED_TIME = None
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

US_EAST5_A_TPU_PROD_ENV_AUTOMATED = gcp_config.GCPConfig(
    Project.TPU_PROD_ENV_AUTOMATED.value,
    Zone.US_EAST5_A.value,
    metric_config.DatasetOption.XLML_DATASET,
)


US_EAST5_B_CLOUD_ML_BENCHMARKING = gcp_config.GCPConfig(
    Project.CLOUD_ML_BENCHMARKING.value,
    Zone.US_EAST5_B.value,
    metric_config.DatasetOption.XLML_DATASET,
)


@task_group(prefix_group_id=False)
def torchvision():
  mnist_v2_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-mnist-pjrt-func-v2-8-1vm"
      ),
      US_CENTRAL1_C,
  )
  resnet_v2_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-resnet50-pjrt-fake-v2-8-1vm",
          reserved=True,
      ),
      US_CENTRAL1_C,
  )
  resnet_v3_8_tests = [
      task.run_queued_resource_test(
          test_config.JSonnetTpuVmTest.from_pytorch(test, reserved=True),
          US_EAST1_D,
      )
      for test in (
          "pt-2-7-resnet50-pjrt-fake-v3-8-1vm",
          "pt-2-7-resnet50-pjrt-ddp-fake-v3-8-1vm",
      )
  ]
  resnet_v4_8_tests = [
      task.run_queued_resource_test(
          test_config.JSonnetTpuVmTest.from_pytorch(test),
          US_CENTRAL2_B,
      )
      for test in (
          "pt-2-7-resnet50-pjrt-fake-v4-8-1vm",
          "pt-2-7-resnet50-pjrt-ddp-fake-v4-8-1vm",
          "pt-2-7-resnet50-spmd-batch-fake-v4-8-1vm",
          "pt-2-7-resnet50-spmd-spatial-fake-v4-8-1vm",
      )
  ]
  resnet_v4_32 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-resnet50-pjrt-fake-v4-32-1vm"
      ),
      US_CENTRAL2_B,
  )
  resnet_v5lp_4 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-resnet50-pjrt-fake-v5litepod-4-1vm",
          network=V5_NETWORKS,
          subnetwork=V5E_SUBNETWORKS,
          reserved=True,
      ),
      US_EAST1_C,
  )

  mnist_v2_8 >> (resnet_v2_8, *resnet_v4_8_tests, resnet_v4_32, resnet_v5lp_4)
  resnet_v2_8 >> resnet_v3_8_tests

  resnet_v100_2x2 = task.GpuGkeTask(
      test_config.GpuGkeTest.from_pytorch("pt-2-7-resnet50-mp-fake-v100-x2x2"),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()
  resnet_v100_2x2_spmd = task.GpuGkeTask(
      test_config.GpuGkeTest.from_pytorch(
          "pt-2-7-resnet50-spmd-batch-fake-v100-x2x2"
      ),
      US_CENTRAL1,
      "gpu-uc1",
  ).run()
  resnet_v100_2x2 >> resnet_v100_2x2_spmd


@task_group(prefix_group_id=False)
def huggingface():
  accelerate_v2_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-accelerate-smoke-v2-8-1vm", reserved=True
      ),
      US_CENTRAL1_C,
  )
  accelerate_v4_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-accelerate-smoke-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  )

  accelerate_v4_8 >> accelerate_v2_8

  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-hf-bert-pjrt-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  )

  # Stable Diffusion 2
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-stable-diffusion-2-train-func-v6e-4-1vm",
          network=BM_NETWORKS,
          subnetwork=V5P_BM_SUBNETWORKS,
      ),
      US_EAST5_B_CLOUD_ML_BENCHMARKING,
  )
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-stable-diffusion-2-train-func-v5p-8-1vm",
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )
  task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-stable-diffusion-2-train-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  )


@task_group(prefix_group_id=False)
def llama():
  llama_inference_v4_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-llama2-infer-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  )
  llama_train_v4_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-llama2-train-spmd-func-v4-8-1vm"
      ),
      US_CENTRAL2_B,
  )
  llama_2_inference_v5_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-llama2-infer-func-v5p-8-1vm",
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )
  llama_2_train_v5p_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-llama2-train-spmd-func-v5p-8-1vm",
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )
  llama_3_train_trillium = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-llama3-train-func-v6e-4-1vm",
          network=BM_NETWORKS,
          subnetwork=V5P_BM_SUBNETWORKS,
      ),
      US_EAST5_B_CLOUD_ML_BENCHMARKING,
  )
  llama_3_train_v5p_2_slices = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-llama3-train-2-slice-func-v5p-8-1vm",
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
          num_slices=2,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )
  llama_3_train_v5p_8 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-llama3-train-func-v5p-8-1vm",
          reserved=True,
          network=V5_NETWORKS,
          subnetwork=V5P_SUBNETWORKS,
      ),
      US_EAST5_A_TPU_PROD_ENV_AUTOMATED,
  )


with models.DAG(
    dag_id="pytorchxla-r2-7",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "r2-7", "supported", "xlml"],
    start_date=datetime.datetime(2023, 7, 12),
    catchup=False,
):
  torchvision()
  huggingface()
  llama()

  ci_v5lp_4 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-ci-func-v5litepod-4-1vm",
          network=V5_NETWORKS,
          subnetwork=V5E_SUBNETWORKS,
          reserved=True,
      ),
      US_EAST1_C,
  )

  ci_trillium_4 = task.run_queued_resource_test(
      test_config.JSonnetTpuVmTest.from_pytorch(
          "pt-2-7-ci-func-v6e-4-1vm",
          network=BM_NETWORKS,
          subnetwork=V5P_BM_SUBNETWORKS,
      ),
      US_EAST5_B_CLOUD_ML_BENCHMARKING,
  )
