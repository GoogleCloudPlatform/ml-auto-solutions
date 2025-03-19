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

"""A DAG to run all PyTorch/XLA tests with nightly version."""

from airflow import models
import datetime
from dags import composer_env
from dags.pytorch_xla.configs import pytorchxla_torchbench_config as config
import dags.common.vm_resource as resource

# Schudule the job to run everyday at 3:00AM PST (11:00AM UTC).
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None

def run_test_code_on_persistent_TPUVM():
  gcloud_command = (
      f"gcloud compute tpus tpu-vm ssh manfei-2025-v6e-4 --zone=us-east5-b --project=cloud-ml-benchmarking --ssh-flag='-4 -L 6009:localhost:6009'",
  )
  return gcloud_command


def make_sure_docker_container_cleaned_on_persistent_TPUVM():
  gcloud_command = (
      f"gcloud compute tpus tpu-vm ssh manfei-2025-v6e-4 --zone=us-east5-b --project=cloud-ml-benchmarking --ssh-flag='-t -4 -L 6009:localhost:6009' --worker=all --command=\"sudo docker stop testooo && sudo docker rm testooo\"",
  )
  return gcloud_command


@task
def run_on_v6e_4_persistant_TPUVM():
  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                run_test_code_on_persistent_TPUVM()
                + make_sure_docker_container_cleaned_on_persistent_TPUVM()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"


with models.DAG(
    dag_id="pytorchxla-vllm-nightly",
    schedule=SCHEDULED_TIME,
    tags=["pytorchxla", "nightly", "torchbench"],
    start_date=datetime.datetime(2024, 1, 1),
    catchup=False,
) as dag:
    # follow example in https://github.com/GoogleCloudPlatform/ml-auto-solutions/blob/bda4d59ed7fd9dd3b244a8b2612385c4f5c9a8a9/dags/multipod/maxtext_gpu_end_to_end.py#L41
    run_on_v6e_4_persistant_TPUVM()

    
  # # Running on V6E
  # config.get_torchbench_tpu_config(
  #     tpu_version=resource.TpuVersion.TRILLIUM,
  #     tpu_cores=8,
  #     project=resource.Project.CLOUD_ML_BENCHMARKING,
  #     tpu_zone=resource.Zone.US_CENTRAL2_B,
  #     runtime_version=resource.RuntimeVersion.V2_ALPHA_TPUV6,
  #     network=resource.BM_NETWORKS,
  #     subnetwork=resource.V4_BM_SUBNETWORKS,
  #     time_out_in_min=1600,
  #     model_name=model,
  #     reserved=False,
  #     preemptible=False,
  #     extraFlags=" ".join(torchbench_extra_flags),
  # )


# # merge all PyTorch/XLA tests ino one Dag
# with models.DAG(
#   dag_id="pytorch_xla_model_regression_test_on_trillium",
#   schedule="0 0 * * *",  # everyday at midnight # job["schedule"],
#   tags=["mantaray", "pytorchxla", "xlml"],
#   start_date=datetime.datetime(2024, 4, 22),
#   catchup=False,
# ) as dag:
# for workload_file_name in workload_file_name_list:
#   run_workload = mantaray.run_workload.override(
#       task_id=workload_file_name.split(".")[0]
#   )(
#       workload_file_name=workload_file_name,
#   )
#   run_workload
