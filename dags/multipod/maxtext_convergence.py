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

"""
A DAG to run MaxText convergence tests for both bf16 and int8.
"""
import datetime
from airflow import models
from dags import composer_env, test_owner, gcs_bucket
from dags.vm_resource import TpuVersion, Zone, DockerImage, ClusterName, GpuVersion
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from xlml.apis import gcp_config, metric_config, task, test_config

# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_convergence",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable"],
    start_date=datetime.datetime(2024, 3, 1),
    catchup=False,
    concurrency=2,
) as dag:
  current_time = datetime.datetime.now()
  current_date = current_time.strftime("%Y-%m-%d")
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext/stable/automated/{current_date}"
  )
  dataset_path = gcs_bucket.MAXTEXT_DIR

  steps = 10200  # Half Chinchilla
  tpu_loss_threshold = 2.7
  gpu_loss_threshold = 4.5

  base_tpu_convergence_command = f"bash end_to_end/tpu/test_convergence_1b_params.sh OUTPUT_PATH={base_output_directory} DATASET_PATH={dataset_path} LOSS_THRESHOLD={tpu_loss_threshold} STEPS={steps}"
  tpu_convergence_tests = {
      "maxtext-convergence-bf16": ((base_tpu_convergence_command),),
      "maxtext-convergence-int8": (
          (f"export M_QUANTIZATION=int8; {base_tpu_convergence_command}"),
      ),
      "maxtext-convergence-subset-hosts": (
          (
              f"export M_EXPANSION_FACTOR_REAL_DATA=2; {base_tpu_convergence_command}"
          ),
      ),
  }

  for test_name, run_command in tpu_convergence_tests.items():
    maxtext_v4_configs_test = gke_config.get_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=128,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        cluster_name=ClusterName.V4_128_MULTISLICE_CLUSTER.value,
        time_out_in_min=300,
        test_name=test_name,
        run_model_cmds=run_command,
        docker_image=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
        test_owner=test_owner.MATT_D,
        base_output_directory=base_output_directory,
        metric_aggregation_strategy=metric_config.AggregationStrategy.LAST,
    ).run_with_run_name_generation()


  base_gpu_convergence_command = f"bash end_to_end/gpu/a3/test_convergence_1b_params.sh OUTPUT_PATH={base_output_directory} DATASET_PATH={dataset_path} LOSS_THRESHOLD={gpu_loss_threshold} STEPS={steps}"
  gpu_convergence_tests = {
      "maxtext-convergence-bf16": ((base_gpu_convergence_command),),
      "maxtext-convergence-int8": (
          (f"export QUANTIZATION=int8; {base_gpu_convergence_command}"),
      ),
  }

  for test_name, run_command in gpu_convergence_tests.items():
    pinned_a3_gpu = gke_config.get_maxtext_end_to_end_gpu_gke_test_config(
        accelerator_type=GpuVersion.XPK_H100,
        gpu_zone=Zone.US_CENTRAL1_C.value,
        time_out_in_min=300,
        test_name=f"{test_name}-pinned",
        run_model_cmds=run_command,
        num_slices=1,
        cluster_name=ClusterName.A3_CLUSTER.value,
        docker_image=DockerImage.MAXTEXT_GPU_JAX_PINNED.value,
        test_owner=test_owner.Anfal_S,
    ).run()
