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
from dags.vm_resource import TpuVersion, Zone, DockerImage
from dags.multipod.configs import maxtext_gke_config
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
      f"{gcs_bucket.XLML_OUTPUT_DIR}/maxtext/stable/automated/{current_date}"
  )
  dataset_path = gcs_bucket.MAXTEXT_DIR

  steps = 10# Half Chinchilla
  loss_threshold = 20.7 # 2.7
  
  base_convergence_command = f"bash end_to_end/test_convergence_1b_params.sh OUTPUT_PATH={base_output_directory} DATASET_PATH={dataset_path} LOSS_THRESHOLD={loss_threshold} STEPS={steps}"
  convergence_tests = {
      "maxtext-convergence-bf16": ((base_convergence_command),),
      "maxtext-convergence-int8": (
          (f"export M_QUANTIZATION=int8; {base_convergence_command}"),
      ),
  }

  for test_name, run_command in convergence_tests.items():
    maxtext_v4_configs_test = maxtext_gke_config.get_maxtext_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=128,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=300,
        test_name=test_name,
        run_model_cmds=run_command,
        docker_image=DockerImage.MAXTEXT_JAX_STABLE.value,
        test_owner=test_owner.MATT_D,
    ).run()














  # Testing configurations
  model_configs = {
      # accelerator: [(model_size, num_cores), ...],
      "v4": [("22b", 128), ("52b", 384)],
      "v5e": [("16b", 256), ("32b", 256), ("64b", 256), ("128b", 256)],
      "v5p": [
          ("32b", 128),
          ("64b", 128),
          ("128b", 256),
          ("128b", 512),
          ("256b", 1024),
          ("512b", 1024),
          ("1024b", 2048),
          ("1024b", 4096),
      ],
  }
  num_slices = [1, 2]
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_JAX_STABLE),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_JAX_NIGHTLY),
  ]

  run_model_cmds_dict = {}
  for tpu, models in model_configs.items():
    run_model_cmds = []
    for model_size, num_cores in models:
      for n in num_slices:
        cmd = f"bash MaxText/configs/{tpu}/{model_size}.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY={tpu}-{num_cores} M_COMPILE_TOPOLOGY_NUM_SLICES={n}"
        run_model_cmds.append(cmd)
    run_model_cmds_dict[tpu] = run_model_cmds

  for mode, image in docker_images:
    maxtext_v4_configs_test = maxtext_gke_config.get_maxtext_configs_aot_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"maxtext-aot-v4-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v4"],
        docker_image=image.value,
        test_owner=test_owner.RAYMOND_Z,
    ).run()

    maxtext_v5e_configs_test = maxtext_gke_config.get_maxtext_configs_aot_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"maxtext-aot-v5e-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v5e"],
        docker_image=image.value,
        test_owner=test_owner.RAYMOND_Z,
    ).run()

    maxtext_v5p_configs_test = maxtext_gke_config.get_maxtext_configs_aot_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name=f"maxtext-aot-v5p-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v5p"],
        docker_image=image.value,
        test_owner=test_owner.RAYMOND_Z,
    ).run()

    maxtext_v4_configs_test >> maxtext_v5e_configs_test >> maxtext_v5p_configs_test