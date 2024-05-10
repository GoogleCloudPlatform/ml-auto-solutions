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
A DAG to run AOT compilation and HybridSim tests for MaxText model configs on TPU v4, v5e.
"""
import datetime
from airflow import models
from dags import composer_env, test_owner
from dags.vm_resource import TpuVersion, Zone, DockerImage, ClusterName, Project
from dags.multipod.configs import gke_config
from xlml.utils import name_format
from airflow.utils.task_group import TaskGroup
from dags.multipod.configs import gke_config
from xlml.apis import metric_config

# Run once a day at 1 pm UTC (5 am PST / 6 am PDT)
SCHEDULED_TIME = "0 13 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_configs_aot_hybridsim",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "nightly"],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
    concurrency=10,
) as dag:
  # Test setup values
  model_configs = {
      # accelerator: [(model_size, num_cores), ...],
      TpuVersion.V4: [("22b", 128), ("52b", 384)],
      TpuVersion.V5E: [("16b", 256), ("32b", 256), ("64b", 256), ("128b", 256)],
  }
  num_slices = [1, 2, 4, 8]
  clusters = {
      # accelerator: [(cluster_name, cluster_zone, cluster_project, num_cores)],
      TpuVersion.V4: (
          ClusterName.V4_8_MULTISLICE_CLUSTER.value,
          Zone.US_CENTRAL2_B.value,
          Project.TPU_PROD_ENV_MULTIPOD.value,
          8,
      ),
      TpuVersion.V5E: (
          ClusterName.V5E_256_US_WEST_4_MULTISLICE_CLUSTER.value,
          Zone.US_WEST4_B.value,
          Project.TPU_PROD_ENV_MULTIPOD.value,
          256,
      ),
  }
  v5e_alt = "5e"

  for tpu, models in model_configs.items():
    for model_size, num_cores in models:
      for n in num_slices:
        test_group_id = (
            f"{model_size}-{n}xv{tpu.value}-{num_cores}-aot-hybridsim"
        )
        with TaskGroup(group_id=test_group_id, prefix_group_id=False) as group:
          gcs_subfolder = f"{test_owner.Team.MULTIPOD.value}/maxtext"
          shared_gcs_location = (
              name_format.generate_gcs_folder_location.override(
                  task_id=f"{test_group_id}_generate_gcs_folder_location"
              )(
                  f"{gcs_subfolder}/maxtext_configs_aot_hybridsim/v{tpu.value}",
                  test_group_id,
              )
          )

          # Run AOT workload: generate HLO, upload to GCS
          aot_cmd = (
              'export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump/"',
              f"bash MaxText/configs/v{v5e_alt if tpu.value == TpuVersion.V5E.value else tpu.value}/{model_size}.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY=v{v5e_alt if tpu.value == TpuVersion.V5E.value else tpu.value}-{num_cores} M_COMPILE_TOPOLOGY_NUM_SLICES={n}",
              "gsutil cp gs://cloud-hybridsim-prod/desanitize_and_upload_hlo.sh .",
              "bash desanitize_and_upload_hlo.sh LOCAL_DIR=/tmp/xla_dump/ GCS_OUTPUT_PATH=${GCS_OUTPUT}",
          )
          maxtext_aot = gke_config.get_gke_config(
              tpu_version=TpuVersion.V4,
              tpu_cores=8,
              tpu_zone=Zone.US_CENTRAL2_B.value,
              time_out_in_min=240,
              test_name=f"maxtext-{model_size}-{n}xv{tpu.value}-{num_cores}-aot",
              run_model_cmds=aot_cmd,
              docker_image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
              test_owner=test_owner.RAYMOND_Z,
          ).run(gcs_location=shared_gcs_location)

          # Run HybridSim workload: read HLO from GCS, generate estimated step time
          cluster_name, zone, project_name, cores = clusters[tpu]
          chip_config = "default" if tpu == TpuVersion.V5E else "megacore"
          hybridsim_cmd = (
              "gsutil cp gs://cloud-hybridsim-prod/run_hybridsim.sh .",
              f"bash run_hybridsim.sh GCS_XLA_DUMP_PATH=${{GCS_OUTPUT}}xla_dump GCS_OUTPUT_PATH=${{GCS_OUTPUT}} CHIP_CONFIG={chip_config}",
          )
          job_metric_config = metric_config.MetricConfig(
              json_lines=metric_config.JSONLinesConfig(
                  file_location="estimated_cost_ns.jsonl",
              ),
              use_runtime_generated_gcs_folder=True,
          )
          maxtext_hybridsim = gke_config.get_gke_config(
              tpu_version=tpu,
              tpu_cores=cores,
              tpu_zone=zone,
              project_name=project_name,
              cluster_name=cluster_name,
              time_out_in_min=240,
              test_name=f"maxtext-{model_size}-{n}xv{tpu.value}-{num_cores}-hybridsim",
              run_model_cmds=hybridsim_cmd,
              docker_image=DockerImage.CLOUD_HYBRIDSIM_NIGHTLY.value,
              test_owner=test_owner.RAYMOND_Z,
              user_specified_job_metric_config=job_metric_config,
          ).run(gcs_location=shared_gcs_location)

          shared_gcs_location >> maxtext_aot >> maxtext_hybridsim
