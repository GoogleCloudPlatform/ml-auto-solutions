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
A DAG to run AOT compilation and HybridSim tests
for MaxText model configs on TPU v4, v5e.
"""
import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup

from xlml.utils import name_format
from xlml.apis import metric_config

from dags import composer_env
from dags.common.quarantined_tests import QuarantineTests
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, DockerImage, XpkClusters
from dags.multipod.configs import gke_config


# Run once a day at 1 pm UTC (5 am PST / 6 am PDT)
SCHEDULED_TIME = "30 2 * * *" if composer_env.is_prod_env() else None


def hybridsim_compile_and_run(group_id):
  with TaskGroup(group_id=group_id, prefix_group_id=True) as _:
    gcs_subfolder = f"{test_owner.Team.MULTIPOD.value}/maxtext"
    shared_gcs_location = name_format.generate_gcs_folder_location.override(
        task_id=f"{group_id}_generate_gcs_folder_location"
    )(
        f"{gcs_subfolder}/maxtext_configs_aot_hybridsim/v{tpu.value}",
        group_id,
    )

    tpu_version_str = (
        v5e_alt if tpu.value == TpuVersion.V5E.value else tpu.value
    )
    # Run AOT workload: generate HLO, upload to GCS
    aot_cmd = (
        (
            'export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump/'
            ' --xla_dump_large_constants"'
        ),
        (
            f"bash src/MaxText/configs/v{tpu_version_str}/{model_size}.sh"
            " EXECUTABLE=train_compile"
            f" M_COMPILE_TOPOLOGY=v{tpu_version_str}-{num_cores}"
            f" M_COMPILE_TOPOLOGY_NUM_SLICES={n}"
            f" DATASET_PATH=dummy-dataset OUTPUT_PATH=dummy-output-dir"
        ),
        "gsutil -m cp -r /tmp/xla_dump/ ${GCS_OUTPUT}",
    )
    maxtext_aot = gke_config.get_gke_config(
        time_out_in_min=240,
        test_name=f"maxtext-{model_size}-{n}xv{tpu.value}-{num_cores}-aot",
        run_model_cmds=aot_cmd,
        docker_image=DockerImage.MAXTEXT_TPU_JAX_NIGHTLY.value,
        test_owner=test_owner.AIRFLOW,
    ).run(gcs_location=shared_gcs_location)

    # Run HybridSim workload: read HLO from GCS, generate estimated step time
    cluster = clusters[tpu]
    chip_config = "default" if tpu == TpuVersion.V5E else "megacore"
    hybridsim_cmd = (
        "gsutil cp gs://cloud-hybridsim-prod/run_hybridsim.sh .",
        (
            f"bash run_hybridsim.sh GCS_XLA_DUMP_PATH=${{GCS_OUTPUT}}xla_dump"
            f" GCS_OUTPUT_PATH=${{GCS_OUTPUT}}estimated_cost_ns.jsonl"
            f" CHIP_CONFIG={chip_config} MODULE_NAME_PATTERN=jit_train_step*"
        ),
    )
    job_metric_config = metric_config.MetricConfig(
        json_lines=metric_config.JSONLinesConfig(
            file_location="estimated_cost_ns.jsonl",
        ),
        use_runtime_generated_gcs_folder=True,
    )
    maxtext_hybridsim = gke_config.get_gke_config(
        cluster=cluster,
        time_out_in_min=240,
        test_name=(
            f"maxtext-{model_size}-{n}xv{tpu.value}-{num_cores}-hybridsim"
        ),
        run_model_cmds=hybridsim_cmd,
        docker_image=DockerImage.CLOUD_HYBRIDSIM_NIGHTLY.value,
        test_owner=test_owner.AIRFLOW,
        user_specified_job_metric_config=job_metric_config,
    ).run(gcs_location=shared_gcs_location)

    _ = shared_gcs_location >> maxtext_aot >> maxtext_hybridsim


with models.DAG(
    dag_id="maxtext_configs_aot_hybridsim",
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "nightly",
        "mlscale_perfx",
        "TPU",
        "v4-8",
        "v5e-256",
    ],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
    concurrency=10,
) as dag:
  # Test setup values
  model_configs = {
      # accelerator: [(model_size, num_cores), ...],
      TpuVersion.V4: [("22b", 128), ("52b", 384)],
      TpuVersion.V5E: [
          ("16b", 256),
          ("32b", 256),
          ("64b", 256),
          ("128b", 256),
      ],
  }
  num_slices = [1, 2, 4, 8]
  clusters = {
      TpuVersion.V4: XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER,
      TpuVersion.V5E: XpkClusters.TPU_V5E_256_CLUSTER,
  }
  v5e_alt = "5e"

  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  for tpu, models in model_configs.items():
    for model_size, num_cores in models:
      for n in num_slices:
        test_group_id = (
            f"{model_size}-{n}xv{tpu.value}-{num_cores}-aot-hybridsim"
        )
        if QuarantineTests.is_quarantined(test_group_id):
          with quarantine_task_group:
            hybridsim_compile_and_run(test_group_id)
        else:
          hybridsim_compile_and_run(test_group_id)
