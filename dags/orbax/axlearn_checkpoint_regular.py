# Copyright 2025 Google LLC
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
Airflow DAG for validating AXLearn regular orbax (Native) checkpoint saving
functionality
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.orbax.util import test_config_util, validation_util
from xlml.utils.gke import zone_to_region
from xlml.utils import axlearn
from xlml.apis.xpk_cluster_config import XpkClusterConfig


SCHEDULE = "0 17 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "axlearn_reg_save"

RESERVE_TIME_FOR_OTHERS = datetime.timedelta(minutes=5)
RESERVE_TIME_FOR_PRE_WORKLOAD = datetime.timedelta(minutes=5)
RESERVE_TIME_FOR_WORKLOAD = datetime.timedelta(minutes=90)
RESERVE_TIME_FOR_POST_WORKLOAD = datetime.timedelta(minutes=5)
DAG_TIMEOUT = (
    RESERVE_TIME_FOR_OTHERS
    + RESERVE_TIME_FOR_PRE_WORKLOAD
    + RESERVE_TIME_FOR_WORKLOAD
    + RESERVE_TIME_FOR_POST_WORKLOAD
)

with models.DAG(
    dag_id=DAG_TEST_NAME,
    dagrun_timeout=DAG_TIMEOUT,
    start_date=datetime.datetime(2025, 6, 30),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "axlearn",
        "regular",
        "nightly",
        "jax0.5.3",
        "python3.10",
        "TPU",
        "v5p-128",
    ],
    description="""
      DAG that verifies the AXLearn regular (Native) checkpointing saving
      functionality
      """,
    doc_md="""
      # AXLearn Regular Checkpoint Validation DAG.

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      checkpoint saving when using **AXLearn Native Checkpointer** features.
      It will check that the checkpoints are being stored in the GCS bucket.
      Also the steps flag controls how many steps the job will run.

      ### Prerequisites
      To run this test, you need an existing cluster.

      ### Procedures
      1. **Install necessary dependencies for AXLearn:** Setup AXLearn CLI and
        all AXLearn necessary dependencies.
      2. **Run AXLearn JobSets:** The DAG runs a AXLearn JobSet.
      3. The DAG validates that **AXLearn checkpoints** are being saved
        correctly in the `GCS bucket` by checking bucket and pod logs.
    """,
    concurrency=2,
) as dag:
  checkpointing = test_config_util.Checkpointing(
      name="reg",
      enable_multi_tier_checkpointing=False,
      enable_emergency_checkpoint=False,
  )
  axlearn_configs = [
      test_config_util.TestConfigAXLearn(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          slices=[2],
          short_id=f"axlearn-{checkpointing.name}-sav",
          module="text.gpt.c4_trainer",
          label="tpu-v5p",
          model_name="fuji-7B-v2-flash",
          steps=200,
          trainer_dir=test_config_util.DEFAULT_BUCKET_AXLEARN,
          data_dir="gs://axlearn-public/tensorflow_datasets",
          trace_steps=[40, 90, 140, 190],
          workload_provision_timeout=RESERVE_TIME_FOR_PRE_WORKLOAD,
          workload_run_timeout=RESERVE_TIME_FOR_WORKLOAD,
          workload_post_test_timeout=RESERVE_TIME_FOR_POST_WORKLOAD,
      ),
  ]
  for mode, image in test_config_util.DOCKER_IMAGES_AXLEARN:
    image_repo, image_name = image.value.rsplit("/", 1)

    for axlearn_config in axlearn_configs:
      for slice_num in axlearn_config.slices:
        workload_id = axlearn.generate_workload_id()

        start_time = validation_util.generate_timestamp()

        # AXLearn head against JAX 0.5.3
        # Runs Fuji training on v5p-128 in the provided GCP Project
        run = axlearn_config.generate_axlearn_tpu_config(
            test_suffix="reg",
            test_owner=test_owner.CAMILO_Q,
            docker_image_name=image_name,
            docker_image_repo=image_repo,
            docker_image_full_url=image.value,
            num_slices=slice_num,
        ).run(
            workload_id=workload_id,
        )

        end_time = validation_util.generate_timestamp()

        validate_steps = (
            validation_util.validate_checkpoints_save_regular_axlearn(
                project_id=axlearn_config.cluster.project,
                run_name=workload_id,
                location=zone_to_region(axlearn_config.cluster.zone),
                cluster_name=axlearn_config.cluster.name,
                steps_to_validate=axlearn_config.generate_step_to_validate(),
                pod_pattern=".*-0",
                start_time=start_time,
                end_time=end_time,
            )
        )

        _ = workload_id >> start_time >> run >> end_time >> validate_steps
