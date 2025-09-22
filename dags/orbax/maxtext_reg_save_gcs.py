"""
A DAG to run MaxText regular checkpointing tests.

This DAG performs a series of tests to save and validate checkpoints
for the MaxText model using the regular checkpointer.
The tests are executed on a TPU multi-pod cluster.
"""

import datetime
from typing import Optional

from airflow import models

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import validation_util, checkpoint_util, test_config_util
from xlml.utils.xpk import MAIN_BRANCH
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 16 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_regular_save"

# Only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
)]


with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 9, 17),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "regular_checkpointing",
        "nightly",
        "orbax",
    ],
    description="A DAG to test MaxText regular checkpoint saving functionality.",
    doc_md="""
      # MaxText Regular Checkpointing Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      regular checkpoint saving for the MaxText model. The DAG runs a single
      MaxText training job and validates that checkpoints are saved correctly
      at specified intervals.

      ### Test Scenario
      The DAG tests the standard checkpointing scenario where:
      1. Training runs normally with regular checkpoints saved to GCS at defined intervals
      2. Checkpoint logs are validated to ensure proper save events
      3. GCS bucket is validated to ensure checkpoint files exist

      ### Prerequisites
      - An existing TPU cluster configured for MaxText training
      - Access to a GCS bucket for checkpoint storage

      ### Test Flow
      1. **Start Training:** Run MaxText training job for 100 steps
         - Saves regular checkpoints to GCS every 25 steps (0, 25, 50, 75, 99)
         - No local checkpoints or emergency features are used
      2. **Log Validation:** Verify checkpoint save events in logs
         - Looks for 'Finished async_save (blocking + background)' messages
         - Validates that saves occurred at expected steps
      3. **File Validation:** Verify checkpoint files exist in GCS bucket
         - Checks that actual checkpoint files are present for each expected step

      ### Key Parameters
      - **checkpoint_step=25:** Regular checkpoint interval for GCS saves
      - **step=100:** Total training steps (checkpoints at 0, 25, 50, 75, 99)
      - **Model:** llama2-7b on v5p-128 TPU slices

      ### Success Criteria
      The test passes when:
      1. All expected checkpoint save logs are found at steps 0, 25, 50, 75, 99
      2. All corresponding checkpoint files exist in the GCS bucket
      3. Only regular checkpointing is used (no emergency or multi-tier features)
    """,
    concurrency=2,
) as dag:
  checkpointing = test_config_util.Checkpointing(
      name="reg",
      enable_multi_tier_checkpointing=False,
      enable_emergency_checkpoint=False,
  )
  test_configs = [
      test_config_util.TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-reg-save",
          step=100,
          checkpoint_step=25,
          replicator_backup_time=30,
          local_checkpoint_step=30,
          base_dir=test_config_util.DEFAULT_BUCKET,
      ),
  ]
  for mode, image in DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        run_name = validation_util.generate_run_name(
            short_id=test_config.short_id,
            checkpointing_type=checkpointing.name,
            slice_number=slice_num,
            accelerator=test_config.accelerator,
        )

        workload_command = test_config.generate_workload_command(
            checkpoint_dir=test_config_util.DEFAULT_RAM_DISK,
            run_name=run_name,
            slice_num=slice_num,
            out_folder=DAG_TEST_NAME,
            enable_multi_tier_checkp=checkpointing.enable_multi_tier_checkpointing,
            enable_emergency_checkpoint=checkpointing.enable_emergency_checkpoint,
        )

        start_time = validation_util.generate_timestamp()

        maxtext_chkpt_run_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.JACKY_F,
        ).run(
            xpk_branch=MAIN_BRANCH,
            skip_post_process=True,
        )

        steps_to_validate = test_config.generate_step_to_validate(
            is_local=False
        )

        end_time = validation_util.generate_timestamp()

        validate_steps = validation_util.validate_checkpoint_at_steps_are_saved(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            ram_disk="gcs",
            start_time=start_time,
            end_time=end_time,
            steps_to_validate=steps_to_validate,
        )

        validate_checkpoints_file = validation_util.validate_gcs_checkpoint_files(
            bucket_path=f"{test_config_util.DEFAULT_BUCKET}/{DAG_TEST_NAME}/{run_name}",
            steps_to_validate=steps_to_validate,
        )

        (
            run_name
            >> start_time
            >> maxtext_chkpt_run_test
            >> end_time
            >> validate_steps
            >> validate_checkpoints_file
        )
