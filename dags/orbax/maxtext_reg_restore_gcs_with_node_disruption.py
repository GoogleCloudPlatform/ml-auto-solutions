"""
A DAG to run MaxText regular checkpointing tests.

This DAG performs a series of tests to save and restore and validate checkpoints
for the MaxText model using the regular checkpointer.
The tests are executed on a TPU multi-pod cluster.
"""

import datetime

from airflow import models
from airflow.decorators import task

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.orbax.util import validation_util, test_config_util
from xlml.utils.xpk import MAIN_BRANCH
from xlml.utils.gke import zone_to_region

SCHEDULE = "15 14 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_regular_restore_with_node_disruption"


@task
def generate_workload_checkpoints_location(gcs_ckpt_location: str) -> str:
  return f"{gcs_ckpt_location}/checkpoints"


with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 10, 21),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "regular_checkpointing",
        "nightly",
        "orbax",
        "TPU",
        "v5p-128",
    ],
    description="DAG that verifies MaxText regular checkpointing restoring functionality from GCS bucket .",
    doc_md="""
      # MaxText Regular Checkpointing Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      regular checkpoint saving and restoring with node disruption for the MaxText model.
      It runs a single MaxText training job without emergency or multi-tier checkpointing
      features and validates that checkpoints are saved and restore correctly to and from
      GCS at specified intervals.

      ### Prerequisites
      - An existing TPU cluster configured for MaxText training.
      - A GCS bucket for storing logs and checkpoints.

      ### Procedures
      1.  **Run MaxText Training:** A MaxText training job is initiated with
          regular checkpointing enabled. The job runs for 120 steps and saves
          checkpoints to GCS every 20 steps.
      2.  **Node Interruption:** Simulate a node interruption while step 40
          is saved into GCS bucket completely (commit_message.txt is generated)
      3.  **Log Validation:** Verify checkpoint save events in logs
         - Looks for 'Finished async_save (blocking + background)' messages
         - Validates that saves occurred at expected steps
      4. **File Validation:** Verify checkpoint files exist in GCS bucket
         - Checks that actual checkpoint files are present for each expected step
      5.  **Validate Restore:** The DAG inspects the application logs to confirm
          that an `'restore'` event occurred.
      6.  **Validate Checkpoint Integrity:** It then verifies that the training job
          resumed and continued to save checkpoints correctly after the restore,
          ensuring no data was lost.
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
          short_id="max-reg-res-gcs-node",
          steps=120,
          checkpoint_period=20,
          base_dir=test_config_util.DEFAULT_BUCKET,
      ),
  ]

  step_to_interrupt = 40

  for mode, image in test_config_util.DOCKER_IMAGES:
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
            enable_multi_tier_checkpointing=checkpointing.enable_multi_tier_checkpointing,
            enable_emergency_checkpoint=checkpointing.enable_emergency_checkpoint,
        )
        gcs_location = generate_workload_checkpoints_location.override(
            task_id="gcs_bucket_checkpoints_location"
        )(f"{test_config.base_dir}/{DAG_TEST_NAME}/{run_name}")

        start_time = validation_util.generate_timestamp.override(
            task_id="generate_start_time"
        )()

        maxtext_chkpt_run_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.SHARON_Y,
        ).run_with_node_interruption(
            gcs_location=gcs_location,
            xpk_branch=MAIN_BRANCH,
            skip_post_process=True,
            expect_reach_to_step=step_to_interrupt,
            max_restart=15,
            check_file_exists=True,
        )

        end_time = validation_util.generate_timestamp.override(
            task_id="generate_end_time"
        )()

        validate_restore_step = (
            validation_util.validate_restored_correct_checkpoint(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                pod_pattern=".*0-0",
                interrupt_at_step=step_to_interrupt,
                start_time=start_time,
                end_time=end_time,
            )
        )

        validate_restored_source = validation_util.validate_log_exist.override(
            task_id="validate_reg_restored_from_gcs"
        )(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter="\"'event_type': 'restore'\"",
            start_time=start_time,
            end_time=end_time,
        )

        gcs_steps_to_validate = test_config.generate_step_to_validate(
            is_local=False
        )

        validate_log = validation_util.validate_checkpoint_at_steps_are_saved(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            ram_disk="gcs",
            steps_to_validate=gcs_steps_to_validate,
            start_time=start_time,
            end_time=end_time,
        )

        validate_bucket = validation_util.validate_gcs_checkpoint_files(
            bucket_path=f"{test_config_util.DEFAULT_BUCKET}/{DAG_TEST_NAME}/{run_name}",
            steps_to_validate=gcs_steps_to_validate,
        )

        (
            run_name
            >> gcs_location
            >> start_time
            >> maxtext_chkpt_run_test
            >> end_time
            >> validate_restore_step
            >> validate_restored_source
            >> validate_log
            >> validate_bucket
        )
