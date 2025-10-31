"""
A DAG to run MaxText multi-tier checkpointing tests.

This DAG performs a series of tests to save and validate checkpoints
for the MaxText model. It runs tests in two modes: one with the replicator
service enabled (Multi-tier Checkpointing). The tests are executed on a TPU
multi-pod cluster.
"""

import datetime

from airflow import models
from airflow.utils.trigger_rule import TriggerRule

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.orbax.util import validation_util
from dags.orbax.util import test_config_util
from dags.orbax.util import checkpoint_util
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 20 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_mtc_resume_from_gcs"

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 12),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "emergency_checkpoint_manager",
        "multitier_checkpointing",
        "nightly",
        "orbax",
    ],
    description="A DAG to test MaxText Multi-tier Checkpointing (MTC) GCS restore functionality.",
    doc_md="""
      # Multi-tier Checkpointing (MTC) GCS Restore Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      that **Multi-tier Checkpointing (MTC)** can successfully back up
      checkpoints to GCS and restore from them when local checkpoints are unavailable.

      ### Test Scenario
      The DAG tests the critical scenario where:
      1. Training runs and saves local checkpoints to a RAM disk.
      2. The MTC replicator service asynchronously backs up these local checkpoints to a GCS bucket.
      3. The initial training job is terminated, losing all local checkpoints.
      4. A new training job is started, which then restores from the latest GCS backup created by the replicator.

      ### Prerequisites
      - An existing cluster with Multi-tier Checkpointing configuration enabled
      - A GCS bucket with HNS (Hierarchical Namespace) enabled
      - Emergency Checkpoint Manager functionality enabled

      ### Test Flow
      1. **Apply Configuration:** Deploy Checkpoint Configuration YAML to enable MTC.
      2. **Initial Training (0-200 steps):** Run MaxText with MTC enabled.
         - Saves local checkpoints to RAM disk every 100 steps.
         - The MTC replicator backs up these checkpoints to GCS.
      3. **Simulate Failure:** The initial training job completes, and its pods (with local checkpoints) are deleted.
      4. **Resume Training (200-400 steps):** Continue training from where it left off.
         - A new job is started, which restores from the latest GCS backup since local ones are gone.
         - Training continues and saves additional checkpoints.
      5. **Final Cleanup:** Clean up RAM disk resources
      6. **Comprehensive Validation:**
         - **Replicator Backup Log Validation:** Verify the MTC replicator logs show successful backups to GCS.
         - **Replicator Restore Log Validation:** Verify the MTC replicator logs show a successful restore from a GCS backup.
         - **GCS File Validation:** Verify that the MTC checkpoint files exist in the GCS bucket with the correct backup folder structure.

      ### Key Parameters
      - **`enable_multi_tier_checkpointing=True`**: Enables MTC.
      - **`multi_tier_checkpointing_backup_interval_minutes=1`**: The MTC replicator backs up checkpoints every 1 minute.
      - **`local_checkpoint_period=100`**: Local checkpoints are saved every 100 steps.
      - **Initial training:** 200 steps, **Resume training:** 400 steps total.

      ### Success Criteria
      The test passes when:
      1. The MTC replicator successfully backs up local checkpoints to GCS.
      2. The resumed training job successfully restores from a GCS backup.
      3. All MTC checkpoint files are verified to exist in the expected GCS locations.
      4. Log validation confirms proper backup and restore events occurred.
    """,
    concurrency=2,
) as dag:
  first_training_step = 200
  second_training_step = 400
  out_folder = "maxtext_mtc_orbax_resume_gcs"

  checkpointing = test_config_util.Checkpointing(
      name="mtc",
      enable_multi_tier_checkpointing=True,
  )

  test_configs = [
      test_config_util.TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-mtc-resume-gcs",
          multi_tier_checkpointing_backup_interval_minutes=1,
          steps=first_training_step,
          local_checkpoint_period=100,
          base_dir=test_config_util.DEFAULT_BUCKET,
      ),
  ]

  for mode, image in test_config_util.DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        # We conditionally set the trigger_rule on the first task.
        # If first task group failed the next one can execute.
        wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule=TriggerRule.ALL_DONE
        )(test_config.cpc_config)
        apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)

        # Generate consistent run name for both training phases
        run_name = validation_util.generate_run_name(
            short_id=test_config.short_id,
            checkpointing_type=checkpointing.name,
            slice_number=slice_num,
            accelerator=test_config.accelerator,
        )

        # First training phase - train to step 100
        initial_workload_command = test_config.generate_workload_command(
            checkpoint_dir=test_config_util.DEFAULT_RAM_DISK,
            run_name=run_name,
            slice_num=slice_num,
            out_folder=out_folder,
            enable_multi_tier_checkpointing=checkpointing.enable_multi_tier_checkpointing,
        )

        start_time = validation_util.generate_timestamp.override(
            task_id="generate_start_time"
        )()

        initial_training_run = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}",
            run_model_cmds=initial_workload_command,
            docker_image=image.value,
            test_owner=test_owner.JACKY_F,
        ).run(
            ramdisk_directory=test_config_util.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )

        wait_delete_first_cpc = checkpoint_util.wait_for_cpc_deletion.override(
            task_id="wait_delete_first_cpc",
            trigger_rule=TriggerRule.ALL_DONE,
        )(test_config.cpc_config)
        apply_second_cpc = checkpoint_util.apply_cpc.override(
            task_id="apply_second_cpc"
        )(test_config.cpc_config)

        # Second training phase - continue from checkpoint and reach step 200
        test_config.steps = second_training_step
        resume_workload_command = test_config.generate_workload_command(
            checkpoint_dir=test_config_util.DEFAULT_RAM_DISK,
            run_name=run_name,
            slice_num=slice_num,
            out_folder=out_folder,
            enable_multi_tier_checkpointing=checkpointing.enable_multi_tier_checkpointing,
        )

        resume_training_run = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-restore",
            run_model_cmds=resume_workload_command,
            docker_image=image.value,
            test_owner=test_owner.JACKY_F,
        ).run(
            ramdisk_directory=test_config_util.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )

        end_time = validation_util.generate_timestamp.override(
            task_id="generate_end_time"
        )()

        # Validate that GCS restore happened during the second training run
        validate_gcs_restore = (
            validation_util.validate_restored_correct_checkpoint(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                pod_pattern="max.*-job-0-0",
                interrupt_at_step=first_training_step - 1,
                start_time=start_time,
                end_time=end_time,
            )
        )

        # Validate that replicator backed up checkpoints to GCS
        validate_backup_steps = (
            validation_util.validate_replicator_gcs_backup_log(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                namespace="gke-managed-checkpointing",
                container_name="replication-worker",
                pod_pattern="multitier-driver-.*",
                start_time=start_time,
                end_time=end_time,
            )
        )

        # Validate that replicator restored checkpoints from GCS backup
        validate_replicator_restore = (
            validation_util.validate_replicator_gcs_restore_log(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                namespace="gke-managed-checkpointing",
                container_name="replication-worker",
                pod_pattern="multitier-driver-.*",
                start_time=start_time,
                end_time=end_time,
                backed_up_steps=validate_backup_steps,
            )
        )

        # Validate that MTC checkpoint files exist in GCS bucket with correct backup folder structure
        validate_mtc_gcs_files = (
            validation_util.validate_gcs_checkpoint_files.override(
                task_id="validate_mtc_gcs_files"
            )(
                bucket_path=f"{test_config_util.DEFAULT_BUCKET}/{run_name}",
                steps_to_validate=validate_backup_steps,
                enable_multi_tier_checkpointing=True,
            )
        )

        # Final CPC cleanup to ensure symmetric start/end
        wait_delete_cpc_final = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule=TriggerRule.ALL_DONE,
            task_id="wait_delete_cpc_final",
        )(test_config.cpc_config).as_teardown(setups=apply_cpc)

        (
            wait_delete_cpc
            >> apply_cpc
            >> run_name
            >> start_time
            >> initial_training_run
            >> wait_delete_first_cpc
            >> apply_second_cpc
            >> resume_training_run
            >> end_time
            >> validate_gcs_restore
            >> validate_backup_steps
            >> validate_replicator_restore
            >> validate_mtc_gcs_files
            >> wait_delete_cpc_final
        )
