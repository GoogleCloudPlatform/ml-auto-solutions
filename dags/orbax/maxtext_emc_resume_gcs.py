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
from xlml.utils.gke import zone_to_region

SCHEDULE = "30 15 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_emc_resume_from_gcs"

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 12),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "emergency_checkpointing",
        "nightly",
        "orbax",
        "TPU",
        "v5p-128",
    ],
    description="A DAG to test MaxText Emergency Checkpoint Manager GCS restore functionality.",
    doc_md="""
      # Emergency Checkpoint Manager GCS Restore Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      that the **Emergency Checkpoint Manager (ECM)** can successfully save
      checkpoints to GCS and restore from them when local checkpoints are unavailable.

      ### Test Scenario
      The DAG tests the critical scenario where:
      1. Training runs normally with emergency checkpoints saved to GCS
      2. Local checkpoints are deleted (simulating failure/preemption)
      3. Training resumes and successfully restores from GCS emergency checkpoints

      ### Prerequisites
      - An existing GKE cluster with the necessary CSI driver for RAM disk (`/local`).
      - A GCS bucket with HNS (Hierarchical Namespace) enabled
      - Emergency Checkpoint Manager functionality enabled

      ### Test Flow
      1. **Apply Configuration:** Deploy Checkpoint Configuration YAML to set up the test environment.
      2. **Initial Training (0-100 steps):** Run MaxText with emergency checkpointing enabled
         - Saves regular checkpoints to GCS every 25 steps (0, 25, 50, 75, 99)
         - Saves local checkpoints to RAM disk for faster access
      3. **Simulate Failure:** The initial training job completes, and its pods (with local checkpoints) are deleted.
      4. **Resume Training (100-200 steps):** Continue training from where it left off
         - A new job is started, which restores from the latest GCS checkpoint since local ones are gone.
         - Training continues and saves additional checkpoints (100, 125, 150, 175, 199)
      5. **Final Cleanup:** Clean up RAM disk resources
      6. **Comprehensive Validation:**
         - **Log Validation:** Verify checkpoint save/restore events in logs
         - **GCS Restore Validation:** Confirm restoration from GCS occurred
         - **File Validation:** Verify all expected checkpoint files exist in GCS

      ### Key Parameters
      - **checkpoint_period=25:** Regular checkpoint interval for GCS saves
      - **local_checkpoint_period=20:** Local checkpoint interval for RAM disk
      - **Initial training:** 100 steps, **Resume training:** 200 steps total
      - **Emergency checkpointing:** Enabled throughout the test

      ### Success Criteria
      The test passes when:
      1. All expected checkpoints are saved to GCS during initial training
      2. Local checkpoints are successfully removed during cleanup
      3. Training successfully resumes from GCS checkpoints (not from scratch)
      4. All checkpoint files are verified to exist in the expected GCS locations
      5. Log validation confirms proper save and restore events occurred
    """,
    concurrency=2,
) as dag:
  first_training_step = 100
  second_training_step = 200
  out_folder = "maxtext_emc_orbax_resume_gcs"

  checkpointing = test_config_util.Checkpointing(
      name="emc", enable_multi_tier_checkpointing=False
  )

  test_configs = [
      test_config_util.TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-emc-resume-gcs",
          steps=first_training_step,
          local_checkpoint_period=20,
          checkpoint_period=25,
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
        apply_first_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)

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
            test_owner=test_owner.DEPP_L,
        ).run(
            ramdisk_directory=test_config_util.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            skip_post_process=True,
        )

        wait_delete_first_cpc = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule=TriggerRule.ALL_DONE
        )(test_config.cpc_config)
        apply_second_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)

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
            test_owner=test_owner.DEPP_L,
        ).run(
            ramdisk_directory=test_config_util.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            skip_post_process=True,
        )

        end_time = validation_util.generate_timestamp.override(
            task_id="generate_end_time"
        )()

        # Validation steps for entire training process (0 to 200)
        # Get validation steps for the full 200-step run
        steps_to_validate = test_config.generate_step_to_validate(
            is_local=False
        )
        # Add the end of the first training phase (step 99) to the list
        steps_to_validate.append(100 - 1)
        steps_to_validate = sorted(list(set(steps_to_validate)))

        # Validate that GCS restore happened during the second training run
        validate_gcs_restore = (
            validation_util.validate_restored_correct_checkpoint(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                pod_pattern=f"{test_config.short_id}.*-job-0-0",
                interrupt_at_step=first_training_step - 1,
                start_time=start_time,
                end_time=end_time,
            )
        )

        log_filters = [
            "jsonPayload.message:\"'event_type': 'emergency_restore'\"",
            "jsonPayload.message:\"'is_restoring_slice': True\"",
            "jsonPayload.message:\"'directory': 'gs://\"",
        ]
        validate_restore_source = validation_util.validate_log_exist.override(
            task_id="validate_emc_restored_from_gcs"
        )(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter=" AND ".join(log_filters),
            start_time=start_time,
            end_time=end_time,
        )

        # Validate that checkpoint files exist in GCS bucket
        validate_saved_checkpoints_steps_gcs = (
            validation_util.validate_gcs_checkpoint_files(
                bucket_path=(
                    f"{test_config_util.DEFAULT_BUCKET}/{out_folder}/{run_name}"
                ),
                steps_to_validate=steps_to_validate,
            )
        )

        # Final CPC cleanup to ensure symmetric start/end
        wait_delete_cpc_final = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule=TriggerRule.ALL_DONE,
            task_id="wait_delete_cpc_final",
        )(test_config.cpc_config).as_teardown(setups=apply_first_cpc)

        (
            wait_delete_cpc
            >> apply_first_cpc
            >> run_name
            >> start_time
            >> initial_training_run
            >> wait_delete_first_cpc
            >> apply_second_cpc
            >> resume_training_run
            >> end_time
            >> validate_gcs_restore
            >> validate_restore_source
            >> validate_saved_checkpoints_steps_gcs
            >> wait_delete_cpc_final
        )
