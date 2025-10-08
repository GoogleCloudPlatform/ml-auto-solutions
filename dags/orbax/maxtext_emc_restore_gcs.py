"""A DAG to run MaxText EMC.

Validates the GCS checkpoints are restored as expected
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.orbax.util import checkpoint_util
from dags.orbax.util import test_config_util
from dags.orbax.util import validation_util
from xlml.utils.gke import zone_to_region
from xlml.utils.xpk import BRANCH_ABHINAV_MTC

DAG_TEST_NAME = "maxtext_emc_orbax_res_gcs"
SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 30),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "multipod_team",
        "maxtext",
        "emergency_checkpointing",
        "nightly",
        "orbax",
    ],
    description="DAG to verify MaxText's emergency restore from GCS checkpoints after a full cluster interruption.",
    doc_md="""
      # MaxText Emergency Restore from GCS Validation DAG

      ### Description
      This DAG validates the emergency restore capability of MaxText from
      Google Cloud Storage (GCS) checkpoints. It simulates a full cluster
      failure during a training job and verifies that the job can successfully
      resume from the last saved GCS checkpoint. This test is critical for
      ensuring the resilience of long-running training jobs against catastrophic
      failures where local checkpoints are lost.

      ### Prerequisites
      - An existing GKE cluster with the Multi-tier Checkpointing (MTC)
        configuration enabled, which provides the necessary CSI driver for
        RAM disk (`/local`).
      - A GCS bucket for storing logs and checkpoints.

      ### Procedures
      1.  **Apply MTC Configuration:** A `CheckpointingPolicyConfiguration` (CPC)
          is applied to the cluster to set up the MTC environment.
      2.  **Run MaxText with Interruption:** A MaxText training job is initiated.
          During its execution, the last node is deleted to simulate a full
          cluster interruption, triggering the emergency restore mechanism from GCS.
      3.  **Validate Restore:** The DAG inspects the application logs to confirm
          that an `'emergency_restore'` event occurred from a GCS source.
      4.  **Validate Checkpoint Integrity:** It then verifies that the training job
          resumed and continued to save checkpoints correctly after the restore.
      """,
    concurrency=2,
) as dag:
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
          short_id="max-res-gcs",
          replicator_backup_time=30,
          step=200,
          checkpoint_step=30,
          local_checkpoint_step=200,
          base_dir=test_config_util.DEFAULT_BUCKET,
      ),
  ]

  step_to_interrupt = 60

  for mode, image in test_config_util.DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule="all_done"
        )(test_config.cpc_config)

        apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)

        # Generate consistent run name for both training phases
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
            out_folder="maxtext_emc_orbax_res_gcs",
            enable_multi_tier_checkp=checkpointing.enable_multi_tier_checkpointing,
        )

        start_time = validation_util.generate_timestamp.override(
            task_id="generate_start_time"
        )()

        maxtext_chkpt_run_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-emc",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.DEPP_L,
        ).run_with_node_interruption(
            ramdisk_directory=test_config_util.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
            last_node=True,
            expect_reach_to_step=step_to_interrupt
        )

        end_time = validation_util.generate_timestamp.override(
            task_id="generate_end_time"
        )()

        validate_restore_step = (
            validation_util.validate_restored_correct_checkpoint(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                pod_pattern=f"{test_config.short_id}-emc.*1-0",
                interrupt_at_step=step_to_interrupt,
                start_time=start_time,
                end_time=end_time,
                check_last_two_local_saves=False,
            )
        )

        log_filters = [
            "jsonPayload.message:\"'event_type': 'emergency_restore'\"",
            "jsonPayload.message:\"'is_restoring_slice': True\"",
            "jsonPayload.message:\"'directory': 'gs://\"",
        ]
        validate_restored_source = validation_util.validate_log_exist.override(
            task_id="validate_emc_restored_from_gcs"
        )(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter=" AND ".join(log_filters),
            start_time=start_time,
            end_time=end_time,
        )

        gcs_saved_steps_to_validate = test_config.generate_step_to_validate(
            is_local=False
        )

        validate_saved_checkpoints_steps_gcs = (
            validation_util.validate_gcs_checkpoint_files(
                bucket_path=(
                    f"{test_config_util.DEFAULT_BUCKET}/{DAG_TEST_NAME}/{run_name}"
                ),
                steps_to_validate=gcs_saved_steps_to_validate,
            )
        )

        # Final CPC cleanup to ensure symmetric start/end
        wait_delete_cpc_final = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule="all_done", task_id="wait_delete_cpc_final"
        )(test_config.cpc_config).as_teardown(setups=apply_cpc)

        (
            wait_delete_cpc
            >> apply_cpc
            >> run_name
            >> start_time
            >> maxtext_chkpt_run_test
            >> end_time
            >> validate_restore_step
            >> validate_restored_source
            >> validate_saved_checkpoints_steps_gcs
            >> wait_delete_cpc_final
        )
