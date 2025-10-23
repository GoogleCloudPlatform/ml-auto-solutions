"""A DAG to run MaxText MTC.

Validates the local checkpoints are restored as expected
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

DAG_TEST_NAME = "maxtext_mtc_orbax_res_local"
SCHEDULE = "0 16 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 6, 30),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=["multipod_team", "maxtext", "nightly", "orbax", "mtc"],
    description="DAG to verify MaxText's multi-tier restore from local checkpoints after a node interruption.",
    doc_md="""
      # MaxText Multi-tier Restore from Local Checkpoint Validation DAG

      ### Description
      This DAG validates the multi-tier restore capability of MaxText from local
      checkpoints. It simulates a node failure during a training job and verifies
      that the job can successfully resume from the last saved local checkpoint
      replicated from a peer. This test is critical for ensuring the resilience
      of long-running training jobs against hardware failures when using
      Multi-tier Checkpointing (MTC).

      ### Prerequisites
      - An existing GKE cluster with the Multi-tier Checkpointing (MTC)
        configuration enabled, which provides the necessary CSI driver for
        RAM disk (`/local`).
      - A GCS bucket for storing logs and base model outputs.

      ### Procedures
      1.  **Apply MTC Configuration:** A `CheckpointingPolicyConfiguration` (CPC)
          is applied to the cluster to set up the MTC environment.
      2.  **Run MaxText with Interruption:** A MaxText training job is initiated.
          During its execution, a node interruption is simulated to trigger the
          restore mechanism.
      3.  **Validate Restore:** The DAG inspects the application logs to confirm
          that a `'restore'` event occurred and that the checkpoint was copied
          from a peer node.
      4.  **Validate Checkpoint Integrity:** It then verifies that the training job
          resumed and continued to save checkpoints correctly after the restore,
          ensuring no data was lost.
      """,
    concurrency=2,
) as dag:
  test_configs = [
      test_config_util.TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-res-loc",
          multi_tier_checkpointing_backup_interval_minutes=30,
          steps=300,
          checkpoint_period=100,
          local_checkpoint_period=20,
          base_dir=test_config_util.DEFAULT_BUCKET,
      ),
  ]
  checkpointing = test_config_util.Checkpointing(
      name="mtc", enable_multi_tier_checkpointing=True
  )

  step_to_interrupt = 60

  for mode, image in test_config_util.DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
            trigger_rule="all_done"
        )(test_config.cpc_config)
        apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)

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
            out_folder="maxtext_mtc_orbax_res_local",
            enable_multi_tier_checkpointing=checkpointing.enable_multi_tier_checkpointing,
        )

        start_time = validation_util.generate_timestamp.override(
            task_id="generate_start_time"
        )()

        maxtext_chkpt_run_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-mtc",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.DEPP_L,
        ).run_with_node_interruption(
            ramdisk_directory=test_config_util.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
            expect_reach_to_step=step_to_interrupt,
            max_restart=15,
        )

        end_time = validation_util.generate_timestamp.override(
            task_id="generate_end_time"
        )()

        validate_restore_step = (
            validation_util.validate_restored_correct_checkpoint(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                pod_pattern=f"{test_config.short_id}-mtc.*0-0",
                interrupt_at_step=step_to_interrupt,
                start_time=start_time,
                end_time=end_time,
            )
        )

        log_filters = [
            "jsonPayload.message:\"'event_type': 'restore'\"",
            "jsonPayload.message:\"'directory': '/local\"",
        ]
        validate_restored_source = validation_util.validate_log_exist.override(
            task_id="validate_restore_copy_from_peer"
        )(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter=" AND ".join(log_filters),
            start_time=start_time,
            end_time=end_time,
        )

        steps_to_validate = test_config.generate_step_to_validate(is_local=True)

        validate_local_saved_steps = (
            validation_util.validate_checkpoint_at_steps_are_saved(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                start_time=start_time,
                end_time=end_time,
                steps_to_validate=steps_to_validate,
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
            >> validate_local_saved_steps
            >> wait_delete_cpc_final
        )
