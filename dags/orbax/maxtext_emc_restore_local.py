"""A DAG to run MaxText EMC.
Validates the local checkpoints are restored as expected
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import DockerImage
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import checkpoint_util
from dags.orbax.util import orbax
from dags.orbax.util import validation_util
from xlml.utils.gke import zone_to_region
from xlml.utils.xpk import BRANCH_ABHINAV_MTC

DAG_TEST_NAME = "maxtext_emc_orbax_res_local"
SCHEDULE = "0 11 * * *" if composer_env.is_prod_env() else None

# Only one version of the Docker image is supported at the moment.
# Other versions (e.g., "stable") may be introduced later.
DOCKER_IMAGES = [(
    SetupMode.NIGHTLY,
    DockerImage.MAXTEXT_TPU_JAX_ORBAX_HEAD,
)]

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
    description="DAG to verify MaxText's emergency restore from local checkpoints after a node interruption.",
    doc_md="""
      # MaxText Emergency Restore from Local Checkpoint Validation DAG

      ### Description
      This DAG validates the emergency restore capability of MaxText from local
      checkpoints. It simulates a node failure during a training job and verifies
      that the job can successfully resume from the last saved local checkpoint.
      This test is critical for ensuring the resilience of long-running training
      jobs against hardware failures.

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
          emergency restore mechanism.
      3.  **Validate Restore:** The DAG inspects the application logs to confirm
          that an `'emergency_restore'` event occurred.
      4.  **Validate Checkpoint Integrity:** It then verifies that the training job
          resumed and continued to save checkpoints correctly after the restore,
          ensuring no data was lost.
      """,
    concurrency=2,
) as dag:
  checkpointing = orbax.Checkpointing(
      name="emc", enable_multi_tier_checkpointing=False
  )
  test_configs = [
      orbax.TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-res-loc",
          replicator_backup_time=30,
          step=150,
          checkpoint_step=20,
          local_checkpoint_step=20,
          ram_disk_size_in_mi="800000Mi",
          base_dir=orbax.DEFAULT_BUCKET,
      ),
  ]

  for mode, image in DOCKER_IMAGES:
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
            checkpoint_dir=orbax.DEFAULT_RAM_DISK,
            run_name=run_name,
            out_folder="maxtext_emc_orbax_res_local",
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
            ramdisk_directory=orbax.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )

        end_time = validation_util.generate_timestamp.override(
            task_id="generate_end_time"
        )()

        validate_restore_step = (
            validation_util.validate_restored_correct_checkpoint(
                project_id=test_config.cluster.project,
                location=zone_to_region(test_config.cluster.zone),
                cluster_name=test_config.cluster.name,
                interrupt_at_step=40,
                start_time=start_time,
                end_time=end_time,
            )
        )

        log_filters = [
            "jsonPayload.message:\"'event_type': 'emergency_restore'\"",
            "jsonPayload.message:\"'is_restoring_slice': True\"",
            "jsonPayload.message:\"'directory': '/local/\"",
        ]
        validate_restored_source = validation_util.validate_log_exist.override(
            task_id="validate_emc_restored_from_local"
        )(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            text_filter=" AND ".join(log_filters),
            start_time=start_time,
            end_time=end_time,
        )

        steps_to_validate = test_config.generate_step_to_validate(is_local=True)

        validate_log = validation_util.validate_checkpoint_at_steps_are_saved(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            start_time=start_time,
            end_time=end_time,
            steps_to_validate=steps_to_validate,
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
            >> validate_log
            >> wait_delete_cpc_final
        )
