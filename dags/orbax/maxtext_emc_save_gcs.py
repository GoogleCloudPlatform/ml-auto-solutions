"""
A DAG to run MaxText multi-tier checkpointing with replicator enabled
validates the local checkpoints are replicated (copy) to bucket
with HNS (Hierarchical Namespace)
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from dags.orbax.util import checkpoint_util
from dags.orbax.util import validation_util
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from xlml.utils.gke import zone_to_region
from dags.orbax.util import orbax


SCHEDULE = "0 12 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_emc_save_gcs"

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
    description="DAG that verifies the orbax multi-tier checkpointing saving functionality with replicator to GCS bucket",
    doc_md="""
      # Multi-tier Checkpoint Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      checkpoint saving when using **Emergency Checkpointer Manager** features.
      It will check that the checkpoints are being stored in the GCS bucket.
      Also the steps flag controls how many steps the job will run.

      ### Prerequisites
      To run this test, you need an existing cluster with the Multi-tier
      Checkpointing configuration enabled, as well as a bucket with HNS
      (Hierarchical Namespace) enabled.

      ### Procedures
      1.  **Apply Configuration:** A Checkpoint Configuration YAML file is
      applied to the cluster, enabling Multi-tier Checkpoint (MTC) features.
      2.  **Run Maxtext Jobsets:** The DAG runs a Maxtext jobset.
      3.  The DAG validates that **GCS checkpoints** are being saved correctly
      in the `GCS bucket` by checking bucket and pod logs.
    """,
    concurrency=2,
) as dag:
  checkpointing = orbax.Checkpointing(
      name="emc", enable_multi_tier_checkpointing=False
  )
  test_configs = [
      orbax.TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER_ORBAX,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-sv-loc",
          replicator_backup_time=30,
          step=75,
          local_checkpoint_step=20,
          checkpoint_step=25,
          ram_disk_size_in_mi="800000Mi",
          base_dir=orbax.DEFAULT_BUCKET,
      ),
  ]
  for mode, image in DOCKER_IMAGES:
    for test_config in test_configs:
      for slice_num in test_config.slices:
        # We conditionally set the trigger_rule on the first task.
        # If first task group failed the next one can execute.
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
            out_folder=f"maxtext_emc_orbax_save_gcs",
            enable_multi_tier_checkp=checkpointing.enable_multi_tier_checkpointing,
        )

        start_time = validation_util.generate_timestamp()

        maxtext_chkpt_run_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=test_config.cluster,
            time_out_in_min=60,
            test_name=f"{test_config.short_id}-emc",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.CAMILO_Q,
        ).run(
            ramdisk_directory=orbax.DEFAULT_RAM_DISK,
            mtc_enabled=True,
            xpk_branch=BRANCH_ABHINAV_MTC,
            skip_post_process=True,
        )

        end_time = validation_util.generate_timestamp()

        steps_to_validate = test_config.generate_step_to_validate(
            is_local=False
        )

        validate_steps = validation_util.validate_checkpoint_at_steps_are_saved(
            project_id=test_config.cluster.project,
            location=zone_to_region(test_config.cluster.zone),
            cluster_name=test_config.cluster.name,
            ram_disk="gcs",
            pod_pattern=".*-0-0",
            start_time=start_time,
            end_time=end_time,
            steps_to_validate=steps_to_validate,
        )

        # Validate that GCS restore happened during the second training run
        validate_checkpoints_steps_gcs = validation_util.validate_gcs_checkpoint_files(
            bucket_path=f"{orbax.DEFAULT_BUCKET}/maxtext_emc_orbax_save_gcs/{run_name}",
            steps_to_validate=steps_to_validate,
        )

        (
            wait_delete_cpc
            >> apply_cpc
            >> run_name
            >> start_time
            >> maxtext_chkpt_run_test
            >> end_time
            >> validate_steps
            >> validate_checkpoints_steps_gcs
        )
