"""
A DAG to run MaxText regular checkpointing tests.

This DAG performs a series of tests to save and validate checkpoints
for the MaxText model using the regular checkpointer.
The tests are executed on a TPU multi-pod cluster.
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.orbax.util import validation_util, test_config_util
from xlml.utils.xpk import MAIN_BRANCH
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 12 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_regular_save"

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
    description="DAG that verifies MaxText regular checkpointing functionality to GCS bucket",
    doc_md="""
      # MaxText Regular Checkpointing Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      regular checkpoint saving for the MaxText model. It runs a single
      MaxText training job without emergency or multi-tier checkpointing
      features and validates that checkpoints are saved correctly to GCS
      at specified intervals.

      ### Prerequisites
      - An existing TPU cluster configured for MaxText training.
      - A GCS bucket for storing logs and checkpoints.

      ### Procedures
      1.  **Run MaxText Training:** A MaxText training job is initiated with
          regular checkpointing enabled. The job runs for 100 steps and saves
          checkpoints to GCS every 25 steps.
      2.  **Validate Checkpoint Logs:** The DAG inspects the application logs
          to confirm that checkpoint save events occurred at expected steps
          (0, 25, 50, 75, 99).
      3.  **Validate GCS Files:** The DAG verifies that checkpoint files are
          correctly saved in the GCS bucket for all expected steps.
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
          steps=100,
          checkpoint_period=25,
          base_dir=test_config_util.DEFAULT_BUCKET,
      ),
  ]
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
