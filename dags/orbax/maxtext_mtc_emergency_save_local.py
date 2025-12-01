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
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.orbax.util import validation_util
from dags.orbax.util import checkpoint_util
from xlml.utils.gke import zone_to_region
from dags.orbax.util import test_config_util


SCHEDULE = "45 5 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_emc_and_mtc_orbax_save_local"


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
        "TPU",
        "v5p-128",
    ],
    description="A DAG to run MaxText multi-tier checkpointing tests.",
    doc_md="""
      # Emergency Checkpoint Manager and Multi-tier Checkpoint Validation DAG

      ### Description
      This DAG (Directed Acyclic Graph) automates the process of validating
      checkpoint saving when using both **Emergency Checkpoint Manager**
      and **Multi-tier Checkpoint** features. The flag that controls the
      behaviour of using MTC or EMC is **user_replicatior**.
      Also the steps flag controls how many steps the job will run.

      ### Prerequisites
      To run this test, you need an existing cluster with the Multi-tier
      Checkpointing configuration enabled, as well as a bucket with HNS
      (Hierarchical Namespace) enabled.

      ### Procedures
      1.  **Apply Configuration:** A Checkpoint Configuration YAML file is
      applied to the cluster, enabling Multi-tier Checkpoint (MTC) features.
      2.  **Run Maxtext Jobsets:** The DAG runs a Maxtext jobset twice.
      One run has `replicator_enabled` set to `True` (for MTC), and the
      other has it set to `False` (for Emergency Checkpoint Manager).
      3.  **Validate Checkpoints:** The DAG validates that **local checkpoints**
      are being saved correctly in the `local/` directory for both job runs.

      4.  The validation logic is the same for both the MTC and Emergency
      Checkpoint Manager job runs because their local checkpoint saving
      behavior is similar. This is why a single validation task is used for both.
    """,
    concurrency=2,
) as dag:
  # Only one set of test configurations (e.g., v5p-128) is supported at the moment.
  # Other configurations (e.g., v5e and/or v6e) may be introduced later.
  test_configs = [
      test_config_util.TestConfig(
          cluster=XpkClusters.TPU_V5P_128_CLUSTER,
          machine_type="ct5p-hightpu-4t",
          accelerator="v5p-128",
          slices=[2],
          model_name="llama2-7b",
          short_id="max-sv-loc",
          multi_tier_checkpointing_backup_interval_minutes=30,
          steps=100,
          checkpoint_period=200,
          local_checkpoint_period=20,
          base_dir=test_config_util.DEFAULT_BUCKET,
      ),
  ]

  task_groups = []

  for checkpointing in [
      test_config_util.Checkpointing(
          name="mtc",  # Multi-tier Checkpointing
          enable_multi_tier_checkpointing=True,
      ),
      test_config_util.Checkpointing(
          name="emc",  # Emergency Checkpointing
          enable_multi_tier_checkpointing=False,
      ),
  ]:
    with TaskGroup(
        group_id=f"maxtext_{checkpointing.name}_orbax_save_local",
    ) as group:
      for mode, image in test_config_util.DOCKER_IMAGES:
        for test_config in test_configs:
          for slice_num in test_config.slices:
            # We conditionally set the trigger_rule on the first task.
            # If first task group failed the next one can execute.
            wait_delete_cpc = checkpoint_util.wait_for_cpc_deletion.override(
                trigger_rule=TriggerRule.ALL_DONE
            )(test_config.cpc_config)
            apply_cpc = checkpoint_util.apply_cpc(test_config.cpc_config)

            # Generate consistent run name.
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
                out_folder=f"maxtext_{checkpointing.name}_orbax_save_local",
                enable_multi_tier_checkpointing=checkpointing.enable_multi_tier_checkpointing,
            )

            start_time = validation_util.generate_timestamp()
            maxtext_chkpt_run_test = gke_config.get_gke_config(
                num_slices=slice_num,
                cluster=test_config.cluster,
                time_out_in_min=60,
                test_name=f"{test_config.short_id}-{checkpointing.name}",
                run_model_cmds=workload_command,
                docker_image=image.value,
                test_owner=test_owner.CAMILO_Q,
            ).run(
                ramdisk_directory=test_config_util.DEFAULT_RAM_DISK,
                mtc_enabled=True,
                skip_post_process=True,
                max_restart=15,
            )

            end_time = validation_util.generate_timestamp()

            steps_to_validate = test_config.generate_step_to_validate(
                is_local=True
            )

            validate_local_check_steps = (
                validation_util.validate_checkpoint_at_steps_are_saved(
                    project_id=test_config.cluster.project,
                    location=zone_to_region(test_config.cluster.zone),
                    cluster_name=test_config.cluster.name,
                    ram_disk=test_config_util.DEFAULT_RAM_DISK,
                    start_time=start_time,
                    end_time=end_time,
                    steps_to_validate=steps_to_validate,
                )
            )
            # Final CPC cleanup to ensure symmetric start/end
            wait_delete_cpc_final = (
                checkpoint_util.wait_for_cpc_deletion.override(
                    trigger_rule=TriggerRule.ALL_DONE,
                    task_id="wait_delete_cpc_final",
                )(test_config.cpc_config).as_teardown(setups=apply_cpc)
            )

            (
                wait_delete_cpc
                >> apply_cpc
                >> run_name
                >> start_time
                >> maxtext_chkpt_run_test
                >> end_time
                >> validate_local_check_steps
                >> wait_delete_cpc_final
            )
      # Add to a list of test to chain them sequentially.
      task_groups.append(group)

  # Chain all task groups sequentially.
  for idx in range(len(task_groups) - 1):
    task_groups[idx] >> task_groups[idx + 1]
