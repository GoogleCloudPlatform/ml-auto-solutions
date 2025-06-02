"""Add commentMore actions
A DAG to run MaxText multi-tier checkpointing tests (phase1: save & validate).
"""

import datetime
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from xlml.utils import log_explorer
from xlml.utils import orbax
from xlml.utils.xpk import BRANCH_ABHINAV_MTC
from airflow.utils.task_group import TaskGroup


SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_emtc_orbax_res01_save_local",
    schedule_interval=SCHEDULE,
    tags=[
        "multipod_team",
        "maxtext",
        "multi_tier_p2_chkpt_res_local",
        "nightly",
        "orbax",
    ],
    start_date=datetime.datetime(2025, 6, 30),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.MTC_AUTOMATION_BUCKET}/maxtext_emtc_orbax_res01_save_local"
  )
  docker_images = [(
      SetupMode.NIGHTLY,
      DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
  )]
  ram_disk = "/local"
  test_configs = {"v5p-128": [2]}
  clusters = {"v5p-128": XpkClusters.TPU_V5P_128_CLUSTER_ORBAX}
  step = 250
  local_checkpoint_period = 20
  checkpoint_period = 20
  replicator_backup_interval_minutes = 30
  use_replicator_flags = ["False", "True"]
  model_name = "llama2-7b"
  name_prefix = "max-res-loc"
  tests_to_run_seq = []
  for i, use_replicator_flag in enumerate(use_replicator_flags):
    with TaskGroup(
        group_id=f"maxtext_emtc_orbax_sav01_save_local_replicator_{use_replicator_flag.lower()}"
    ) as task:
      for mode, image in docker_images:
        for accelerator, slices in test_configs.items():
          for slice_num in slices:
            cpc = (
                clusters[accelerator].project,
                clusters[accelerator].zone[:-2],
                clusters[accelerator].name,
                gcs_bucket.MTC_AUTOMATION_BUCKET.split("gs://")[1],
                "ct5p-hightpu-4t",
                "google.com/tpu",
                "800000Mi",
            )

            # This is the key change. We conditionally set the trigger_rule
            # on the very first task of each task group (except the first one).
            # Start from fresh apply cpc yaml mtc driver
            delete_cpc = (
                orbax.delete_cpc.override(trigger_rule="all_done")(*cpc)
                if i > 0
                else orbax.delete_cpc(*cpc)
            )
            apply_cpc = orbax.apply_cpc(*cpc)
            run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H")
            run_name = f"{name_prefix}-{slice_num}x-{accelerator}-{run_time}"

            workload_command = (
                "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
                "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
                "python3 -m MaxText.train MaxText/configs/base.yml remat_policy=full "
                f"global_parameter_scale=1 base_output_directory={base_output_directory} "
                f"dataset_type=synthetic steps={step} per_device_batch_size=1 "
                f"max_target_length=256 model_name={model_name} per_device_batch_size=2 "
                f"reuse_example_batch=1 enable_emergency_checkpoint=true checkpoint_period={checkpoint_period} "
                f"local_checkpoint_directory={ram_disk} local_checkpoint_period={local_checkpoint_period} "
                f"use_replicator_service={use_replicator_flag} replicator_backup_interval_minutes={replicator_backup_interval_minutes} "
                f"run_name={run_name}",
            )

            start_time = log_explorer.generate_timestamp()

            # make launch test with organic interruption of the node.
            maxtext_chkpt_test = gke_config.get_gke_config(
                num_slices=slice_num,
                cluster=clusters[accelerator],
                time_out_in_min=60,
                test_name=f"{name_prefix}-rep-{use_replicator_flag.lower()}",
                run_model_cmds=workload_command,
                docker_image=image.value,
                test_owner=test_owner.CAMILO,
            ).run_with_node_interruption(
                ramdisk_directory=ram_disk,
                mtc_enabled=True,
                xpk_branch=BRANCH_ABHINAV_MTC,
                skip_post_process=True,
            )

            # cleanup run: unique test_name
            cleanup_command = (f"rm -rf {ram_disk}/*",)
            ram_disk_cleanup = gke_config.get_gke_config(
                num_slices=slice_num,
                cluster=clusters[accelerator],
                time_out_in_min=60,
                test_name=f"{name_prefix}",
                run_model_cmds=cleanup_command,
                docker_image=image.value,
                test_owner=test_owner.CAMILO,
            ).run(
                ramdisk_directory=ram_disk,
                mtc_enabled=True,
                xpk_branch=BRANCH_ABHINAV_MTC,
                skip_post_process=True,
            )

            vali_step = step - 1
            vali_step_list = [
                i for i in range(0, vali_step, local_checkpoint_period)
            ]
            vali_step_list.append(vali_step)

            end_time = log_explorer.generate_timestamp()

            # We need to check that the restore event is happening.
            validate_is_restoring = log_explorer.validate_log_exist(
                project_id=clusters[accelerator].project,
                location=clusters[accelerator].zone[:-2],
                cluster_name=clusters[accelerator].name,
                text_filter="'event_type': 'restore'",
                start_time=start_time,
                end_time=end_time,
            )

            # Here we are looking for the string '(blocking + background)'.
            # We will compare expected steps with the ones we found when query this regex. Should be the same
            # If for some reason the restore start from 0 this task will fail because len(valid_step_list) != len(founded_steps)
            validate_log_step = log_explorer.validate_log_with_step(
                project_id=clusters[accelerator].project,
                location=clusters[accelerator].zone[:-2],
                cluster_name=clusters[accelerator].name,
                text_filter="(blocking + background).",
                start_time=start_time,
                end_time=end_time,
                vali_step_list=vali_step_list,
            )
            (
                delete_cpc
                >> apply_cpc
                >> start_time
                >> maxtext_chkpt_test
                >> ram_disk_cleanup
                >> end_time
                >> validate_is_restoring
                >> validate_log_step
            )
    # Add to a global list of test to be run in a sequential way
    tests_to_run_seq.append(task)

  # Chain the task groups sequentially
  for idx_test in range(len(tests_to_run_seq) - 1):
    tests_to_run_seq[idx_test] >> tests_to_run_seq[idx_test + 1]
