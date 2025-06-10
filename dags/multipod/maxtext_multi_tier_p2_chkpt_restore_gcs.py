"""
A DAG to run MaxText multi-tier checkpointing tests (phase1: save & validate).
"""

import datetime
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from xlml.utils import xpk
from xlml.utils import log_explorer

SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_multi_tier_res02_restore_gcs",
    schedule_interval=SCHEDULE,
    tags=[
        "multipod_team",
        "maxtext",
        "multi_tier_p2_chkpt_restore_gcs",
        "nightly",
    ],
    start_date=datetime.datetime(2025, 6, 6),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_multi_tier_res02_restore_gcs"
  )
  docker_images = [(
      SetupMode.JAX_STABLE_STACK,
      DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
  )]
  ram_disk = "/local"
  test_configs = {"v5p-8": [2]}
  clusters = {"v5p-8": XpkClusters.TPU_V5P_8_CLUSTER}
  step = 200
  restore_step = 300
  local_checkpoint_period = 20
  replicator_backup_interval_minutes = 1
  use_replicator = "True"

  for mode, image in docker_images:
    for accelerator, slices in test_configs.items():
      for slice_num in slices:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        run_name = f"maxtext-phase2-chkpt-test-{slice_num}x-{accelerator}-{run_time}"
        bucket_name = f"{gcs_bucket.ERNIE_BASE_OUTPUT_DIR}/{run_name}"
        workload_command = (
            "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
            "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
            "python3 -m MaxText.train MaxText/configs/base.yml remat_policy=full "
            f"global_parameter_scale=1 base_output_directory={base_output_directory} "
            f"dataset_type=synthetic steps={step} per_device_batch_size=1 "
            "max_target_length=256 "
            "reuse_example_batch=1 enable_emergency_checkpoint=true "
            f"local_checkpoint_directory={ram_disk} local_checkpoint_period={local_checkpoint_period} "
            f"use_replicator_service={use_replicator} replicator_backup_interval_minutes={replicator_backup_interval_minutes} "
            f"run_name={run_name}",
        )
        workload_command_restore = (
            "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
            "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
            "python3 -m MaxText.train MaxText/configs/base.yml remat_policy=full "
            f"global_parameter_scale=1 base_output_directory={base_output_directory} "
            f"dataset_type=synthetic steps={restore_step} per_device_batch_size=1 "
            "max_target_length=256 "
            "reuse_example_batch=1 enable_emergency_checkpoint=true "
            f"local_checkpoint_directory={ram_disk} local_checkpoint_period={local_checkpoint_period} "
            f"use_replicator_service={use_replicator} replicator_backup_interval_minutes={replicator_backup_interval_minutes} "
            f"run_name={run_name}",
        )

        workload_id = xpk.generate_workload_id(f'{run_name}')

        start_time = xpk.generate_timestamp()

        # make launch test_name unique
        maxtext_phase2_chkpt_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"maxtext_phase2_chkpt_save",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.ERNIE_C,
        ).run_with_workload_id(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="main",
            skip_post_process=True,
            workload_id=workload_id,
        )

        # cleanup run: unique test_name
        cleanup_command = (f"rm -rf {ram_disk}/*",)
        ram_disk_cleanup = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"maxtext_phase2_chkpt_test-cleanup",
            run_model_cmds=cleanup_command,
            docker_image=image.value,
            test_owner=test_owner.ERNIE_C,
        ).run(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="main",
            skip_post_process=True,
        )

        end_time = xpk.generate_timestamp()
        validate_gcs_bucket_save_step = log_explorer.validate_log_with_gcs(
            project_id=clusters[accelerator].project,
            location=clusters[accelerator].zone[:-2],
            cluster_name=clusters[accelerator].name,
            text_filter="Successful: backup for step",
            namespace="gke-managed-checkpointing",
            container_name="replication-worker",
            pod_pattern="multitier-driver",
            start_time=start_time,
            end_time=end_time,
            bucket_name=bucket_name,
        )

        restore_start_time = xpk.generate_timestamp()

        maxtext_phase2_chkpt_restore = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"maxtext_phase2_chkpt_restore",
            run_model_cmds=workload_command_restore,
            docker_image=image.value,
            test_owner=test_owner.ERNIE_C,
        ).run_with_workload_id(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="main",
            skip_post_process=True,
            workload_id=workload_id,
        )

        # cleanup run: unique test_name
        cleanup_command = (f"rm -rf {ram_disk}/*",)
        ram_disk_cleanup_restore = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"maxtext_phase2_chkpt_test-cleanup2",
            run_model_cmds=cleanup_command,
            docker_image=image.value,
            test_owner=test_owner.ERNIE_C,
        ).run(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="main",
            skip_post_process=True,
        )

        restore_end_time = xpk.generate_timestamp()

        validate_gcs_bucket_restore_step = log_explorer.validate_log_exist(
            project_id=clusters[accelerator].project,
            location=clusters[accelerator].zone[:-2],
            cluster_name=clusters[accelerator].name,
            text_filter=f"Restoring from backup checkpoint {validate_gcs_bucket_save_step}",
            namespace="gke-managed-checkpointing",
            container_name="replication-worker",
            pod_pattern="multitier-driver",
            start_time=restore_start_time,
            end_time=restore_end_time,
        )

        validate_gcs_bucket_restore_file = log_explorer.validate_log_exist(
            project_id=clusters[accelerator].project,
            location=clusters[accelerator].zone[:-2],
            cluster_name=clusters[accelerator].name,
            text_filter="copy backup/gcs/ to local/client/",
            namespace="gke-managed-checkpointing",
            container_name="replication-worker",
            pod_pattern="multitier-driver",
            start_time=restore_start_time,
            end_time=restore_end_time,
        )

        (
            start_time
            >> maxtext_phase2_chkpt_test
            >> ram_disk_cleanup
            >> end_time
            >> validate_gcs_bucket_save_step
            >> restore_start_time
            >> maxtext_phase2_chkpt_restore
            >> ram_disk_cleanup_restore
            >> restore_end_time
            >> validate_gcs_bucket_restore_step
            >> validate_gcs_bucket_restore_file
        )
