"""
A DAG to run MaxText multi-tier checkpointing tests (phase2: save & validate).
"""

import datetime
from datetime import timezone
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from xlml.utils import xpk

SCHEDULE = None if not composer_env.is_prod_env() else "0 10 * * *"

with models.DAG(
    dag_id="maxtext_multi_tier_sav02_save_gcs",
    schedule_interval=SCHEDULE,
    start_date=datetime.datetime(2025, 5, 28),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_multi_tier_sav02_save_gcs"
  )
  dataset_path = gcs_bucket.MLPERF_LLM_DIR
  docker_images = [(
      SetupMode.JAX_STABLE_STACK,
      DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
  )]
  ram_disk = "/local"
  test_configs = {"v5p-8": [2]}
  clusters = {"v5p-8": XpkClusters.TPU_V5P_8_CLUSTER}
  step = "20"
  local_checkpoint_period = "10"
  replicator_backup_interval_minutes = "1"
  use_replicator = "True"
  start_time = datetime.datetime.now(timezone.utc)

  for mode, image in docker_images:
    for accelerator, slices in test_configs.items():
      for slice_num in slices:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        run_name = (
            f"maxtext_phase2_chkpt_test-{slice_num}x-{accelerator}_{run_time}"
        )
        workload_command = (
            "export TPU_PREMAPPED_BUFFER_SIZE=52428800000 && "
            "export TPU_PREMAPPED_BUFFER_TRANSFER_THRESHOLD_BYTES=52428800000 && "
            "python3 -m MaxText.train MaxText/configs/base.yml remat_policy=full "
            f"global_parameter_scale=1 base_output_directory={base_output_directory} "
            f"dataset_type=synthetic steps={step} per_device_batch_size=1 "
            "ici_fsdp_parallelism=-1 ici_tensor_parallelism=4 max_target_length=256 "
            "reuse_example_batch=1 enable_emergency_checkpoint=true "
            f"local_checkpoint_directory={ram_disk} local_checkpoint_period={local_checkpoint_period} "
            f"use_replicator_service={use_replicator} replicator_backup_interval_minutes={replicator_backup_interval_minutes} "
            f"run_name={run_name} dataset_path={dataset_path}",
        )

        # make launch test_name unique
        maxtext_phase2_chkpt_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"maxtext_phase2_chkpt_save",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.ERNIE_C,
        ).run(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="main",
            skip_post_process=True,
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

        validate_gcs = xpk.validate_saving_checkpoint(base_output_directory)
        
        vali_step = int(step) - 1
        end_time = datetime.datetime.now(timezone.utc)
        validate_log = xpk.list_log_entries(
            project_id=clusters[accelerator].project,
            location=clusters[accelerator].zone[:-2],
            cluster_name=clusters[accelerator].name,
            pod_pattern="*",
            text_filter=f"completed step: {str(vali_step)},",
            start_time=start_time,
            end_time=end_time,
          )
        
        (
            maxtext_phase2_chkpt_test
            >> ram_disk_cleanup
            >> validate_gcs
            >> validate_log
        )

