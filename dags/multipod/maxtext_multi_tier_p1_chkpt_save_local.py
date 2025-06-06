"""Add commentMore actions
A DAG to run MaxText multi-tier checkpointing tests (phase2: save & validate).
"""

import datetime
from datetime import timezone, timedelta
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import TpuVersion, Zone, DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode
from xlml.utils import log_explorer
from xlml.utils import xpk

SCHEDULE = None if not composer_env.is_prod_env() else "0 10 * * *"

with models.DAG(
    dag_id="maxtext_multi_tier_p1_sav01_save_local",
    schedule_interval=SCHEDULE,
    tags=[
        "multipod_team",
        "maxtext",
        "multi_tier_p1_chkpt_save_local",
        "nightly",
    ],
    start_date=datetime.datetime(2025, 6, 6),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.ERNIE_BASE_OUTPUT_DIR}/maxtext_multi_tier_p1_sav01_save_local"
  )
  dataset_path = gcs_bucket.MLPERF_LLM_DIR
  docker_images = [(
      SetupMode.JAX_STABLE_STACK,
      DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
  )]
  ram_disk = "/local"
  test_configs = {"v5p-8": [2]}
  clusters = {"v5p-8": XpkClusters.TPU_V5P_8_CLUSTER_ERNIE_CIENET}
  step = 100
  local_checkpoint_period = 20
  replicator_backup_interval_minutes = "1"
  use_replicator = "False"
  name_prefix = "maxtext_phase2_chkpt_save"

  for mode, image in docker_images:
    for accelerator, slices in test_configs.items():
      for slice_num in slices:
        run_time = datetime.datetime.now().strftime("%Y-%m-%d-%H")
        run_name = f"{name_prefix}-{slice_num}x-{accelerator}_{run_time}"
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
            f"run_name={run_name} dataset_path={dataset_path}",
        )

        start_time = xpk.generate_timestamp()

        # make launch test_name unique
        maxtext_phase2_chkpt_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"maxtext_phase1_chkpt_save",
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
            test_name=f"maxtext_phase1_chkpt_test-cleanup",
            run_model_cmds=cleanup_command,
            docker_image=image.value,
            test_owner=test_owner.ERNIE_C,
        ).run(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="main",
            skip_post_process=True,
        )

        vali_step = step - 1
        vali_step_list = [i for i in range(0, vali_step, local_checkpoint_period)]
        vali_step_list.append(vali_step)

        end_time = xpk.generate_timestamp()
        validate_log = log_explorer.validate_log_with_step(
            project_id=clusters[accelerator].project,
            location=clusters[accelerator].zone[:-2],
            cluster_name=clusters[accelerator].name,
            text_filter="Finished asynchronous save `(blocking` `+` `background)` in",
            start_time=start_time,
            end_time=end_time,
            vali_step_list=vali_step_list,
            validation_string="seconds to /local/",
        )

        (
            start_time
            >> maxtext_phase2_chkpt_test
            >> ram_disk_cleanup
            >> end_time
            >> validate_log
        )
