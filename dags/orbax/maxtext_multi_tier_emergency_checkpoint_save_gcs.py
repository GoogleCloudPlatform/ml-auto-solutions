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

SCHEDULE = "0 10 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_emtc_orbax_sav02_save_gcs",
    schedule_interval=SCHEDULE,
    tags=[
        "multipod_team",
        "maxtext",
        "multi_tier_p2_chkpt_save_local",
        "nightly",
        "orbax",
    ],
    start_date=datetime.datetime(2025, 6, 30),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = (
      f"{gcs_bucket.MTC_AUTOMATION_BUCKET}/maxtext_emtc_orbax_sav02_save_gcs"
  )
  docker_images = [(
      SetupMode.NIGHTLY,
      DockerImage.MAXTEXT_TPU_JAX_NIGHTLY,
  )]
  ram_disk = "/local"
  test_configs = {"v5p-128": [2]}
  clusters = {"v5p-128": XpkClusters.TPU_V5P_128_CLUSTER_ORBAX}
  step = 200
  local_checkpoint_period = 20
  checkpoint_period = 20
  replicator_backup_interval_minutes = 1
  use_replicator = "True"
  model_name = "llama2-7b"
  name_prefix = "maxtext-p2-cpt-sv-gcs"

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
        delete_cpc = orbax.delete_cpc(*cpc)
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
            f"use_replicator_service={use_replicator} replicator_backup_interval_minutes={replicator_backup_interval_minutes} "
            f"run_name={run_name}",
        )

        start_time = log_explorer.generate_timestamp()

        # make launch test_name unique
        maxtext_chkpt_test = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"{name_prefix}",
            run_model_cmds=workload_command,
            docker_image=image.value,
            test_owner=test_owner.CAMILO,
        ).run(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="abhinav-mtc",
            skip_post_process=True,
        )

        # cleanup run: unique test_name
        cleanup_command = (f"rm -rf {ram_disk}/*",)
        ram_disk_cleanup = gke_config.get_gke_config(
            num_slices=slice_num,
            cluster=clusters[accelerator],
            time_out_in_min=60,
            test_name=f"{name_prefix}-cleanup",
            run_model_cmds=cleanup_command,
            docker_image=image.value,
            test_owner=test_owner.CAMILO,
        ).run(
            ramdisk_directory=ram_disk,
            mtc_enabled=True,
            xpk_branch="abhinav-mtc",
            skip_post_process=True,
        )

        vali_step = step - 1
        vali_step_list = [
            i for i in range(0, vali_step, local_checkpoint_period)
        ]
        vali_step_list.append(vali_step)

        end_time = log_explorer.generate_timestamp()

        # We need to get logs from replicator_worker from inside mtc driver.
        # Here we are looking for the string 'Successful: backup for step'. This will tell us that the
        # # checkpoint were backup succesfully. Since all replicator workers need to aggre before the backup
        # We only need logs from one pod.
        validate_gcs_bucket = log_explorer.validate_log_with_gcs(
            project_id=clusters[accelerator].project,
            location=clusters[accelerator].zone[:-2],
            cluster_name=clusters[accelerator].name,
            text_filter="Successful: backup for step",
            namespace="gke-managed-checkpointing",
            container_name="replication-worker",
            pod_pattern="multitier-driver",
            start_time=start_time,
            end_time=end_time,
            bucket_name=f"{gcs_bucket.MTC_AUTOMATION_BUCKET}/{run_name}",
        )
        (
            delete_cpc
            >> apply_cpc
            >> start_time
            >> maxtext_chkpt_test
            >> ram_disk_cleanup
            >> end_time
            >> validate_gcs_bucket
        )
