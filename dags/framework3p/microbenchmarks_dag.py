import datetime
from airflow import models
from dags import composer_env
from dags.common import test_owner, vm_resource
from dags.framework3p.configs.microbenchmarks_config import get_microbenchmark_config, get_microbenchmark_xpk_config


# Run once a day at 2 am
SCHEDULED_TIME = "0 2 * * *" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="framework_microbenchmark",
    schedule=SCHEDULED_TIME,
    tags=["framework_team", "microbenchmark", "xlml"],
    start_date=datetime.datetime(2024, 9, 11),
    catchup=False,
) as dag:
  microbenchmarks_v4_8 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V4,
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B,
      time_out_in_min=120,
      runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE,
      project=vm_resource.Project.CLOUD_ML_AUTO_SOLUTIONS,
  )

  microbenchmarks_v4_16 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V4,
      tpu_cores=16,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B,
      time_out_in_min=120,
      runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE,
      project=vm_resource.Project.CLOUD_ML_AUTO_SOLUTIONS,
  )

  microbenchmarks_v5p_8 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V5P,
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_EAST5_A,
      time_out_in_min=60,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5,
      project=vm_resource.Project.TPU_PROD_ENV_AUTOMATED,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5P_SUBNETWORKS,
  )

  microbenchmarks_v5p_256 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V5P,
      tpu_cores=256,
      tpu_zone=vm_resource.Zone.US_EAST5_A,
      time_out_in_min=60,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5,
      project=vm_resource.Project.TPU_PROD_ENV_AUTOMATED,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5P_SUBNETWORKS,
  )

  microbenchmarks_v5e_4 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V5E,
      tpu_cores=4,
      tpu_zone=vm_resource.Zone.US_EAST1_C,
      time_out_in_min=120,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5_LITE,
      project=vm_resource.Project.TPU_PROD_ENV_AUTOMATED,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5E_SUBNETWORKS,
  )

  microbenchmarks_v5e_16 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V5E,
      tpu_cores=16,
      tpu_zone=vm_resource.Zone.US_EAST1_C,
      time_out_in_min=60,
      runtime_version=vm_resource.RuntimeVersion.TPU_VM_TF_NIGHTLY_POD,
      project=vm_resource.Project.TPU_PROD_ENV_AUTOMATED,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5E_SUBNETWORKS,
  )

  microbenchmarks_v5e_256 = get_microbenchmark_xpk_config(
      time_out_in_min=60,
      test_name="framework-microbenchmark-v5e-256",
      docker_image=vm_resource.DockerImage.MICROBENCH_NIGHTLY.value,
      test_owner=test_owner.QINY_Y,
      cluster=vm_resource.XpkClusters.TPU_V5E_256_CLUSTER,
  ).run()

  microbenchmarks_v6e_256 = get_microbenchmark_xpk_config(
      time_out_in_min=60,
      test_name="framework-microbenchmark-v6e-256",
      docker_image=vm_resource.DockerImage.MICROBENCH_NIGHTLY.value,
      test_owner=test_owner.QINY_Y,
      cluster=vm_resource.XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
  ).run()


# Test dependency: run in parallel
