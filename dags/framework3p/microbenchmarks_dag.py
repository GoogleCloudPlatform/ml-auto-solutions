import datetime
from airflow import models
from dags import composer_env, vm_resource, test_owner
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

  all_gather_v4_16 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V4,
      tpu_cores=16,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B,
      time_out_in_min=60,
      runtime_version=vm_resource.RuntimeVersion.TPU_UBUNTU2204_BASE,
      project=vm_resource.Project.CLOUD_ML_AUTO_SOLUTIONS,
  )

  all_gather_v5e_4 = get_microbenchmark_config(
      tpu_version=vm_resource.TpuVersion.V5E,
      tpu_cores=4,
      tpu_zone=vm_resource.Zone.US_EAST1_C,
      time_out_in_min=60,
      runtime_version=vm_resource.RuntimeVersion.V2_ALPHA_TPUV5_LITE,
      project=vm_resource.Project.TPU_PROD_ENV_AUTOMATED,
      network=vm_resource.V5_NETWORKS,
      subnetwork=vm_resource.V5E_SUBNETWORKS,
  )

  all_gather_v5e_256 = get_microbenchmark_xpk_config(
      tpu_version=vm_resource.TpuVersion.V5E,
      tpu_cores=256,
      tpu_zone=vm_resource.Zone.US_WEST4_B,
      time_out_in_min=60,
      test_name="framework-microbenchmark-v5e-256",
      docker_image=vm_resource.DockerImage.XPK_JAX_TEST.value,
      test_owner=test_owner.QINY_Y,
      cluster=vm_resource.XpkClusters.TPU_V5E_256_CLUSTER,
      project=vm_resource.Project.TPU_PROD_ENV_MULTIPOD,      
  ).run()

# Test dependency: run in parallel
