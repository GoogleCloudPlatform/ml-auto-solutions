from xlml.apis import gcp_config, metric_config, task, test_config
from dags import test_owner, gcs_bucket
from dags.multipod.configs import common
from dags.vm_resource import TpuVersion, Project, RuntimeVersion, Zone, ClusterName, DockerImage
import datetime


from airflow import models
from dags import composer_env
from dags.multipod.configs import common
from dags.multipod.configs.common import SetupMode, Platform

# bf16 and int8 convergence tests, each take ~5 hours to complete on a v4-128

# Run once a week at Saturday 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 6 * * SAT" if composer_env.is_prod_env() else None


with models.DAG(
    dag_id="maxtext_convergence",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "stable"],
    start_date=datetime.datetime(2024, 2, 27),
    catchup=False,
) as dag:
    
    current_time = datetime.datetime.now()
    current_date = current_time.strftime("%Y-%m-%d")
    base_output_directory = (
      f"{gcs_bucket.XLML_OUTPUT_DIR}/maxtext/stable/automated/{current_date}"
    )
    dataset_path = "gs://max-datasets-rogue"
    job_gcp_config = gcp_config.GCPConfig(
        project_name=Project.TPU_PROD_ENV_MULTIPOD.value,
        zone=Zone.US_CENTRAL2_B.value,
        dataset_name=metric_config.DatasetOption.XLML_DATASET
    )

    loss_threshold = 2.5
    base_convergence_command = f"bash end_to_end/test_convergence_1b_params.sh OUTPUT_PATH={base_output_directory} DATASET_PATH={dataset_path} LOSS_THRESHOLD={loss_threshold}"
    convergence_tests = {
        "maxtext-convergence-bf16" : ((base_convergence_command),),
        "maxtext-convergence-int8" : ((f"export M_QUANTIZATION=int8; {base_convergence_command}"))
    }

    for test_name, run_command in convergence_tests.items():
        job_test_config = test_config.TpuGkeTest(
            test_config.Tpu(
                version=TpuVersion.V4,
                cores=128,
            ),
            test_name=test_name,
            cluster_name=ClusterName.V4_128_MULTISLICE_CLUSTER.value,
            docker_image=DockerImage.MAXTEXT_JAX_STABLE.value,
            run_model_cmds=run_command,
            set_up_cmds=None,
            time_out_in_min=600,
            task_owner=test_owner.MATT_D,
            num_slices=1,
        )
        task.TpuXpkTask(
            task_test_config=job_test_config,
            task_gcp_config=job_gcp_config,
        ).run()
