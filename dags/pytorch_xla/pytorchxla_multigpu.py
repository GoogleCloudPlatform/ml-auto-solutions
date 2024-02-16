import datetime
from airflow import models

from dags.vm_resource import Project
from xlml.apis import gcp_config, test_config, task


US_CENTRAL1 = gcp_config.GCPConfig(
    Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    # HACK: use region in place of zone, since clusters are regional
    zone='us-central1',
    dataset_name=...,
)


with models.DAG(
    dag_id="pytorchxla-multigpu",
    schedule=None,
    tags=["pytorchxla", "latest", "supported", "xlml"],
    catchup=False,
    start_date=datetime.datetime(2023, 7, 12),
):
    resnet_v100_2x2 = task.GpuGkeTask(test_config.JSonnetGpuTest.from_pytorch('pt-nightly-resnet50-mp-fake-v100-x2x2'), US_CENTRAL1, 'wcromar-test-cluster').run()
