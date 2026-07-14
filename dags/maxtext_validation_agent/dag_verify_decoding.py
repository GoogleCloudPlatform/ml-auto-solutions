"""DAG to automate MaxText checkpoint decoding validation."""

import datetime
from airflow import models
from dags.maxtext_validation_agent.lib import utils

with models.DAG(
    dag_id="dag_verify_decoding",
    schedule=None,
    tags=["maxtext", "experimental", "checkpoint", "xlml"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
) as dag:

    # Trigger the validation task using a v4-8 TPU machine
    validate_checkpoint_job = utils.get_maxtext_validation_config(
        tpu_version="4",
        tpu_cores=8,
        tpu_zone="us-central2-b",
        time_out_in_min=45,
    ).run()
