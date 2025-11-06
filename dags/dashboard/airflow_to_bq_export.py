from airflow import DAG
from datetime import datetime
from airflow.models.param import Param
from airflow.models import Variable

from dags import composer_env
from dags.dashboard.configs import export_config

# Scheduled time
SCHEDULED_TIME = "15 0 * * *" if composer_env.is_prod_env() else None


# Load default config values from Airflow Variables
DEFAULT_GCP_PROJECT_ID = Variable.get(
    "gcp_target_project_id_default", default_var=""
)
DEFAULT_BQ_DATASET_ID = Variable.get(
    "bq_target_dataset_id_default", default_var=""
)
DEFAULT_GCS_BUCKET = Variable.get("gcs_target_bucket_default", default_var="")

params = {
    "target_project_id": Param(
        type="string",
        title="Target GCP Project ID",
        description="The Google Cloud Project ID where the data will be cloned.",
        default=DEFAULT_GCP_PROJECT_ID,
    ),
    "target_bigquery_dataset": Param(
        type="string",
        title="Target BigQuery Dataset ID",
        description="The BigQuery Dataset ID where the tables will be cloned.",
        default=DEFAULT_BQ_DATASET_ID,
    ),
    "target_gcs_bucket": Param(
        type="string",
        title="Target GCS Bucket Name",
        description="The GCS bucket name to use for temporary data export.",
        default=DEFAULT_GCS_BUCKET,
    ),
}

with DAG(
    dag_id="airflow_to_bq_export",
    description="""
  Export selected Airflow metadata tables from the Airflow Postgres metadata
  database to Google Cloud Storage as newline-delimited JSON files, and then
  load them into a BigQuery dataset.
  """,
    start_date=datetime(2025, 7, 1),
    schedule_interval=SCHEDULED_TIME,
    catchup=False,
    tags=["airflow", "bigquery", "gcs", "metadata", "export"],
    default_args={"retries": 0},
    params=params,
) as dag:
  for source_table in export_config.TABLES:
    # Define export task to run export_table Python function
    export_task = export_config.get_export_operator(source_table)
    destination_table = (
        "{{ params['target_project_id'] }}.{{ params['target_bigquery_dataset'] }}.%s"
        % source_table.table_name
    )
    load_task = export_config.get_gcs_to_bq_operator(
        source_table, "{{ params['target_gcs_bucket'] }}", destination_table
    )
    # Set task dependency: export -> load
    export_task >> load_task
