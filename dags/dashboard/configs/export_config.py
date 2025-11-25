import json
import logging
from typing import Callable
from dataclasses import dataclass

import pandas as pd
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator

from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.exceptions import AirflowException

from dags.common import test_owner

# Prefix for exported files in GCS bucket
GCS_PREFIX = "airflow_exports"

GCS_SCHEMA_PREFIX = "airflow_schemas"

# Connection IDs for Airflow connections
GCP_CONN_ID = "google_cloud_default"
POSTGRES_CONN_ID = "airflow_db"

# BigQuery location/region
BQ_LOCATION = "US"

# Max timestamp allowed in BigQuery to clean invalid timestamps
MAX_BQ_TIMESTAMP = pd.Timestamp("9999-12-31 23:59:59.999999", tz="UTC")

# Mapping Postgres data types to BigQuery types
PG_TO_BQ = {
    "integer": "INTEGER",
    "bigint": "INTEGER",
    "smallint": "INTEGER",
    "boolean": "BOOLEAN",
    "text": "STRING",
    "character varying": "STRING",
    "varchar": "STRING",
    "timestamp with time zone": "TIMESTAMP",
    "timestamp without time zone": "TIMESTAMP",
    "date": "DATE",
    "double precision": "FLOAT",
    "real": "FLOAT",
    "numeric": "FLOAT",
    "json": "STRING",
    "jsonb": "STRING",
    "bytea": "BYTES",
}

COMMON_INT_COLS = [
    "job_id",
    "queued_by_job_id",
    "pid",
    "trigger_id",
    "creating_job_id",
    "map_index",
    "try_number",
    "log_template_id",
]


@dataclass
class AirflowTable:
  """A class to set up Airflow table

  Attributes:
    table_name (str): The name of the source table to query for data.
    time_field_for_filtering (str): The name of the column containing
        the timestamp or date used for filtering.
    time_frame (str): A string specifying the time duration for filtering,
        e.g., '1 year', '30 days'.
    post_actions (list[Callable[[pd.DataFrame], any]]): A list of functions
        to be executed sequentially on the pandas DataFrame.
  """

  table_name: str
  time_field_for_filtering: str
  time_frame: str
  post_actions: list[Callable[[pd.DataFrame], any]]


# List of Airflow metadata tables to export and load
TABLES: list[AirflowTable] = [
    # Stores the high-level metadata for each DAG.
    AirflowTable(
        table_name="dag",
        time_field_for_filtering="",
        time_frame="",
        post_actions=[lambda df: clean_timestamp(df, "last_parsed_time")],
    ),
    # Tracks each execution instance of a DAG.
    AirflowTable(
        table_name="dag_run",
        time_field_for_filtering="updated_at",
        time_frame="60 days",
        post_actions=[
            lambda df: dataframe_inplace_apply(df, "conf", safe_json)
        ],
    ),
    # Contains the tags used for filtering and organizing DAGs.
    AirflowTable(
        table_name="dag_tag",
        time_field_for_filtering="",
        time_frame="",
        post_actions=[],
    ),
    # Records the status and details for each task execution.
    AirflowTable(
        table_name="task_instance",
        time_field_for_filtering="updated_at",
        time_frame="60 days",
        post_actions=[
            lambda df: dataframe_inplace_apply(df, "executor_config", safe_json)
        ],
    ),
    # Logs information about failed task instances.
    AirflowTable(
        table_name="task_fail",
        time_field_for_filtering="start_date",
        time_frame="60 days",
        post_actions=[lambda df: cast_int(df, "duration")],
    ),
    # Holds the rendered parameters and templates for each task instance.
    AirflowTable(
        table_name="rendered_task_instance_fields",
        time_field_for_filtering="",
        time_frame="",
        post_actions=[
            lambda df: dataframe_inplace_apply(df, "rendered_fields", safe_json)
        ],
    ),
    # Stores the serialized DAG files for quick parsing and loading.
    AirflowTable(
        table_name="serialized_dag",
        time_field_for_filtering="",
        time_frame="",
        post_actions=[
            lambda df: dataframe_inplace_apply(df, "data", safe_json)
        ],
    ),
]


def get_export_operator(source_table: AirflowTable):
  return PythonOperator(
      task_id=f"export_{source_table.table_name}",
      owner=test_owner.SEVERUS_H,
      python_callable=export_table_schema_and_data,
      op_kwargs={"table": source_table},
  )


def get_gcs_to_bq_operator(
    source_table: AirflowTable, source_bucket: str, destination_table: str
):
  # Loads files from Google Cloud Storage into BigQuery
  table_name = source_table.table_name
  return GCSToBigQueryOperator(
      task_id=f"load_{table_name}_to_bq",
      owner=test_owner.SEVERUS_H,
      bucket=source_bucket,
      source_objects=[f"{GCS_PREFIX}/{table_name}_part_*.json"],
      destination_project_dataset_table=destination_table,
      schema_object=f"{GCS_SCHEMA_PREFIX}/{table_name}_schema.json",
      source_format="NEWLINE_DELIMITED_JSON",
      write_disposition="WRITE_TRUNCATE",
      autodetect=False,
      location=BQ_LOCATION,
      gcp_conn_id=GCP_CONN_ID,
  )


def fetch_schema(table):
  """
  Fetch the schema from Postgres and map it to BigQuery schema format.
  Returns a list of dicts with 'name', 'type', and 'mode' keys.
  """
  pg = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
  rows = pg.get_records(
      f"""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = '{table}'
    ORDER BY ordinal_position;
  """
  )
  schema = []
  for col, dtype, is_null in rows:
    bqtype = PG_TO_BQ.get(dtype, "STRING")
    mode = "NULLABLE" if is_null == "YES" else "REQUIRED"
    schema.append({"name": col, "type": bqtype, "mode": mode})
  return schema


def cast_int(df, col):
  """Please look into the errors[] collection for more details
  Cast a column to nullable integer (Int64) after coercing errors.
  """
  if col in df:
    df[col] = pd.to_numeric(df[col], errors="coerce").dropna().astype("Int64")


def clean_timestamp(df, col):
  """
  Convert column to UTC timestamp and replace out-of-bound values with NaT.
  """
  if col in df:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df[col] = df[col].apply(
        lambda x: x if pd.isnull(x) or x <= MAX_BQ_TIMESTAMP else pd.NaT
    )


def dataframe_inplace_apply(df: pd.DataFrame, col: str, apply_func: Callable):
  df[col] = df.get(col).apply(apply_func)


def safe_json(obj):
  """
  Safely convert Python objects to JSON strings, return None if error.
  """
  try:
    if isinstance(obj, memoryview):
      obj = obj.tobytes()
    return json.dumps(obj)
  except Exception as e:
    logging.error(f"Exception: {e}")


def export_table_schema_and_data(table: AirflowTable, **kwargs):
  """
  Export data from Postgres to GCS as newline-delimited JSON.
  Cleans old exported files before upload to prevent stale data.
  """
  # Resolve GCS bucket param from DAG run config or DAG params
  gcs_bucket_param = kwargs["dag_run"].conf.get("target_gcs_bucket") or kwargs[
      "params"
  ].get("target_gcs_bucket")
  if not gcs_bucket_param:
    raise AirflowException(
        f"Missing required 'target_gcs_bucket' parameter for export_table. "
        f"Please ensure a value is provided via Airflow Variables or manual trigger."
    )

  table_name = table.table_name
  logging.info(f"export table {table_name}.")
  pg = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
  gcs = GCSHook(gcp_conn_id=GCP_CONN_ID)

  # Clean up previous export files for this table in GCS
  existing_objects = gcs.list(
      bucket_name=gcs_bucket_param, prefix=f"{GCS_PREFIX}/{table_name}_part_"
  )
  for obj in existing_objects:
    gcs.delete(bucket_name=gcs_bucket_param, object_name=obj)
  logging.info(
      f"Cleaned up {len(existing_objects)} existing JSON files for {table_name}."
  )

  # Query entire table into a Pandas dataframe
  select_clause = f"SELECT * FROM {table_name}"
  if table.time_field_for_filtering and table.time_frame:
    select_clause += f" WHERE {table.time_field_for_filtering} >= NOW() - INTERVAL '{table.time_frame}'"

  logging.info(f"sql = '{select_clause}'")
  df = pg.get_pandas_df(select_clause)

  # Table-specific post actions
  for action in table.post_actions:
    action(df)

  for col in COMMON_INT_COLS:
    cast_int(df, col)

  if len(df) == 0:
    logging.warning(f"No data to export for {table_name}.")
    return
  chunk_size = 100000  # Upload in chunks for large tables
  # Calculate the number of chunks for all cases.
  # If the DataFrame is smaller than chunk_size, num_chunks will be 1.
  num_chunks = (len(df) + chunk_size - 1) // chunk_size
  # Loop through all chunks, regardless of the DataFrame's size.
  for i in range(num_chunks):
    chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
    chunk_data = chunk_df.to_json(
        orient="records", lines=True, date_format="iso"
    )
    path = f"{GCS_PREFIX}/{table_name}_part_{i}.json"
    gcs.upload(
        bucket_name=gcs_bucket_param,
        object_name=path,
        data=chunk_data,
        mime_type="application/json",
    )

  logging.info(f"Exported {table_name} in {num_chunks} chunk(s)")

  schema = fetch_schema(table_name)
  schema_json = json.dumps(schema, indent=2)
  gcs.upload(
      bucket_name=gcs_bucket_param,
      object_name=f"{GCS_SCHEMA_PREFIX}/{table_name}_schema.json",
      data=schema_json,
      mime_type="application/json",
  )
  logging.info(f"Exported {table_name} schema")
