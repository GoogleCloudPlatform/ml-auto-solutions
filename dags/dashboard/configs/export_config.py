import json
import logging

import pandas as pd
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator

from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.exceptions import AirflowException

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


def get_export_operator(source_table: str):
  return PythonOperator(
      task_id=f"export_{source_table}",
      python_callable=export_table_schema_and_data,
      op_kwargs={"table_name": source_table},
  )


def get_gcs_to_bq_operator(
    source_table: str, source_bucket: str, destination_table: str
):
  # Loads files from Google Cloud Storage into BigQuery
  return GCSToBigQueryOperator(
      task_id=f"load_{source_table}_to_bq",
      bucket=source_bucket,
      source_objects=[f"{GCS_PREFIX}/{source_table}_part_*.json"],
      destination_project_dataset_table=destination_table,
      schema_object=f"{GCS_SCHEMA_PREFIX}/{source_table}_schema.json",
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
  """
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


def export_table_schema_and_data(table_name, **kwargs):
  """
  Export data from Postgres to GCS as newline-delimited JSON.
  Cleans old exported files before upload to prevent stale data.
  """
  # Resolve GCS bucket param from DAG run config or DAG params
  gcs_bucket_param = kwargs["dag_run"].conf.get("target_gcs_bucket") or kwargs[
      "params"
  ].get("target_gcs_bucket")
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

  if not gcs_bucket_param:
    raise AirflowException(
        f"Missing required 'target_gcs_bucket' parameter for export_table. "
        f"Please ensure a value is provided via Airflow Variables or manual trigger."
    )

  # Query entire table into a Pandas dataframe
  df = pg.get_pandas_df(f"SELECT * FROM {table_name}")
  common_int_cols = [
      "job_id",
      "queued_by_job_id",
      "pid",
      "trigger_id",
      "creating_job_id",
      "map_index",
      "try_number",
      "log_template_id",
  ]

  # Table-specific cleanup
  if table_name == "dag":
    clean_timestamp(df, "last_parsed_time")
  if table_name == "dag_run":
    df["conf"] = df.get("conf").apply(safe_json)
  if table_name == "task_instance":
    df["executor_config"] = df.get("executor_config").apply(safe_json)
  if table_name == "task_fail":
    cast_int(df, "duration")
  if table_name == "rendered_task_instance_fields":
    df["rendered_fields"] = df.get("rendered_fields").apply(safe_json)
  if table_name == "log":
    clean_timestamp(df, "dttm")
    df["extra"] = df.get("extra").apply(safe_json)
  if table_name == "serialized_dag":
    df["data"] = df.get("data").apply(safe_json)
  for col in common_int_cols:
    cast_int(df, col)

  chunk_size = 100000  # Upload in chunks for large tables
  num_chunks = 1

  if len(df) == 0:
    logging.warning(f"No data to export for {table_name}.")
    return
  if len(df) <= chunk_size:
    # Single chunk export
    data = df.to_json(orient="records", lines=True, date_format="iso")
    path = f"{GCS_PREFIX}/{table_name}_part_0.json"
    gcs.upload(
        bucket_name=gcs_bucket_param,
        object_name=path,
        data=data,
        mime_type="application/json",
    )
  else:
    # Multiple chunk export to handle large dataframes
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
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
