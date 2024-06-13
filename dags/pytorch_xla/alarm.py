from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.utils.email import send_email
import pandas as pd
import pandas_gbq

# Replace with your project ID and BigQuery table details
PROJECT_ID = 'your-project-id'
DATASET_ID = 'your-dataset-id'
TABLE_ID = 'your-table-id'
GCS_CONN_ID = 'google_cloud_default'
BQ_CONN_ID = 'google_cloud_default'

# Define the SQL query to read data from BigQuery
QUERY = f"""
SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
"""

default_args = {
    'owner': 'airflow',
    'email': ['pytorchxla-dev@google.com'],
    'email_on_failure': True,
    'email_on_retry': False,
}

with DAG(
    'bigquery_to_email',
    default_args=default_args,
    description='A simple DAG to read from BigQuery and send an email',
    schedule_interval='@daily',
    start_date=days_ago(1),
    tags=['example'],
) as dag:

  @task()
  def read_bigquery_table():
    # Read data from BigQuery
    df = pandas_gbq.read_gbq(QUERY, project_id=PROJECT_ID)
    # Convert DataFrame to HTML for the email body
    html_table = df.to_html()
    return html_table

  @task()
  def send_email_task(html_table):
    subject = 'BigQuery Data Report'
    body = f"""
        <h3>BigQuery Data:</h3>
        {html_table}
        """
    to = ['recipient@example.com']
    send_email(to=to, subject=subject, html_content=body)

  # Task dependencies
  html_table = read_bigquery_table()
  send_email_task(html_table)
