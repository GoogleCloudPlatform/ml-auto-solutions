# ================================================================
# Step 0: Imports & Global Config
# ================================================================
import enum
import json
import logging
from typing import List, Any, Dict
from datetime import datetime, timezone

import google
from airflow import DAG
from airflow.decorators import task
from airflow.models.param import Param
from airflow.models import Variable
from airflow.operators.python import get_current_context
from airflow.providers.google.suite.hooks.sheets import GSheetsHook
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, container_v1

from dags.common import test_owner

# --- Airflow DAG Schedule ---
SCHEDULED_TIME = "0 */4 * * *"  # Per 4 hours

# --- BigQuery Config ---
DEFAULT_PROJECT_ID = "cienet-cmcs"
DEFAULT_DATASET_ID = "amy_xlml_poc_prod"

# --- Google Sheets Config ---
GSPREAD_INSERT_ENABLED = True
DEFAULT_GSPREAD_SHEET_ID = Variable.get("buganizer_sheet_id", default_var="")
GSPREAD_WORKSHEET_NAME = "unhealthy_clusters"


# ================================================================
# Step 1: Enums & Constants
# ================================================================
class IssueType(enum.Enum):
  CLUSTER = "Cluster"
  NODE_POOL = "NodePool"
  CLUSTER_AND_NODE_POOL = "Cluster_Nodepool"


class Status(enum.Enum):
  NOT_EXIST = "NOT EXIST"
  ERROR = "ERROR"
  RUNNING = "RUNNING"
  RECONCILING = "RECONCILING"
  PROVISIONING = "PROVISIONING"
  STOPPING = "STOPPING"


NOT_FOUND_MSG = "Cluster not found in GKE API."
UNKNOW_LOCATION = "N/A"


# ================================================================
# Step 2: Helper Functions (Row builders / Logging / GSheet)
# ================================================================
def build_malfunction_row(
    issue_type: IssueType,
    proj: str,
    cluster_name: str,
    cluster_status: str,
    cluster_status_message: str,
    now_utc: str,
    cluster_location: str = UNKNOW_LOCATION,
    node_pools: List[Dict[str, Any]] | None = None,
) -> List[str]:
  if node_pools is None:
    node_pools = []
  return [
      issue_type.value,
      proj,
      cluster_name,
      cluster_location,
      cluster_status,
      cluster_status_message,
      json.dumps(node_pools),
      now_utc,
  ]


def insert_cluster_status_lists(
    credentials, status_list, project_name, cluster_name, location, now_utc
):
  """Insert both cluster + nodepool status into list"""
  try:
    #  Cannot query cluster info without location
    if not location:
      raise NotFound(
          "The requested resource can not be found on the server without location."
      )
    info = get_cluster_status(credentials, project_name, location, cluster_name)
    cluster_status = info["status"]
    cluster_status_message = info["status_message"]
    mal_node_pools = [
        node_pool
        for node_pool in info["node_pools"]
        if node_pool["status"]
        not in [
            Status.RUNNING.value,
            Status.PROVISIONING.value,
            Status.STOPPING.value,
        ]
    ]
    is_cluster_malfunction = (
        cluster_status != Status.RUNNING.value
        and cluster_status != Status.RECONCILING.value
    )
    malfunction_node_pools_exist = len(mal_node_pools) > 0

    if is_cluster_malfunction and malfunction_node_pools_exist:
      status_list.append(
          build_malfunction_row(
              issue_type=IssueType.CLUSTER_AND_NODE_POOL,
              proj=project_name,
              cluster_name=cluster_name,
              cluster_location=location,
              cluster_status=cluster_status,
              cluster_status_message=cluster_status_message,
              node_pools=mal_node_pools,
              now_utc=now_utc,
          )
      )
    elif is_cluster_malfunction:
      status_list.append(
          build_malfunction_row(
              issue_type=IssueType.CLUSTER,
              proj=project_name,
              cluster_location=location,
              cluster_name=cluster_name,
              cluster_status=cluster_status,
              cluster_status_message=cluster_status_message,
              now_utc=now_utc,
          )
      )
    elif malfunction_node_pools_exist:
      status_list.append(
          build_malfunction_row(
              issue_type=IssueType.NODE_POOL,
              proj=project_name,
              cluster_name=cluster_name,
              cluster_location=location,
              cluster_status=cluster_status,
              cluster_status_message=cluster_status_message,
              node_pools=mal_node_pools,
              now_utc=now_utc,
          )
      )
  except NotFound:
    status_list.append(
        build_malfunction_row(
            issue_type=IssueType.CLUSTER,
            proj=project_name,
            cluster_name=cluster_name,
            cluster_location=UNKNOW_LOCATION,
            cluster_status=Status.NOT_EXIST.value,
            cluster_status_message=NOT_FOUND_MSG,
            now_utc=now_utc,
        )
    )
  except Exception as e:
    logging.info(
        f"Error fetching details for {project_name}/{cluster_name}: {e}"
    )
    status_list.append(
        build_malfunction_row(
            issue_type=IssueType.CLUSTER,
            proj=project_name,
            cluster_name=cluster_name,
            cluster_location=UNKNOW_LOCATION,
            cluster_status=Status.ERROR.value,
            cluster_status_message=str(e),
            now_utc=now_utc,
        )
    )


def print_failed_cluster_info(cluster_status_rows):
  """Log failed cluster info for debugging"""
  result_rows_list = []
  for cluster_status_row in cluster_status_rows:
    result_row = {
        "type": cluster_status_row[0],
        "project_id": cluster_status_row[1],
        "cluster_name": cluster_status_row[2],
        "cluster_location": cluster_status_row[3],
        "status": cluster_status_row[4],
        "status_message": cluster_status_row[5],
        "node_pools": cluster_status_row[6],
        "load_time": cluster_status_row[7],
    }
    result_rows_list.append(result_row)
  logging.info(f"result: {result_rows_list}")


def insert_gspread_rows(rows: List[List[str]], sheet_id: str):
  """Insert rows into Google Sheet via Airflow GSheetsHook"""
  try:
    hook = GSheetsHook(gcp_conn_id="google_cloud_default")
    logging.info(
        f"Inserting {len(rows)} rows into Google Sheet ID: {sheet_id}..."
    )
    if len(sheet_id) == 0:
      raise Exception(
          f"Sheet ID is empty. Please insert Sheet ID or set 'buganizer_sheet_id' in Variables"
      )
    hook.append_values(
        spreadsheet_id=sheet_id,
        range_=GSPREAD_WORKSHEET_NAME,
        values=rows,
        insert_data_option="INSERT_ROWS",
        value_input_option="RAW",
    )
    logging.info(f"Successfully appended {len(rows)} rows to Google Sheet.")
  except Exception as e:
    logging.error(f"An error occurred while writing to Google Sheet: {e}")


# ================================================================
# Step 3: BigQuery Functions
# ================================================================
def get_clusters_from_view(credential) -> List[Any]:
  """Get cluster list from BigQuery view"""
  context = get_current_context()
  project_id = context["params"]["source_bq_project_id"] or DEFAULT_PROJECT_ID
  dataset_id = context["params"]["source_bq_dataset_id"] or DEFAULT_DATASET_ID
  client = bigquery.Client(project=project_id, credentials=credential)
  query = f"""
        SELECT project_name, cluster_name, region
        FROM `{project_id}.{dataset_id}.cluster_view`
    """
  rows = list(client.query(query).result())
  logging.info(f"Fetched {len(rows)} rows from view.")
  return rows


# ================================================================
# Step 4: GKE Functions
# ================================================================
def get_cluster_status(
    credentials, project_id, location, cluster_name
) -> Dict[str, Any]:
  """Get detailed GKE cluster + nodepool status"""
  client = container_v1.ClusterManagerClient(credentials=credentials)
  name = f"projects/{project_id}/locations/{location}/clusters/{cluster_name}"
  request = container_v1.GetClusterRequest(name=name)
  cluster = client.get_cluster(request=request)
  cluster_mode = "Autopilot" if cluster.autopilot.enabled else "Standard"

  node_pools_info = []
  if cluster_mode == "Standard":
    for np in cluster.node_pools:
      node_pools_info.append({
          "name": np.name,
          "status": container_v1.NodePool.Status(np.status).name
          if np.status
          else "UNKNOWN",
          "status_message": np.status_message or None,
          "version": np.version,
          "autoscaling_enabled": np.autoscaling.enabled
          if np.autoscaling
          else False,
          "initial_node_count": np.initial_node_count,
          "machine_type": np.config.machine_type if np.config else None,
          "disk_size_gb": np.config.disk_size_gb if np.config else None,
          "preemptible": np.config.preemptible if np.config else False,
      })

  return {
      "project_id": project_id,
      "location": location,
      "cluster_name": cluster_name,
      "status": container_v1.Cluster.Status(cluster.status).name
      if cluster.status
      else "UNKNOWN",
      "status_message": cluster.status_message or None,
      "node_pools": node_pools_info,
  }


# ================================================================
# Step 5: Workflow Functions
# ================================================================
def fetch_clusters(credential) -> List[str]:
  clusters = get_clusters_from_view(credential=credential)
  if not clusters:
    logging.info("No rows found in view.")
    return []
  return clusters


def fetch_clusters_status(credentials, clusters: List[Any]) -> List[Any]:
  """Fetch status for all clusters & node pools"""
  cluster_status_rows = []
  now_utc = datetime.now(timezone.utc).isoformat()
  for cluster in clusters:
    insert_cluster_status_lists(
        credentials,
        cluster_status_rows,
        cluster.project_name,
        cluster.cluster_name,
        cluster.region,
        now_utc,
    )
  print_failed_cluster_info(cluster_status_rows)
  return cluster_status_rows


# ================================================================
# Step 6: Airflow Tasks
# ================================================================
@task(owner=test_owner.SEVERUS_H)
def pull_clusters_status() -> List[Any]:
  credentials, _ = google.auth.default(
      scopes=["https://www.googleapis.com/auth/cloud-platform"]
  )
  target_clusters = fetch_clusters(credentials)
  return fetch_clusters_status(credentials, target_clusters)


@task(owner=test_owner.SEVERUS_H)
def insert_gsheet_rows_task(gspread_rows_to_insert):
  if GSPREAD_INSERT_ENABLED and gspread_rows_to_insert:
    context = get_current_context()
    google_sheet_id = (
        context["params"]["target_gsheet_id"] or DEFAULT_GSPREAD_SHEET_ID
    )
    insert_gspread_rows(gspread_rows_to_insert, google_sheet_id)


# ================================================================
# Step 7: DAG Definition
# ================================================================
params = {
    "source_bq_project_id": Param(
        type="string",
        title="Source BigQuery GCP Project ID",
        description="The Source Google Cloud Project ID where the big query belong to",
        default=DEFAULT_PROJECT_ID,
    ),
    "source_bq_dataset_id": Param(
        type="string",
        title="Source Big Query Dataset ID",
        description="The Source BigQuery dataset ID where the data would be queried",
        default=DEFAULT_DATASET_ID,
    ),
    "target_gsheet_id": Param(
        type="string",
        title="Target Google Sheet ID",
        description="The Target Google Sheet ID where the Buganizer issue information are stored",
        default=DEFAULT_GSPREAD_SHEET_ID,
    ),
}

with DAG(
    dag_id="xlml_to_buganizer",
    description="A DAG that periodically queries the status of GKE clusters to monitor their health, and writes the collected data as rows into the target Google Sheet.",
    start_date=datetime(2025, 9, 3),
    schedule=SCHEDULED_TIME,
    catchup=False,
    tags=[],
    default_args={"retries": 0},
    params=params,
) as dag:
  target_clusters_status = pull_clusters_status()
  insert_gsheet_rows_task(target_clusters_status)
