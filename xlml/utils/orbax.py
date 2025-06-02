"""This module provides utility functions for managing Orbax checkpoints, including creating, applying, and deleting CheckpointConfigurations (CPCs)."""

from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException
from xlml.utils import gke
import yaml
import time

# --- Utility Functions ---


def _get_custom_objects_api_client(
    project_id: str, region: str, cluster_name: str
) -> k8s_client.CustomObjectsApi:
  """Create a CustomObjectsApi client for the given cluster."""
  client = gke.get_authenticated_client(project_id, region, cluster_name)
  custom_api = k8s_client.CustomObjectsApi(client)
  logging.info(
      "Successful initialize k8s CustomObjectsApi client from cluster response."
  )
  return custom_api


def create_cpc_content(
    gcs_bucket: str,
    machine_type: str,
    toleration_key: str,
    memory_size: str,
) -> str:
  """
  Creates the CheckpointConfiguration YAML content string with placeholders.
  Returns the templated YAML content as a string.
  """
  cpc_yaml_template = f"""
apiVersion: checkpointing.gke.io/v1
kind: CheckpointConfiguration
metadata:
  name: my-checkpointconfiguration # This name will be used for deletion
spec:
  cloudStorageBucketName: {gcs_bucket}
  nodeSelector:
    node.kubernetes.io/instance-type: {machine_type}
  tolerations:
  - key: {toleration_key}
    operator: Exists
    effect: NoSchedule
  inMemoryVolumeSize: {memory_size}
"""
  logging.info(f"Generated CPC YAML content: \n{cpc_yaml_template}")
  return cpc_yaml_template


# --- Airflow Tasks ---


@task
def apply_cpc(
    project_id: str,
    region: str,
    cluster_name: str,
    gcs_bucket: str,
    machine_type: str,
    toleration_key: str,
    memory_size: str,
) -> None:
  """Applies the CheckpointConfiguration to the cluster (create or replace)."""
  custom_api = _get_custom_objects_api_client(project_id, region, cluster_name)

  cpc_yaml_string = create_cpc_content(
      gcs_bucket,
      machine_type,
      toleration_key,
      memory_size,
  )
  cpc_body = yaml.safe_load(cpc_yaml_string)

  api_version = cpc_body.get("apiVersion")
  kind = cpc_body.get("kind")
  name = cpc_body.get("metadata", {}).get("name")

  group, version = api_version.split("/", 1)
  plural = f"{kind.lower()}s"

  try:
    logging.info(f"Checking if CheckpointConfiguration '{name}' exists...")
    custom_api.get_cluster_custom_object(
        group=group, version=version, plural=plural, name=name
    )
    logging.info(
        f"CheckpointConfiguration '{name}' found. Attempting to replace."
    )
    custom_api.replace_cluster_custom_object(
        group=group, version=version, plural=plural, name=name, body=cpc_body
    )
    logging.info(f"CheckpointConfiguration '{name}' replaced successfully.")
  except ApiException as api_error:
    if api_error.status == 404:
      logging.info(
          f"CheckpointConfiguration '{name}' not found. Attempting to create."
      )
      custom_api.create_cluster_custom_object(
          group=group, version=version, plural=plural, body=cpc_body
      )
      logging.info(f"CheckpointConfiguration '{name}' created successfully.")
    else:
      logging.error(
          "Error applying CheckpointConfiguration: "
          f"{api_error.status} - {api_error.reason} - {api_error.body}"
      )
      raise AirflowFailException(
          f"Failed to apply CheckpointConfiguration: {api_error.reason}"
      ) from api_error
  except Exception as e:
    logging.error(f"An unexpected error occurred during apply_cpc: {e}")
    raise AirflowFailException(f"Unexpected error during apply_cpc: {e}") from e


@task
def delete_cpc(
    project_id: str,
    region: str,
    cluster_name: str,
    gcs_bucket: str,
    machine_type: str,
    toleration_key: str,
    memory_size: str,
    poll_interval_seconds: int = 10,
) -> None:
  """
  Deletes the CheckpointConfiguration from the cluster and waits indefinitely
  for its complete deletion.
  """
  custom_api = _get_custom_objects_api_client(project_id, region, cluster_name)

  cpc_yaml_string = create_cpc_content(
      gcs_bucket,
      machine_type,
      toleration_key,
      memory_size,
  )
  cpc_body = yaml.safe_load(cpc_yaml_string)
  name_to_delete = cpc_body.get("metadata", {}).get("name")

  if not name_to_delete:
    logging.error(
        "Could not determine CheckpointConfiguration name for deletion."
    )
    raise AirflowFailException("Failed to determine CPC name for deletion.")

  api_version = cpc_body.get("apiVersion")
  kind = cpc_body.get("kind")
  group, version = api_version.split("/", 1)
  plural = f"{kind.lower()}s"

  delete_options = k8s_client.V1DeleteOptions(propagation_policy="Foreground")

  try:
    logging.info(
        f"Sending delete request for CheckpointConfiguration: {name_to_delete}"
    )
    custom_api.delete_cluster_custom_object(
        group=group,
        version=version,
        plural=plural,
        name=name_to_delete,
        body=delete_options,
    )
    logging.info(
        f"Delete request sent for CheckpointConfiguration '{name_to_delete}'."
    )

    # --- Wait indefinitely for deletion to complete ---
    logging.info(
        "Waiting indefinitely for CheckpointConfiguration"
        f" '{name_to_delete}' to be deleted."
    )
    while True:
      try:
        custom_api.get_cluster_custom_object(
            group=group, version=version, plural=plural, name=name_to_delete
        )
        logging.info(
            f"CheckpointConfiguration '{name_to_delete}' "
            f"still exists. Polling again in {poll_interval_seconds}s..."
        )
        time.sleep(poll_interval_seconds)
      except ApiException as e:
        if e.status == 404:
          logging.info(
              f"CheckpointConfiguration: {name_to_delete} successfully deleted"
          )
          return
        else:
          logging.error(
              f"API error while waiting for deletion: "
              f"{e.status} - {e.reason} - {e.body}"
          )
          raise AirflowFailException(
              f"API error during CPC deletion wait: {e.reason}"
          ) from e
      except Exception as e:
        logging.error(
            f"An unexpected error occurred while waiting for CPC deletion: {e}"
        )
        raise AirflowFailException(
            f"Unexpected error during CPC deletion wait: {e}"
        ) from e

  except ApiException as e:
    if e.status == 404:
      logging.info(
          f"CheckpointConfiguration '{name_to_delete}' not found. "
          "Already deleted or never existed. Skipping deletion."
      )
      return
    else:
      logging.error(
          "Error during initial delete request for CheckpointConfiguration: "
          f"{e.status} - {e.reason} - {e.body}"
      )
      raise AirflowFailException(
          f"Failed to delete CheckpointConfiguration: {e.reason}"
      ) from e
  except Exception as e:
    logging.error(f"An unexpected error occurred during delete_cpc: {e}")
    raise AirflowFailException(
        f"Unexpected error during delete_cpc: {e}"
    ) from e
