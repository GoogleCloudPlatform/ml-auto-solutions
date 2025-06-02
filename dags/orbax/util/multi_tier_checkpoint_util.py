"""Utility functions for managing Multi-tier Cluster Configuration.

This module provides tasks for creating, applying, and deleting
Multi-tier Driver cluster Configurations for enebale Multi Tier Checkpointing.
"""

from absl import logging
import yaml
import time
from dataclasses import dataclass

from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException
from airflow.decorators import task
from airflow.exceptions import AirflowFailException

from xlml.utils import gke


# ramdisk_memory_in_mi: This should be Mi
class CheckpointConfiguration:
  """A dataclass to hold attributes of a Cloud Public Compute (CPC) instance."""

  def __init__(self,
    project_id: str,
    region: str,
    cluster_name: str,
    gcs_bucket: str,
    machine_type: str,
    ramdisk_memory: str,
    toleration_key: str = "google.com/tpu",
  ):
    """
    Initializes the CheckpointConfiguration.

    Args:
        project_id (str): The Google Cloud project ID.
        region (str): The Google Cloud region.
        cluster_name (str): The name of the GKE cluster.
        gcs_bucket (str): The name of the GCS bucket for checkpoints.
        machine_type (str): The machine type for the instance.
        ramdisk_memory_in_mi (str): The size of the RAM disk in mebibytes (Mi).
            Defaults to "2G".
        toleration_key (str): The toleration key for the Kubernetes pod.
            Defaults to "google.com/tpu".
    """
    self.project_id = project_id
    self.region = region
    self.cluster_name = cluster_name
    self.gcs_bucket = gcs_bucket
    self.machine_type = machine_type
    self.ramdisk_memory = ramdisk_memory
    self.toleration_key = toleration_key

  def _get_custom_objects_api_client(
      self, project_id: str, region: str, cluster_name: str
  ) -> k8s_client.CustomObjectsApi:
    """Create a CustomObjectsApi client for the given cluster."""
    return k8s_client.CustomObjectsApi(
      gke.get_authenticated_client(
        self.project_id,
        self.region,
        self.cluster_name
      )
    )

  # memory_size: This should be Mi
  def _create_cpc_content(
      self,
      gcs_bucket: str,
      machine_type: str,
      toleration_key: str,
      memory_size_in_mi: str,
  ) -> str:
    """Creates the CheckpointConfiguration YAML content as a string.

    This method generates a templated YAML string for a GKE CheckpointConfiguration
    resource, including parameters for the GCS bucket, machine type, toleration,
    and RAM disk size.

    Args:
        gcs_bucket (str): The name of the GCS bucket for storing checkpoints.
        machine_type (str): The instance type of the node.
        toleration_key (str): The toleration key to allow scheduling on specific nodes.
        memory_size_in_mi (str): The desired size of the in-memory volume.
          The unit is in mebibytes (Mi) but the value should be passed as a string
          with the unit, e.g., "2G" or "1024M".

    Returns:
        str: The templated YAML content as a string.
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
    inMemoryVolumeSize: {memory_size_in_mi}
  """
    logging.info(f"Generated CPC YAML content: \n{cpc_yaml_template}")
    return cpc_yaml_template


@task
def apply_cpc(cpc: CheckpointConfiguration) -> None:
  """Applies the CheckpointConfiguration to the cluster (create or replace)."""
  custom_api = cpc._get_custom_objects_api_client(
      cpc.project_id, cpc.region, cpc.cluster_name
  )

  cpc_yaml_string = cpc._create_cpc_content(
      cpc.gcs_bucket,
      cpc.machine_type,
      cpc.toleration_key,
      cpc.ramdisk_memory,
  )
  cpc_body = yaml.safe_load(cpc_yaml_string)

  api_version = cpc_body.get("apiVersion")
  kind = cpc_body.get("kind")
  name = cpc_body.get("metadata", {}).get("name")

  group, version = api_version.split("/", 1)
  plural = f"{kind.lower()}s"

  try:
    # Here we first try to create a reasource
    logging.info(f"Attempting to create CheckpointConfiguration '{name}'...")
    custom_api.create_cluster_custom_object(
        group=group, version=version, plural=plural, body=cpc_body
    )
    logging.info(f"CheckpointConfiguration '{name}' created successfully.")

  except ApiException as api_error:

    # If it already exists (409 Conflict), then try to replace it
    # See https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/409
    if api_error.status == 409:
      logging.info(
          f"CheckpointConfiguration '{name}' already exists. Attempting to replace..."
      )
      try:
        custom_api.replace_cluster_custom_object(
            group=group,
            version=version,
            plural=plural,
            name=name,
            body=cpc_body,
        )
        logging.info(f"CheckpointConfiguration '{name}' replaced successfully.")
      except ApiException as replace_error:
        logging.error(
            f"Error replacing CheckpointConfiguration:"
            f" {replace_error.status} - {replace_error.reason} - {replace_error.body}"
        )
        raise AirflowFailException(
            f"Failed to replace CheckpointConfiguration: {replace_error.reason}"
        )
    else:
      raise AirflowFailException(
          f"Failed to apply CheckpointConfiguration: {api_error.reason}"
      )


@task
def initiate_cpc_deletion(cpc: CheckpointConfiguration) -> None:
  """
  Sends the delete request for the CheckpointConfiguration.
  """
  custom_api = cpc._get_custom_objects_api_client(
      cpc.project_id, cpc.region, cpc.cluster_name
  )
  cpc_body = yaml.safe_load(
      cpc._create_cpc_content(
          cpc.gcs_bucket,
          cpc.machine_type,
          cpc.toleration_key,
          cpc.ramdisk_memory,
      )
  )
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
        f"Attempting to delete CheckpointConfiguration: {name_to_delete}"
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

  except ApiException as e:
    # The resource is already gone (404), so we can exit successfully
    # See https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/409
    if e.status == 404:
      logging.info(
          f"CheckpointConfiguration '{name_to_delete}' not found. Already deleted or never existed."
      )
      return
    else:
      raise AirflowFailException(
          f"Unexpected error during initiate_cpc_deletion: {e}"
      )


@task.sensor(poke_interval=10)
def wait_for_cpc_deletion(cpc: CheckpointConfiguration) -> bool:
  """
  A sensor that waits for the CheckpointConfiguration to be completely deleted.
  """
  custom_api = cpc._get_custom_objects_api_client(
      cpc.project_id, cpc.region, cpc.cluster_name
  )
  cpc_body = yaml.safe_load(
      cpc._create_cpc_content(
          cpc.gcs_bucket,
          cpc.machine_type,
          cpc.toleration_key,
          cpc.ramdisk_memory,
      )
  )
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

  try:
    custom_api.get_cluster_custom_object(
        group=group, version=version, plural=plural, name=name_to_delete
    )
    logging.info(
        f"CheckpointConfiguration '{name_to_delete}' still exists. "
        f"Polling again in 10s..."
    )
    return False
  except ApiException as e:
    # The resource is already gone (404), so we can exit successfully
    # See https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/409
    if e.status == 404:
      logging.info(
          f"CheckpointConfiguration: {name_to_delete} successfully deleted"
      )
      return True  # Return True to tell the sensor to succeed
    else:
      logging.error(
          f"API error while waiting for deletion: "
          f"{e.status} - {e.reason} - {e.body}"
      )
      raise AirflowFailException(
          f"API error during CPC deletion wait: {e.reason}"
      )
