"""Utility functions for managing Multi-tier Cluster Configuration.

This module provides tasks for creating, applying, and deleting
Multi-tier Driver cluster Configurations for enable Multi Tier Checkpointing.
"""

from absl import logging
import yaml
from dataclasses import dataclass

from kubernetes import client as k8s_client
from kubernetes.client.rest import ApiException
from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from http import HTTPStatus

from xlml.utils import gke


@dataclass
class CheckpointConfiguration:
  """A dataclass to hold attributes of a Cloud Public Compute (CPC) instance."""

  project_id: str
  region: str
  cluster_name: str
  gcs_bucket: str
  machine_type: str
  ramdisk_memory: str
  toleration_key: str

  def __init__(
      self,
      project_id: str,
      region: str,
      cluster_name: str,
      gcs_bucket: str,
      machine_type: str,
      ramdisk_memory_in_mi: str,
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
      ram_disk_memory_in_mi (str): The size of the RAM disk in mebibytes (Mi).
        The unit is in mebibytes (Mi) but the value should be passed as a
        string with the unit, e.g., "1G" or "1024Mi".
      toleration_key (str): The toleration key for the Kubernetes pod.
        Defaults to "google.com/tpu".
    """
    self.project_id = project_id
    self.region = region
    self.cluster_name = cluster_name
    self.gcs_bucket = gcs_bucket
    self.machine_type = machine_type
    self.ramdisk_memory = ramdisk_memory_in_mi
    self.toleration_key = toleration_key

  def load_yaml_and_parse_body(
      self,
  ) -> tuple[str, str, str, str, dict[str, any]]:
    """
    Loads a YAML string template, populates it with class attributes, and parses the resulting body.

    This method constructs a CheckpointConfiguration YAML manifest as a string,
    using class attributes such as `self.gcs_bucket`, `self.machine_type`,
    `self.toleration_key`, and `self.ramdisk_memory` to fill in the
    placeholders. It then uses `yaml.safe_load` to convert this YAML string
    into a Python dictionary.

    Finally, it extracts key fields—group, version, plural, and name—from the
    loaded dictionary for use in API requests or other operations.

    Returns:
      tuple[str, str, str, str, dict[str, any]]: A tuple containing the
      extracted API group, API version, plural name, resource name, and the
      full parsed YAML body as a dictionary.
    """

    cpc_yaml_template = f"""
      apiVersion: checkpointing.gke.io/v1
      kind: CheckpointConfiguration
      metadata:
        name: my-checkpointconfiguration # This name will be used for deletion
      spec:
        cloudStorageBucketName: {self.gcs_bucket}
        nodeSelector:
          node.kubernetes.io/instance-type: {self.machine_type}
        tolerations:
        - key: {self.toleration_key}
          operator: Exists
          effect: NoSchedule
        inMemoryVolumeSize: {self.ramdisk_memory}
    """
    cpc_body = yaml.safe_load(cpc_yaml_template)
    group, version = cpc_body.get("apiVersion").split("/", 1)
    plural = cpc_body.get("kind").lower() + "s"
    name = cpc_body.get("metadata", {}).get("name")

    return group, version, plural, name, cpc_body

  def create_custom_objects_api_client(self) -> k8s_client.CustomObjectsApi:
    """Create a CustomObjectsApi client for the given cluster."""
    return k8s_client.CustomObjectsApi(
        gke.get_authenticated_client(
            self.project_id, self.region, self.cluster_name
        )
    )


@task
def apply_cpc(cpc: CheckpointConfiguration) -> None:
  """Applies the CheckpointConfiguration to the cluster (create or replace)."""

  custom_api = cpc.create_custom_objects_api_client()
  group, version, plural, name, cpc_body = cpc.load_yaml_and_parse_body()

  logging.info(f"Attempting to create CheckpointConfiguration '{name}'...")
  custom_api.create_cluster_custom_object(
      group=group, version=version, plural=plural, body=cpc_body
  )


def _delete_cpc(cpc: CheckpointConfiguration) -> bool:
  """
  Sends the delete request for the CheckpointConfiguration.
  Returns if the deletion was success in boolean.
  """

  custom_api = cpc.create_custom_objects_api_client()
  group, version, plural, name, _ = cpc.load_yaml_and_parse_body()

  if not name:
    logging.error(
        "Could not determine CheckpointConfiguration name for deletion."
    )
    raise AirflowFailException("Failed to determine CPC name for deletion.")

  delete_options = k8s_client.V1DeleteOptions(propagation_policy="Foreground")

  try:
    logging.info(f"Attempting to delete CheckpointConfiguration: {name}")
    custom_api.delete_cluster_custom_object(
        group=group,
        version=version,
        plural=plural,
        name=name,
        body=delete_options,
    )
    logging.info(f"Delete request sent for CheckpointConfiguration '{name}'.")
  except Exception as e:
    logging.info(
        f"An warning has occurred while deleting CPC. Please take a look: {e}"
    )
    pass

  try:
    custom_api.get_cluster_custom_object(
        group=group, version=version, plural=plural, name=name
    )
  except ApiException as e:
    # A `CheckpointConfiguration not found` error indicates that the deletion was successful.
    return e.status == HTTPStatus.NOT_FOUND


@task.sensor(poke_interval=10)
def wait_for_cpc_deletion(cpc: CheckpointConfiguration) -> bool:
  """
  A sensor that waits for the CheckpointConfiguration to be completely deleted.
  """
  return _delete_cpc(cpc)
