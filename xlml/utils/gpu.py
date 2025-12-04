# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to create, delete, and SSH with GPUs."""

from __future__ import annotations


from absl import logging
import airflow
from airflow.decorators import task, task_group
import datetime
import fabric
from google.cloud import compute_v1
import io
import paramiko
import re
import time
from typing import Dict, Iterable
import uuid
from xlml.apis import gcp_config, test_config
from xlml.utils import ssh, composer


def get_image_from_family(project: str, family: str) -> compute_v1.Image:
  """
  Retrieve the newest image that is part of a given family in a project.

  Args:
    project: project ID or project number of the Cloud project to get image.
    family: name of the image family you want to get image from.

  Returns:
    An Image object.
  """
  image_client = compute_v1.ImagesClient()
  # List of public operating system (OS) images:
  # https://cloud.google.com/compute/docs/images/os-details
  newest_image = image_client.get_from_family(project=project, family=family)
  return newest_image


def disk_from_image(
    disk_type: str,
    boot: bool,
    source_image: str,
    disk_size_gb: int = 100,
    auto_delete: bool = True,
) -> compute_v1.AttachedDisk:
  """
  Create an AttachedDisk object to be used in VM instance creation.
  Uses an image as the source for the new disk.

  Args:
    disk_type: the type of disk you want to create. This value uses the
        following format:
        "zones/{zone}/diskTypes/(pd-standard|pd-ssd|pd-balanced|pd-extreme)".
        For example: "zones/us-west3-b/diskTypes/pd-ssd"
    disk_size_gb: size of the new disk in gigabytes
    boot: boolean flag indicating whether this disk should be used as a boot
        disk of an instance
    source_image: source image to use when creating this disk. You must have
        read access to this disk. This can be one of the publicly available
        images or an image from one of your projects. This value uses the
        following format: "projects/{project_name}/global/images/{image_name}"
    auto_delete: boolean flag indicating whether this disk should be
        deleted with the VM that uses it

  Returns:
    AttachedDisk object configured to be created using the specified image.
  """
  boot_disk = compute_v1.AttachedDisk()
  initialize_params = compute_v1.AttachedDiskInitializeParams()
  initialize_params.source_image = source_image
  initialize_params.disk_size_gb = disk_size_gb
  initialize_params.disk_type = disk_type
  boot_disk.initialize_params = initialize_params
  # Remember to set auto_delete to True if you want the disk to be
  # deleted when you delete your VM instance.
  boot_disk.auto_delete = auto_delete
  boot_disk.boot = boot
  return boot_disk


def local_ssd_disk(zone: str) -> compute_v1.AttachedDisk:
  """
  Create an AttachedDisk object to be used in VM instance creation. The created disk contains
  no data and requires formatting before it can be used.

  Args:
      zone: The zone in which the local SSD drive will be attached.

  Returns:
      AttachedDisk object configured as a local SSD disk.
  """
  disk = compute_v1.AttachedDisk(interface="NVME")
  disk.type_ = compute_v1.AttachedDisk.Type.SCRATCH.name
  initialize_params = compute_v1.AttachedDiskInitializeParams()
  initialize_params.disk_type = f"zones/{zone}/diskTypes/local-ssd"
  disk.initialize_params = initialize_params
  disk.auto_delete = True
  return disk


def create_metadata(key_val: Dict[str, str]) -> compute_v1.Metadata:
  metadata = compute_v1.Metadata()
  metadata.items = [{"key": key, "value": val} for key, val in key_val.items()]
  return metadata


@task
def generate_gpu_name() -> str:
  # note: GPU vm name need to match regex
  # '(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?)', while TPU vm allows '_'. return
  # f'{base_gpu_name}-{str(uuid.uuid4())}'.replace('_', '-')
  # If we use the above base_gpu_name in the return, some potion of the can be
  # longer than 61 as in the regex.
  return f"gpu-{str(uuid.uuid4())}"


@task
def get_existing_resource(
    instance_name: str,
    ssh_keys: ssh.SshKeys,
    gcp: gcp_config.GCPConfig,
) -> airflow.XComArg:
  """Reach a resource node that is already created.

  Args:
    instance_name: name of the existing instance.
    ssh_keys: airflow.XComArg,
    gcp: GCP project/zone configuration.

  Returns:
    The ip address of the GPU VM.
  """
  instance_client = compute_v1.InstancesClient()
  instance_request = compute_v1.GetInstanceRequest(
      instance=instance_name,
      project=gcp.project_name,
      zone=gcp.zone,
  )
  instance = instance_client.get(request=instance_request)
  logging.info(
      f"Resource retrieve status: {instance.status}, {instance.status_message}"
  )

  ip_address = instance.network_interfaces[0].network_i_p
  metadata = instance.metadata
  items = metadata.items or []
  ssh_key_exist = False
  for item in metadata.items:
    if item.key == "ssh-keys":
      ssh_key_exist = True
      item.value = (
          item.value + "\n" + f"cloud-ml-auto-solutions:{ssh_keys.public}"
      )
      break
  if not ssh_key_exist:
    items.append({
        "key": "ssh-keys",
        "value": f"cloud-ml-auto-solutions:{ssh_keys.public}",
    })
    metadata.items = items
  metadata_request = compute_v1.SetMetadataInstanceRequest(
      instance=instance_name,
      project=gcp.project_name,
      zone=gcp.zone,
      metadata_resource=metadata,
  )
  operation = instance_client.set_metadata(request=metadata_request)
  if operation.error:
    logging.error(
        (
            "Error during instance set metadata: [Code:"
            f" {operation.http_error_status_code}]:"
            f" {operation.http_error_message}"
            f" {operation.error}"
        ),
    )
    raise operation.exception() or RuntimeError(operation.http_error_message)
  elif operation.warnings:
    logging.warning("Warnings during instance set metadata:\n")
    for warning in operation.warnings:
      logging.warning(f" - {warning.code}: {warning.message}")

  return ip_address


@task(trigger_rule="all_done")
def clean_up_ssh_keys(
    instance_name: str,
    ssh_keys: ssh.SshKeys,
    gcp: gcp_config.GCPConfig,
) -> airflow.XComArg:
  """Remove the generated one-time use ssh_keys from existing instance.

  Args:
    instance_name: name of the existing instance.
    ssh_keys: airflow.XComArg,
    gcp: GCP project/zone configuration.
  """
  instance_client = compute_v1.InstancesClient()
  instance_request = compute_v1.GetInstanceRequest(
      instance=instance_name,
      project=gcp.project_name,
      zone=gcp.zone,
  )
  instance = instance_client.get(request=instance_request)
  logging.info(
      f"Resource get status: {instance.status}, {instance.status_message}"
  )
  metadata = instance.metadata
  for item in metadata.items:
    if item.key == "ssh-keys":
      item.value = item.value.replace(
          f"\ncloud-ml-auto-solutions:{ssh_keys.public}", ""
      )
      break
  metadata_request = compute_v1.SetMetadataInstanceRequest(
      instance=instance_name,
      project=gcp.project_name,
      zone=gcp.zone,
      metadata_resource=metadata,
  )
  operation = instance_client.set_metadata(request=metadata_request)
  if operation.error:
    logging.error(
        (
            "Error during instance set metadata: [Code:"
            f" {operation.http_error_status_code}]:"
            f" {operation.http_error_message}"
            f" {operation.error}"
        ),
    )
    raise operation.exception() or RuntimeError(operation.http_error_message)
  elif operation.warnings:
    logging.warning("Warnings during instance set metadata:\n")
    for warning in operation.warnings:
      logging.warning(f" - {warning.code}: {warning.message}")


@task_group
def create_resource(
    gpu_name: airflow.XComArg,
    image_project: str,
    image_family: str,
    accelerator: test_config.Gpu,
    gcp: gcp_config.GCPConfig,
    ssh_keys: airflow.XComArg,
    timeout: datetime.timedelta,
    install_nvidia_drivers: bool = False,
    reservation: bool = False,
) -> airflow.XComArg:
  """Request a resource and wait until the nodes are created.

  Args:
    gpu_name: XCom value for unique GPU name.
    image_project: project of the image.
    image_family: family of the image.
    accelerator: Description of GPU to create.
    gcp: GCP project/zone configuration.
    ssh_kpeys: XCom value for SSH keys to communicate with these GPUs.
    timeout: Amount of time to wait for GPUs to be created.
    install_nvidia_drivers: Whether to install Nvidia drivers.
    reservation: Whether to use an existing reservation

  Returns:
    The ip address of the GPU VM.
  """
  project_id = gcp.project_name
  zone = gcp.zone

  @task
  def create_resource_request(
      instance_name: str,
      accelerator: test_config.Gpu,
      ssh_keys: ssh.SshKeys,
      instance_termination_action: str,
      external_access=True,
      spot: bool = False,
      delete_protection: bool = False,
      install_nvidia_drivers: bool = False,
      reservation: bool = False,
  ) -> airflow.XComArg:
    """
    Send an instance creation request to the Compute Engine API and wait for
    it to complete.

    Args:
        instance_name: name of the new virtual machine (VM) instance.
        accelerator: Description of GPU to create.
        ssh_keys: XCom value for SSH keys to communicate with these GPUs.
        instance_termination_action: What action should be taken once a Spot VM
            is terminated. Possible values: "STOP", "DELETE"
        external_access: boolean flag indicating if the instance should have an
            external IPv4 address assigned.
        spot: boolean value indicating if the new instance should be a Spot VM
            or not.
        delete_protection: boolean value indicating if the new virtual machine
            should be protected against deletion or not.
        install_nvidia_drivers: boolean value indicating whether to install
            Nvidia drivers.
        reservation: boolean value indicating whether to use VM reservation
    Returns:
        Ip address of the instance object created.
    """
    # Log required info for XLML PLX Dashboard
    composer.log_metadata_for_xlml_dashboard({
        "instance_name": instance_name,
        "cluster_project": gcp.project_name,
        "zone": gcp.zone,
        "dataset_name": gcp.dataset_name.value,
        "composer_project": gcp.composer_project,
        "dataset_project": gcp.dataset_project,
        "accelerator": {
            "type": accelerator.name,
            "num_cores": accelerator.count,
            "runtime_version": accelerator.runtime_version,
            "machine_type": accelerator.machine_type,
            "image_family": accelerator.image_family,
        },
        "accelerator_type": accelerator.machine_type,
    })

    machine_type = accelerator.machine_type
    image = get_image_from_family(project=image_project, family=image_family)
    disk_type = f"zones/{gcp.zone}/diskTypes/pd-ssd"
    disks = [
        disk_from_image(
            disk_type, True, image.self_link, accelerator.disk_size_gb
        )
    ]
    if accelerator.attach_local_ssd:
      for _ in range(accelerator.count):
        disks.append(local_ssd_disk(gcp.zone))
    metadata = create_metadata({
        "install-nvidia-driver": str(install_nvidia_drivers),
        "proxy-mode": "project_editors",
        "ssh-keys": f"cloud-ml-auto-solutions:{ssh_keys.public}",
    })

    accelerators = [
        compute_v1.AcceleratorConfig(
            accelerator_count=accelerator.count,
            accelerator_type=(
                f"projects/{gcp.project_name}/zones/{gcp.zone}/"
                f"acceleratorTypes/{accelerator.accelerator_type}"
            ),
        )
    ]
    service_account = compute_v1.ServiceAccount(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    instance_client = compute_v1.InstancesClient()
    # Use the network interface provided in the network_link argument.
    network_interface = compute_v1.NetworkInterface()
    if accelerator.subnetwork:
      network_interface.network = accelerator.network
    if accelerator.subnetwork:
      network_interface.subnetwork = accelerator.subnetwork

    if external_access:
      access = compute_v1.AccessConfig()
      access.type_ = compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name
      access.name = "External NAT"
      access.network_tier = access.NetworkTier.PREMIUM.name
      network_interface.access_configs = [access]

    # Collect information into the Instance object.
    instance = compute_v1.Instance()
    instance.network_interfaces = [network_interface]
    instance.name = instance_name
    instance.disks = disks
    if re.match(r"^zones/[a-z\d\-]+/machineTypes/[a-z\d\-]+$", machine_type):
      instance.machine_type = machine_type
    else:
      instance.machine_type = f"zones/{zone}/machineTypes/{machine_type}"

    instance.scheduling = compute_v1.Scheduling()
    if accelerators:
      instance.guest_accelerators = accelerators
      instance.scheduling.on_host_maintenance = (
          compute_v1.Scheduling.OnHostMaintenance.TERMINATE.name
      )

    if metadata:
      instance.metadata = metadata

    if service_account:
      instance.service_accounts = [service_account]

    if spot:
      # Set the Spot VM setting
      instance.scheduling.provisioning_model = (
          compute_v1.Scheduling.ProvisioningModel.SPOT.name
      )
      instance.scheduling.instance_termination_action = (
          instance_termination_action
      )

    if delete_protection:
      # Set the delete protection bit
      instance.deletion_protection = True

    if reservation:
      # Set reservation affinity if specified
      reservation_affinity = compute_v1.ReservationAffinity()
      reservation_affinity.consume_reservation_type = (
          compute_v1.ReservationAffinity.ConsumeReservationType.ANY_RESERVATION.name
      )
      instance.reservation_affinity = reservation_affinity

    # Prepare the request to insert an instance.
    request = compute_v1.InsertInstanceRequest()
    request.zone = zone
    request.project = project_id
    request.instance_resource = instance

    # Wait for the create operation to complete.
    logging.info(f"Creating the {instance_name} instance in {zone}...")

    operation = instance_client.insert(request=request)
    return operation.name

  @task.sensor(
      poke_interval=60, timeout=timeout.total_seconds(), mode="reschedule"
  )
  def wait_for_resource_creation(operation_name: airflow.XComArg):
    # Retrives the delete opeartion to check the status.
    client = compute_v1.ZoneOperationsClient()
    request = compute_v1.GetZoneOperationRequest(
        operation=operation_name,
        project=project_id,
        zone=zone,
    )
    operation = client.get(request=request)
    status = operation.status.name
    if status in ("RUNNING", "PENDING"):
      logging.info(
          f"Resource create status: {status}, {operation.status_message}"
      )
      return False
    else:
      if operation.error:
        logging.error(
            (
                "Error during resource creation: [Code:"
                f" {operation.http_error_status_code}]:"
                f" {operation.http_error_message}"
                f" {operation.error}"
            ),
        )
        raise operation.exception() or RuntimeError(
            operation.http_error_message
        )
      elif operation.warnings:
        logging.warning("Warnings during resource creation:\n")
        for warning in operation.warnings:
          logging.warning(f" - {warning.code}: {warning.message}")
      return True

  @task
  def get_ip_address(instance: str) -> airflow.XComArg:
    # It takes time to be able to use the ssh with the ip address
    # even though the creation request is complete. We intentionally
    # sleep for 60s to wait for the ip address to be accessible.
    time.sleep(60)
    instance_client = compute_v1.InstancesClient()
    instance = instance_client.get(
        project=project_id, zone=zone, instance=instance
    )
    if len(instance.network_interfaces) > 1:
      logging.warning(
          f"GPU instance {gpu_name} has more than one network interface."
      )
    return instance.network_interfaces[0].network_i_p

  operation = create_resource_request(
      instance_name=gpu_name,
      accelerator=accelerator,
      ssh_keys=ssh_keys,
      instance_termination_action="STOP",
      install_nvidia_drivers=install_nvidia_drivers,
      reservation=reservation,
  )
  ip_address = get_ip_address(gpu_name)
  wait_for_resource_creation(operation) >> ip_address
  return ip_address


@task
def ssh_host(
    ip_address: str,
    cmds: Iterable[str],
    ssh_keys: ssh.SshKeys,
    env: Dict[str, str] = None,
) -> None:
  """SSH GPU and run commands in multi process.

  Args:
   ip_address: The ip address of the vm resource.
   cmds: The commands to run on a GPU.
   ssh_keys: The SSH key pair to use for authentication.
   env: environment variables to be pass to the ssh runner session using dict.
  """
  pkey = paramiko.RSAKey.from_private_key(io.StringIO(ssh_keys.private))
  logging.info(f"Connecting to IP addresses {ip_address}")

  ssh_group = fabric.ThreadingGroup(
      ip_address,
      user="cloud-ml-auto-solutions",
      connect_kwargs={
          "auth_strategy": paramiko.auth_strategy.InMemoryPrivateKey(
              "cloud-ml-auto-solutions", pkey
          )
      },
  )
  ssh_group.run(cmds, env=env)


@task_group
def delete_resource(instance_name: airflow.XComArg, project_id: str, zone: str):
  @task(trigger_rule="all_done")
  def delete_resource_request(
      instance_name: str, project_id: str, zone: str
  ) -> airflow.XComArg:
    client = compute_v1.InstancesClient()
    request = compute_v1.DeleteInstanceRequest(
        instance=instance_name,
        project=project_id,
        zone=zone,
    )
    operation = client.delete(request=request)

    return operation.name

  @task.sensor(poke_interval=60, timeout=1800, mode="reschedule")
  def wait_for_resource_deletion(operation_name: airflow.XComArg):
    # Retrives the delete opeartion to check the status.
    client = compute_v1.ZoneOperationsClient()
    request = compute_v1.GetZoneOperationRequest(
        operation=operation_name,
        project=project_id,
        zone=zone,
    )
    operation = client.get(request=request)
    status = operation.status.name
    if status in ("RUNNING", "PENDING"):
      logging.info(
          f"Resource deletion status: {status}, {operation.status_message}"
      )
      return False
    else:
      if operation.error:
        logging.error(
            (
                "Error during resource deletion: [Code:"
                f" {operation.http_error_status_code}]:"
                f" {operation.http_error_message}"
            ),
        )
        logging.error(f"Operation ID: {operation.name}")
        raise operation.exception() or RuntimeError(
            operation.http_error_message
        )
      elif operation.warnings:
        logging.warning("Warnings during resource deletion:\n")
        for warning in operation.warnings:
          logging.warning(f" - {warning.code}: {warning.message}")
      return True

  op = delete_resource_request(instance_name, project_id, zone)
  wait_for_resource_deletion(op)
