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
from __future__ import annotations

import subprocess
from typing import Optional, Sequence, TextIO
from absl import logging


class RunError(Exception):
  pass


class gpu_api:
  """Implement api for GPU VM similar to replace google.cloud.tpu_v2alpha1."""

  node = None

  def __new__(cls):
    raise TypeError('This is a static class and cannot be instantiated.')

  def _exec(
      args: Sequence[str], stdout: Optional[TextIO], stderr: Optional[TextIO]
  ) -> int:
    """Executes commands and streams stdout and stderr."""
    logging.info('Executing command: %s', ' '.join(args))
    print('Executing command: ', ' '.join(args))
    try:
      # Run in a shell since command prefixes may reference bash functions.
      # We interpret stdout=None or stderr=None to mean the user does not want to
      # capture that output.
      return subprocess.run(
          ' '.join(args),
          stdout=stdout or subprocess.DEVNULL,
          stderr=stderr or subprocess.DEVNULL,
          shell=True,
          check=True,
      ).returncode
    except subprocess.CalledProcessError as e:
      # Include the last 10 lines of stderr for better context.
      msg = str(e)
      if stderr and stderr.seekable() and stderr.readable():
        stderr.flush()
        stderr.seek(0)
        lines = stderr.readlines()[-10:]
        msg += f'.\n    Last {len(lines)} lines of standard error:\n      '
        msg += '      '.join(lines)
      raise RunError(msg) from e

  #   def run(vm_name, zone, project, cmd):
  #     stdout = open('/tmp/flowvm.out', 'w+')
  #     stderr = open('/tmp/flowvm.err', 'w+')
  #     gpu_api._exec(
  #         [
  #             'gcloud',
  #             'compute',
  #             'ssh',
  #             vm_name,
  #             f'--project={project}',
  #             f'--zone={zone}',
  #             f'--command="{cmd}"',
  #             "-- -o ProxyCommand='corp-ssh-helper %h %p'",
  #         ],
  #         stdout,
  #         stderr,
  #     )
  #     stdout.close()
  #     stderr.close()

  def _seek_last_line(stdout: TextIO) -> str:
    stdout.seek(0)
    try:
      last_line = stdout.readlines()[-1].strip()
    except IndexError as e:
      raise RunError(str(e)) from e
    return last_line

  class Node:

    def __init__(
        self,
        instance_name,
        instance_zone,
        project,
        image_project,
        image_family,
        accelerator,
        ssh_keys,
    ):
      self.stdout_file = '/tmp/airflow_gpurun.out'
      self.stderr_file = '/tmp/airflow_gpurun.err'
      self.image_project = image_project
      self.image_family = image_family
      self.instance_name = instance_name
      self.instance_zone = instance_zone
      self.project = project
      self.accelerator_type = accelerator.accelerator_type
      self.count = accelerator.count
      self.machine_type = accelerator.machine_type
      self.ssh_keys = ssh_keys
      stdout = open(self.stdout_file, 'w')
      stdout.close()
      stderr = open(self.stderr_file, 'w')
      stderr.close()

    def build(self):
      args = [
          f'gcloud alpha compute instances create {self.instance_name}',
          f'--zone={self.instance_zone}',
          f'--machine-type={self.machine_type}',
          f'--image-family={self.image_family}',
          f'--image-project={self.image_project}',
          f'--project={self.project}',
          f'--accelerator=type={self.accelerator_type},count={self.count}',
          '--boot-disk-size=500GB',
          f'--metadata="install-nvidia-driver=True,proxy-mode=project_editors,ssh-keys=cloud-ml-auto-solutions:{self.ssh_keys.public}"',
          #   '--scopes=https://www.googleapis.com/auth/cloud-platform',
          '--on-host-maintenance=TERMINATE',
          '--instance-termination-action=DELETE',
          f'--max-run-duration={self.vm_duration}m',
      ]
      try:
        stdout = open(self.stdout_file, 'a+')
        stderr = open(self.stderr_file, 'a+')
        gpu_api._exec(args, stdout, stderr)
        stdout.seek(0)
      finally:
        stdout.close()
        stderr.close()
        self.ip_address = self._get_ip_address()
        logging.info(f'self ip_address: {self.ip_address}')

    def _get_ip_address(self) -> str:
      prefix = 'networkIP:'
      args = [
          f'gcloud compute instances describe {self.instance_name}',
          f'--project={self.project}',
          f'--zone={self.instance_zone} | grep {prefix}',
      ]
      ip_address = '0.0.0.0'
      try:
        stdout = open(self.stdout_file, 'a+')
        stderr = open(self.stderr_file, 'a+')
        gpu_api._exec(args, stdout, stderr)
        last_line = gpu_api._seek_last_line(stdout)
        try:
          pos = last_line.find(prefix)
          if pos == -1:
            raise RunError(f'Unable to capture IP from {self.instance_name}.')
          logging.info(f'read last line: {last_line}')
          ip_address = last_line[pos + len(prefix) :].strip()
        except:
          logging.info(
              'Unable to parse the ip address. Last line of gpu instance'
              f' describe: {last_line}.'
          )
      except:
        logging.info(
            'Unable to open stdout_file, this may due to fail to create GPU'
            ' resource.'
        )
      finally:
        stdout.close()
        stderr.close()
        return ip_address

  def create_node_request(
      instance_name,
      instance_zone,
      project,
      image_project,
      image_family,
      accelerator,
      ssh_keys,
  ):
    gpu_api.node = gpu_api.Node(
        instance_name,
        instance_zone,
        project,
        image_project,
        image_family,
        accelerator,
        ssh_keys,
    )
    gpu_api.node.build()


# cloud API

import re
import sys
from typing import Any
import warnings

from google.api_core.extended_operation import ExtendedOperation
from google.cloud import compute_v1


def disk_from_image(
    disk_type: str,
    disk_size_gb: int,
    boot: bool,
    source_image: str,
    auto_delete: bool = True,
) -> compute_v1.AttachedDisk:
  """
  Create an AttachedDisk object to be used in VM instance creation. Uses an image as the
  source for the new disk.

  Args:
       disk_type: the type of disk you want to create. This value uses the following format:
          "zones/{zone}/diskTypes/(pd-standard|pd-ssd|pd-balanced|pd-extreme)".
          For example: "zones/us-west3-b/diskTypes/pd-ssd"
      disk_size_gb: size of the new disk in gigabytes
      boot: boolean flag indicating whether this disk should be used as a boot disk of an instance
      source_image: source image to use when creating this disk. You must have read access to this disk. This can be one
          of the publicly available images or an image from one of your projects.
          This value uses the following format: "projects/{project_name}/global/images/{image_name}"
      auto_delete: boolean flag indicating whether this disk should be deleted with the VM that uses it

  Returns:
      AttachedDisk object configured to be created using the specified image.
  """
  boot_disk = compute_v1.AttachedDisk()
  initialize_params = compute_v1.AttachedDiskInitializeParams()
  initialize_params.source_image = source_image
  initialize_params.disk_size_gb = disk_size_gb
  initialize_params.disk_type = disk_type
  boot_disk.initialize_params = initialize_params
  # Remember to set auto_delete to True if you want the disk to be deleted when you delete
  # your VM instance.
  boot_disk.auto_delete = auto_delete
  boot_disk.boot = boot


def create_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    disks: list[compute_v1.AttachedDisk],
    machine_type: str = 'n1-standard-1',
    network_link: str = 'global/networks/default',
    subnetwork_link: str = None,
    internal_ip: str = None,
    external_access: bool = False,
    external_ipv4: str = None,
    accelerators: list[compute_v1.AcceleratorConfig] = None,
    preemptible: bool = False,
    spot: bool = False,
    instance_termination_action: str = 'STOP',
    custom_hostname: str = None,
    delete_protection: bool = False,
) -> compute_v1.Instance:
  """
  Send an instance creation request to the Compute Engine API and wait for it to complete.

  Args:
      project_id: project ID or project number of the Cloud project you want to use.
      zone: name of the zone to create the instance in. For example: "us-west3-b"
      instance_name: name of the new virtual machine (VM) instance.
      disks: a list of compute_v1.AttachedDisk objects describing the disks
          you want to attach to your new instance.
      machine_type: machine type of the VM being created. This value uses the
          following format: "zones/{zone}/machineTypes/{type_name}".
          For example: "zones/europe-west3-c/machineTypes/f1-micro"
      network_link: name of the network you want the new instance to use.
          For example: "global/networks/default" represents the network
          named "default", which is created automatically for each project.
      subnetwork_link: name of the subnetwork you want the new instance to use.
          This value uses the following format:
          "regions/{region}/subnetworks/{subnetwork_name}"
      internal_ip: internal IP address you want to assign to the new instance.
          By default, a free address from the pool of available internal IP addresses of
          used subnet will be used.
      external_access: boolean flag indicating if the instance should have an external IPv4
          address assigned.
      external_ipv4: external IPv4 address to be assigned to this instance. If you specify
          an external IP address, it must live in the same region as the zone of the instance.
          This setting requires `external_access` to be set to True to work.
      accelerators: a list of AcceleratorConfig objects describing the accelerators that will
          be attached to the new instance.
      preemptible: boolean value indicating if the new instance should be preemptible
          or not. Preemptible VMs have been deprecated and you should now use Spot VMs.
      spot: boolean value indicating if the new instance should be a Spot VM or not.
      instance_termination_action: What action should be taken once a Spot VM is terminated.
          Possible values: "STOP", "DELETE"
      custom_hostname: Custom hostname of the new VM instance.
          Custom hostnames must conform to RFC 1035 requirements for valid hostnames.
      delete_protection: boolean value indicating if the new virtual machine should be
          protected against deletion or not.
  Returns:
      Instance object.
  """
  instance_client = compute_v1.InstancesClient()

  # Use the network interface provided in the network_link argument.
  network_interface = compute_v1.NetworkInterface()
  network_interface.network = network_link
  if subnetwork_link:
    network_interface.subnetwork = subnetwork_link

  if internal_ip:
    network_interface.network_i_p = internal_ip

  if external_access:
    access = compute_v1.AccessConfig()
    access.type_ = compute_v1.AccessConfig.Type.ONE_TO_ONE_NAT.name
    access.name = 'External NAT'
    access.network_tier = access.NetworkTier.PREMIUM.name
    if external_ipv4:
      access.nat_i_p = external_ipv4
    network_interface.access_configs = [access]

  # Collect information into the Instance object.
  instance = compute_v1.Instance()
  instance.network_interfaces = [network_interface]
  instance.name = instance_name
  instance.disks = disks
  if re.match(r'^zones/[a-z\d\-]+/machineTypes/[a-z\d\-]+$', machine_type):
    instance.machine_type = machine_type
  else:
    instance.machine_type = f'zones/{zone}/machineTypes/{machine_type}'

  instance.scheduling = compute_v1.Scheduling()
  if accelerators:
    instance.guest_accelerators = accelerators
    instance.scheduling.on_host_maintenance = (
        compute_v1.Scheduling.OnHostMaintenance.TERMINATE.name
    )

  if preemptible:
    # Set the preemptible setting
    warnings.warn('Preemptible VMs are being replaced by Spot VMs.', DeprecationWarning)
    instance.scheduling = compute_v1.Scheduling()
    instance.scheduling.preemptible = True

  if spot:
    # Set the Spot VM setting
    instance.scheduling.provisioning_model = (
        compute_v1.Scheduling.ProvisioningModel.SPOT.name
    )
    instance.scheduling.instance_termination_action = instance_termination_action

  if custom_hostname is not None:
    # Set the custom hostname for the instance
    instance.hostname = custom_hostname

  if delete_protection:
    # Set the delete protection bit
    instance.deletion_protection = True

  # Prepare the request to insert an instance.
  request = compute_v1.InsertInstanceRequest()
  request.zone = zone
  request.project = project_id
  request.instance_resource = instance

  # Wait for the create operation to complete.
  print(f'Creating the {instance_name} instance in {zone}...')

  operation = instance_client.insert(request=request)

  wait_for_extended_operation(operation, 'instance creation')

  print(f'Instance {instance_name} created.')
  return instance_client.get(project=project_id, zone=zone, instance=instance_name)
