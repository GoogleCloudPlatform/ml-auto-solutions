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

import io
from typing import Iterable, Tuple
import uuid

from absl import logging
import airflow
from airflow.decorators import task, task_group
from airflow.utils.task_group import TaskGroup
from apis import gcp_config, test_config
import fabric
import google.api_core.exceptions
import google.auth
import google.longrunning.operations_pb2 as operations
from implementations.utils import ssh
from implementations.utils.vm_api import gpu_api
import paramiko


@task
def generate_gpu_name(base_gpu_name: str) -> str:
  # note: GPU vm name need to match regex '(?:[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?)', while TPU vm allows '_'.
  # return f'{base_gpu_name}-{str(uuid.uuid4())}'.replace('_', '-')
  # If we use the above base_gpu_name in the return, some potion of the can be longer than 61 as in the regex.
  return f'gpu-{str(uuid.uuid4())}'


def create_resource(
    gpu_name: airflow.XComArg,
    image_project: str,
    image_family: str,
    accelerator,
    vm_duration,
    gcp: gcp_config.GCPConfig,
    ssh_keys: airflow.XComArg,
) -> Tuple[TaskGroup, airflow.XComArg]:
  """Request a resource and wait until the nodes are created.

  Args:
    gpu_name: XCom value for unique GPU name
    accelerator: Description of GPU to create.
    gcp: GCP project/zone configuration.
    ssh_keys: XCom value for SSH keys to communicate with these GPUs.
    timeout: Amount of time to wait for GPUs to be created.

  Returns:
    A TaskGroup for the entire create operation and an XCom value for the
    qualified queued_resource name.
  """

  @task
  def create_resource_request(gpu_name: str, ssh_keys: ssh.SshKeys) -> str:
    gpu_api.create_node_request(
        gpu_name,
        gcp.zone,
        gcp.project_name,
        image_project,
        image_family,
        accelerator,
        vm_duration,
        ssh_keys,
    )
    logging.info(
        'CREATE TPU RESOURCE: create_node_request. IP:' f' {gpu_api.node.ip_address}'
    )
    return gpu_api.node.ip_address

  with TaskGroup(group_id='create_queued_resource') as tg:
    qualified_name = create_resource_request(gpu_name, ssh_keys)
    # It takes time for the SSH key to be propagated to the instance.
    # We may fail to connect in the first trial.
  return tg, qualified_name


@task
def ssh_gpu(ip_address: str, cmds: Iterable[str], ssh_keys: ssh.SshKeys) -> None:
  """SSH TPU and run commands in multi process.

  Args:
   qualified_name: The qualified name of a queued resource.
   cmds: The commands to run on a TPU.
   ssh_keys: The SSH key pair to use for authentication.
  """
  pkey = paramiko.RSAKey.from_private_key(io.StringIO(ssh_keys.private))
  logging.info(f'Connecting to IP addresses {ip_address}')

  ssh_group = fabric.ThreadingGroup(
      ip_address,
      user='cloud-ml-auto-solutions',
      connect_kwargs={
          'auth_strategy': paramiko.auth_strategy.InMemoryPrivateKey(
              'cloud-ml-auto-solutions', pkey
          )
      },
  )
  ssh_group.run(cmds)
