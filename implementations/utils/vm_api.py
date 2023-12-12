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

  def run(vm_name, zone, project, cmd):
    stdout = open('/tmp/flowvm.out', 'w+')
    stderr = open('/tmp/flowvm.err', 'w+')
    gpu_api._exec(
        [
            'gcloud',
            'compute',
            'ssh',
            vm_name,
            f'--project={project}',
            f'--zone={zone}',
            f'--command="{cmd}"',
            "-- -o ProxyCommand='corp-ssh-helper %h %p'",
        ],
        stdout,
        stderr,
    )
    stdout.close()
    stderr.close()

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
        vm_duration,
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
      self.vm_duration = vm_duration
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
          '--scopes=https://www.googleapis.com/auth/cloud-platform',
          '--on-host-maintenance=TERMINATE',
          '--instance-termination-action=DELETE',
          f'--max-run-duration={self.vm_duration}',
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
      vm_duration,
      ssh_keys,
  ):
    gpu_api.node = gpu_api.Node(
        instance_name,
        instance_zone,
        project,
        image_project,
        image_family,
        accelerator,
        vm_duration,
        ssh_keys,
    )
    gpu_api.node.build()
