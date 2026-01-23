# Copyright 2025 Google LLC
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

"""Utilities for executing Python commands within the TPU Monitoring SDK."""

import os
import tempfile
import textwrap

from dags.tpu_observability.utils import jobset_util as jobset
from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils import subprocess_util as subprocess


class TpuMonitoringScript:
  """Predefined Python scripts for TPU monitoring SDK."""

  HELP = textwrap.dedent(
      """
      from libtpu.sdk import tpumonitoring
      tpumonitoring.help()
      """
  )

  LIST_SUPPORTED_METRICS = textwrap.dedent(
      """
      from libtpu.sdk import tpumonitoring
      print(tpumonitoring.list_supported_metrics())
      """
  )


def execute_sdk_command(
    info: node_pool.Info,
    pod_name: str,
    script: TpuMonitoringScript,
    namespace: str = "default",
) -> str:
  """Executes a predefined Python script inside a specific TPU pod via kubectl exec.

  Args:
    info: Node pool and cluster information.
    pod_name: The name of the target pod.
    script: The Python script to run (use TpuMonitoringScript options).
    namespace: Kubernetes namespace.

  Returns:
    The standard output of the executed command.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        jobset.Command.get_credentials_command(info),
        (
            f"kubectl exec {pod_name} -n {namespace} "
            f"-- python3 -c '{script}'"
        ),
    ])
    return subprocess.run_exec(cmd, env=env)
