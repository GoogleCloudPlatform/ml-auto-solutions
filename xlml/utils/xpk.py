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

"""Utilities to run workloads with xpk
(https://github.com/AI-Hypercomputer/xpk)."""

import os
import re
import tempfile
import uuid

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags.common.vm_resource import GpuVersion
from xlml.apis import metric_config
from xlml.utils import composer

# NOTE: This version needs to be pinned to ensure compatibility when using
# xpk.py for workload creation.
MAIN_BRANCH = "v1.16.0"

# Duration = past 7 days
LOGGING_URL_FORMAT = (
    "https://pantheon.corp.google.com/logs/query;"
    + "query=resource.type%3D%22k8s_container%22%0A"
    + "resource.labels.project_id%3D%22{project}%22%0A"
    + "resource.labels.location%3D%22{region}%22%0A"
    + "resource.labels.cluster_name%3D%22{cluster}%22%0A"
    + "resource.labels.namespace_name%3D%22{namespace}%22%0A"
    + "labels.k8s-pod%2Fjobset_sigs_k8s_io%2F"
    + "jobset-name%3D%22{workload_id}%22%20severity%3E%3DDEFAULT;"
    + "storageScope=project;duration=P7D?e=13803378&"
    + "mods=allow_workbench_image_override&project={project}"
)


def get_xpk_setup_cmd(
    tmpdir,
    branch: str = MAIN_BRANCH,
    *,
    host: bool = True,
):
  clone_branch = (
      f"git clone --branch {branch} https://github.com/AI-Hypercomputer/xpk"
      f" {tmpdir}/xpk"
  )

  bash_setup = "set -xue"

  # Create venv, install uv in it, then use uv to install xpk
  if host:
    setup_xpk = (
        f"python3 -m venv {tmpdir}/xpk_venv && "
        f"source {tmpdir}/xpk_venv/bin/activate && "
        f"pip install uv && "
        f"uv pip install -e {tmpdir}/xpk"
    )
  # Running xpk setup commands in container
  else:
    # uv handles venv creation and package fetching independently of Debian's stripped Python pip
    setup_xpk = (
        f"uv venv {tmpdir}/xpk_venv && "
        f"source {tmpdir}/xpk_venv/bin/activate && "
        f"uv pip install -e {tmpdir}/xpk"
    )

  cmds = [
      bash_setup,
      clone_branch,
      setup_xpk,
  ]
  return cmds


def is_valid_gpu_version(accelerator_type: str):
  if accelerator_type in [member.value for member in GpuVersion]:
    return True
  return False


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate a valid workload ID."""

  short_id = str(uuid.uuid4())[:8]
  # Remove all non-alphanumeric characters, and truncate to ensure the result
  # is less than 40 characters.
  short_benchmark = re.sub(r"[^a-zA-Z0-9-]+", "", benchmark_id)[:32]
  return f"{short_benchmark}{short_id}"


@task
def run_workload(
    task_id: str,
    cluster_project: str,
    zone: str,
    cluster_name: str,
    benchmark_id: str,
    workload_id: str,
    gcs_path: str,
    docker_image: str,
    accelerator_type: str,
    run_cmds: str,
    num_slices: int = 1,
    use_vertex_tensorboard: bool = False,
    use_pathways: bool = False,
    # Directory for enabling emergency checkpointing
    ramdisk_directory: str = "",
    mtc_enabled: bool = False,  # It enables MTC phase-2 drivers
    xpk_branch: str = MAIN_BRANCH,
    max_restart: int = 0,
    # to avoid workload preemption by manual tests.
    priority: str = "high",
    namespace: str = "default",
):
  """Run workload through xpk tool."""

  # Log required info for XLML PLX Dashboard
  composer.log_metadata_for_xlml_dashboard({
      "cluster_project": cluster_project,
      "zone": zone,
      "cluster_name": cluster_name,
      "task_id": task_id,
      "workload_id": workload_id,
      "gcs_path": gcs_path,
      "benchmark_id": benchmark_id,
      "docker_image": docker_image,
      "accelerator_type": accelerator_type,
      "num_slices": num_slices,
  })

  with tempfile.TemporaryDirectory() as tmpdir:
    if accelerator_type in [
        GpuVersion.XPK_H100.value,
        GpuVersion.XPK_H100_MEGA.value,
    ]:
      multi_keyword = "num-nodes"
    else:
      multi_keyword = "num-slices"

    create_field = "create-pathways" if use_pathways else "create"
    type_field = "tpu-type" if use_pathways else "device-type"

    workload_create_cmd = (
        f"source {tmpdir}/xpk_venv/bin/activate && "
        f"xpk workload {create_field}"
        f" --cluster={cluster_name} --workload={workload_id}"
        f" --command='{run_cmds}' --{type_field}={accelerator_type}"
        f" --{multi_keyword}={num_slices} --docker-image={docker_image}"
        f" --project={cluster_project} --zone={zone}"
        f" --priority={priority}"
        f" --env {metric_config.SshEnvVars.GCS_OUTPUT.name}={gcs_path}"
    )

    if namespace and namespace != "default":
      workload_create_cmd += f" --namespace={namespace}"

    if ramdisk_directory and not use_pathways:
      workload_create_cmd += f" --ramdisk-directory={ramdisk_directory}"

    if mtc_enabled:
      workload_create_cmd += " --mtc-enabled"

    # For Orbax DAG add flag '--max-restars=50' it is need it to test
    # resiliency during Maxtext training with Emergency Checkpointer and
    # Multi-tier Checkpointing.
    if max_restart > 0:
      workload_create_cmd += f" --max-restarts={max_restart}"

    # If using a valid GPU and the XPK branch is set to "main"
    # then branch is switch to "v0.4.1".
    if is_valid_gpu_version(accelerator_type) and xpk_branch == MAIN_BRANCH:
      xpk_branch = "v0.4.1"
      # "v0.4.1" doesn't support the "--skip_validation" parameter, ignore.
    else:
      # Default parameter to skip validation procedure.
      workload_create_cmd += " --skip-validation"

    cmds = get_xpk_setup_cmd(tmpdir, xpk_branch)
    if accelerator_type == GpuVersion.XPK_H100_MEGA.value:
      workload_create_cmd += " --scheduler=gke.io/topology-aware-auto"
    if use_vertex_tensorboard:
      workload_create_cmd += " --use-vertex-tensorboard"
      vertex_ai_dependency = (
          "pip install -U google-cloud-aiplatform cloud-accelerator-diagnostics"
      )
      cmds.append(vertex_ai_dependency)
    cmds.append(workload_create_cmd)
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={**os.environ, "KUBECONFIG": os.path.join(tmpdir, "xpk.conf")},
    )
    assert (
        result.exit_code == 0
    ), f"XPK command failed with code {result.exit_code}"


@task(trigger_rule="all_done")
def clean_up_workload(
    workload_id: str,
    project_id: str,
    zone: str,
    cluster_name: str,
    xpk_branch: str = MAIN_BRANCH,
    namespace: str = "default",
) -> bool:
  """Delete workload."""
  with tempfile.TemporaryDirectory() as tmpdir:
    workload_delete_cmd = (
        f"source {tmpdir}/xpk_venv/bin/activate && "
        f"xpk workload delete"
        f" --cluster={cluster_name} --workload={workload_id}"
        f" --project={project_id} --zone={zone}"
    )
    if namespace and namespace != "default":
      workload_delete_cmd += f" --namespace={namespace}"

    cmds = get_xpk_setup_cmd(tmpdir, xpk_branch)
    cmds.append(workload_delete_cmd)
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
        env={**os.environ, "KUBECONFIG": os.path.join(tmpdir, "xpk.conf")},
    )
    assert (
        result.exit_code == 0
    ), f"XPK clean-up failed with code {result.exit_code}"
