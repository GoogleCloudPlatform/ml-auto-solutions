# Copyright 2026 Google LLC
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

"""Utility functions for Cluster Toolkit (gcluster) orchestration."""

from collections.abc import Iterable, Sequence
import os
import re
import shlex
import tempfile
import uuid

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags.common.vm_resource import GpuVersion
from xlml.utils import composer, gke

DEFAULT_GCLUSTER_VERSION = "v1.105.0"

LOGGING_URL_FORMAT = gke.LOGGING_URL_FORMAT


def _run_command(
    command: Sequence[str],
    env: dict[str, str],
) -> None:
  """Run a subprocess command and raise an error on non-zero exit code."""
  result = SubprocessHook().run_command(list(command), env=env)
  if result.exit_code != 0:
    raise RuntimeError(
        f"Command {' '.join(command)} failed with exit code {result.exit_code}"
    )


def _get_gcluster_paths(tmpdir: str, version: str) -> tuple[str, str, str, str]:
  """Return (bin_dir, bundle_path, gcluster_bin, bundle_url) for gcluster."""
  bin_dir = os.path.join(tmpdir, "bin")
  bundle_path = os.path.join(tmpdir, "gcluster_bundle.tgz")
  gcluster_bin = os.path.join(bin_dir, "gcluster")
  bundle_url = (
      "https://github.com/GoogleCloudPlatform/cluster-toolkit/releases/"
      f"download/{version}/gcluster_bundle_linux_amd64.tgz"
  )
  return bin_dir, bundle_path, gcluster_bin, bundle_url


def _download_gcluster_command(bundle_url: str, bundle_path: str) -> list[str]:
  """Construct curl command to download the gcluster bundle."""
  return ["curl", "-fsSL", bundle_url, "-o", bundle_path]


def _extract_gcluster_command(bundle_path: str, bin_dir: str) -> list[str]:
  """Construct tar command to extract the gcluster bundle."""
  return ["tar", "-xzf", bundle_path, "-C", bin_dir]


def _setup_gcluster(
    tmpdir: str,
    version: str,
    env: dict[str, str],
) -> str:
  """Download, unpack, and prepare the gcluster binary."""
  bin_dir, bundle_path, gcluster_bin, bundle_url = _get_gcluster_paths(
      tmpdir, version
  )

  os.makedirs(bin_dir, exist_ok=True)
  _run_command(
      _download_gcluster_command(bundle_url, bundle_path),
      env,
  )
  _run_command(
      _extract_gcluster_command(bundle_path, bin_dir),
      env,
  )
  if os.path.exists(gcluster_bin):
    os.chmod(gcluster_bin, 0o755)

  return gcluster_bin


def _get_credentials_command(
    cluster_name: str,
    location: str,
    project_id: str,
) -> list[str]:
  """Construct gcloud command to get GKE cluster credentials."""
  return [
      "gcloud",
      "container",
      "clusters",
      "get-credentials",
      cluster_name,
      f"--location={location}",
      f"--project={project_id}",
  ]


def _set_namespace_command(namespace: str) -> list[str]:
  """Construct kubectl command to set current context namespace."""
  return [
      "kubectl",
      "config",
      "set-context",
      "--current",
      f"--namespace={namespace}",
  ]


def get_gcluster_setup_cmd(
    tmpdir: str,
    cluster_name: str,
    location: str,
    project_id: str,
    version: str = DEFAULT_GCLUSTER_VERSION,
    namespace: str = "default",
) -> list[str]:
  """Construct shell commands to install gcluster and configure credentials."""
  bin_dir, bundle_path, gcluster_bin, bundle_url = _get_gcluster_paths(
      tmpdir, version
  )
  kubeconfig_path = os.path.join(tmpdir, "gcluster_kube.conf")
  return [
      "set -xue",
      f"mkdir -p {shlex.quote(bin_dir)}",
      shlex.join(_download_gcluster_command(bundle_url, bundle_path)),
      shlex.join(_extract_gcluster_command(bundle_path, bin_dir)),
      f"chmod 755 {shlex.quote(gcluster_bin)}",
      f"export PATH={shlex.quote(bin_dir)}:$PATH",
      f"export KUBECONFIG={shlex.quote(kubeconfig_path)}",
      "unset KUBERNETES_SERVICE_HOST KUBERNETES_SERVICE_PORT",
      shlex.join(
          _get_credentials_command(
              cluster_name=cluster_name,
              location=location,
              project_id=project_id,
          )
      ),
      shlex.join(_set_namespace_command(namespace)),
  ]


def is_valid_gpu_version(accelerator_type: str) -> bool:
  """Check whether the given accelerator type corresponds to a valid GPU."""
  return accelerator_type in [member.value for member in GpuVersion]


WORKLOAD_ID_PREFIX_MAX_LEN = 19


def get_workload_id_prefix(benchmark_id: str) -> str:
  """Return the sanitized, truncated prefix used in gcluster workload IDs.

  Workload IDs are `<prefix>-<5 hex chars>`, and pod names are derived from
  them, so any log filter on pod names must be built from this prefix rather
  than from the full benchmark ID.
  """
  clean_benchmark = (
      re.sub(r"[^a-z0-9]+", "-", benchmark_id.lower())
      .strip("-")[:WORKLOAD_ID_PREFIX_MAX_LEN]
      .rstrip("-")
  )
  return clean_benchmark or "job"


@task
def generate_workload_id(benchmark_id: str) -> str:
  """Generate unique workload ID conforming to RFC 1123 for Cluster Toolkit."""
  short_id = uuid.uuid4().hex[:5]
  return f"{get_workload_id_prefix(benchmark_id)}-{short_id}"


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
    pathways_gcs_location: str = "",
    ramdisk_directory: str = "",
    mtc_enabled: bool = False,
    gcluster_version: str = DEFAULT_GCLUSTER_VERSION,
    max_restart: int = 0,
    restart_on_exit_codes: Iterable[int] | None = None,
    priority: str = "high",
    namespace: str = "default",
    mounts: str | Iterable[str] | None = None,
    queue: str = "default",
):
  """Run workload through Cluster Toolkit (gcluster) tool."""
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
    env = {
        **os.environ,
        "KUBECONFIG": os.path.join(tmpdir, "gcluster_kube.conf"),
    }
    env.pop("KUBERNETES_SERVICE_HOST", None)
    env.pop("KUBERNETES_SERVICE_PORT", None)

    gcluster_bin = _setup_gcluster(tmpdir, gcluster_version, env)
    slice_keyword = (
        "--num-nodes"
        if is_valid_gpu_version(accelerator_type)
        else "--num-slices"
    )

    submit_cmd = [
        gcluster_bin,
        "job",
        "submit",
        f"--cluster={cluster_name}",
        f"--location={zone}",
        f"--project={cluster_project}",
        f"--name={workload_id}",
        f"--command={run_cmds}",
        f"--compute-type={accelerator_type}",
        f"--image={docker_image}",
        f"{slice_keyword}={num_slices}",
        f"--priority={priority}",
        f"--env=GCS_OUTPUT={gcs_path}",
        "--skip-prereqs",
    ]

    if queue:
      submit_cmd.append(f"--queue={queue}")
    if namespace:
      submit_cmd.append(f"--gke-namespace={namespace}")
    if use_pathways:
      submit_cmd.append("--pathways")
      location = pathways_gcs_location or f"{gcs_path}/pathways"
      submit_cmd.append(f"--pathways-gcs-location={location}")
    if use_vertex_tensorboard:
      submit_cmd.append("--use-vertex-tensorboard")
    if ramdisk_directory:
      submit_cmd.append(f"--gke-mtc-ramdisk-dir={ramdisk_directory}")
    if mtc_enabled:
      submit_cmd.append("--gke-mtc-enabled")
    if max_restart > 0:
      submit_cmd.append(f"--restarts={max_restart}")
    codes = [] if restart_on_exit_codes is None else list(restart_on_exit_codes)
    if codes:
      submit_cmd.append(f"--restart-on-exit-codes={','.join(map(str, codes))}")
    if is_valid_gpu_version(accelerator_type):
      submit_cmd.append("--gke-scheduler=gke.io/topology-aware-auto")

    if mounts:
      mount_list = [mounts] if isinstance(mounts, str) else list(mounts)
      for m in mount_list:
        submit_cmd.append(f"--mount={m}")

    _run_command(
        _get_credentials_command(
            cluster_name=cluster_name,
            location=zone,
            project_id=cluster_project,
        ),
        env,
    )
    _run_command(_set_namespace_command(namespace), env)
    _run_command(submit_cmd, env)


@task(trigger_rule="all_done")
def clean_up_workload(
    workload_id: str,
    project_id: str,
    zone: str,
    cluster_name: str,
    gcluster_version: str = DEFAULT_GCLUSTER_VERSION,
    namespace: str = "default",
) -> None:
  """Delete/cancel workload using Cluster Toolkit."""
  with tempfile.TemporaryDirectory() as tmpdir:
    env = {
        **os.environ,
        "KUBECONFIG": os.path.join(tmpdir, "gcluster_kube.conf"),
    }
    env.pop("KUBERNETES_SERVICE_HOST", None)
    env.pop("KUBERNETES_SERVICE_PORT", None)

    gcluster_bin = _setup_gcluster(tmpdir, gcluster_version, env)

    _run_command(
        _get_credentials_command(
            cluster_name=cluster_name,
            location=zone,
            project_id=project_id,
        ),
        env,
    )
    _run_command(_set_namespace_command(namespace), env)

    cancel_cmd = [
        gcluster_bin,
        "job",
        "cancel",
        workload_id,
        f"--cluster={cluster_name}",
        f"--location={zone}",
        f"--project={project_id}",
        "--skip-prereqs",
    ]
    if namespace:
      cancel_cmd.append(f"--gke-namespace={namespace}")

    _run_command(cancel_cmd, env)
