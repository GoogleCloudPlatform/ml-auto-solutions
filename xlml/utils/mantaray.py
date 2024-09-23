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

"""Utilities to run workloads with mantaray."""
import subprocess
import tempfile
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags import composer_env


MANTARAY_G3_GS_BUCKET = "gs://borgcron/cmcs-benchmark-automation/mantaray"


def load_file_from_gcs(gs_file_path):
  """Loads a file from a Google Cloud Storage bucket."""
  with tempfile.TemporaryDirectory() as tmpdir:
    subprocess.run(
        f"gsutil -m cp {gs_file_path} {tmpdir}/file",
        check=False,
        shell=True,
    )
    with open(f"{tmpdir}/file", "r") as f:
      return f.read()


@task
def run_workload(
    workload_file_name: str, mantaray_gcs_bucket: str = MANTARAY_G3_GS_BUCKET
):
  with tempfile.TemporaryDirectory() as tmpdir:
    cmds = (
        f"cd {tmpdir}",
        f"gsutil -m cp -r {mantaray_gcs_bucket} .",
        "sudo apt-get update && sudo apt-get install -y rsync",  # Install rsync
        f"cd mantaray && pip install -e .",
        f"python xlml_jobs/{workload_file_name}",  # Run the workload
    )
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
    )
    assert (
        result.exit_code == 0
    ), f"Mantaray command failed with code {result.exit_code}"


@task
def build_docker_image():
  with tempfile.TemporaryDirectory() as tmpdir:
    cmds = (
        f"cd {tmpdir}",
        f"gsutil -m cp -r {MANTARAY_G3_GS_BUCKET} .",
        "cd mantaray",
        "gcloud builds submit --config docker/cloudbuild.yaml --substitutions _DATE=$(date +%Y%m%d)",  # Create nightly docker image
    )
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
    )
    assert (
        result.exit_code == 0
    ), f"Mantaray command failed with code {result.exit_code}"
