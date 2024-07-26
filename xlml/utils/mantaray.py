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
    workload_file_name: str,
):
  gs_bucket = composer_env.get_gs_bucket()
  with tempfile.TemporaryDirectory() as tmpdir:
    cmds = (
        f"cd {tmpdir}",
        "sudo apt-get update && sudo apt-get install -y rsync",  # Install rsync
        "pip uninstall -y -q mantaray",  # Download and install mantaray
        (
            "gsutil cp"
            f" {gs_bucket}/mantaray/mantaray-0.1-py2.py3-none-any.whl ."
        ),
        ("gsutil cp -r" f" {gs_bucket}/mantaray/xlml_jobs ."),
        f"pip install mantaray-0.1-py2.py3-none-any.whl",
        f"python xlml_jobs/{workload_file_name}",  # Run the workload
    )
    hook = SubprocessHook()
    result = hook.run_command(
        ["bash", "-c", ";".join(cmds)],
    )
    assert (
        result.exit_code == 0
    ), f"Mantaray command failed with code {result.exit_code}"
