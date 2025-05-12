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

"""Workload functions for AOTC inference reproducibility benchmarks."""

import os
import tempfile

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from google.cloud import storage
from dags.map_reproducibility.utils.common_utils import configure_project_and_cluster
from dags.map_reproducibility.utils.common_utils import install_helm_cmds
from dags.map_reproducibility.utils.common_utils import namespace_cmds
from dags.map_reproducibility.utils.common_utils import internal_wait_for_jobs_cmds
from dags.map_reproducibility.utils.common_utils import cleanup_cmds
from dags.map_reproducibility.utils.common_utils import git_cookie_authdaemon
from dags.map_reproducibility.utils.common_utils import clone_recipes_gob, clone_internal_recipes_gob
from dags.map_reproducibility.utils.common_utils import get_internal_pre_workload_cmds, get_internal_pre_workload_job_name
from dags.map_reproducibility.utils.common_utils import get_gpu_recipe_cmd
from dags.map_reproducibility.utils.common_utils import get_recipe_repo_path, get_internal_recipe_repo_path
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.common_utils import parse_internal_config_filename
from dags.map_reproducibility.utils.common_utils_inference import helm_apply_cmds_internal_run_inference, extract_autoregressive_write_to_jsonl, copy_inference_output_cmds, write_jsonl_to_bigquery, get_gcs_output_location, BUCKET_NAME
from dags.map_reproducibility.utils.common_utils import parse_internal_config_content
from dags.map_reproducibility.utils.constants import KUEUE_NAME


@task
def run_internal_aotc_inference_workload(
    relative_config_yaml_path,
    test_run=False,
    timeout=None,
    image_version=None,
):
  """Runs the AOTC inference workload benchmark.

  Args:
    relative_config_yaml_path: Path to the config YAML relative to the repo root
  """
  # Parse config from filename
  config_yaml_name = relative_config_yaml_path.rsplit("/", maxsplit=1)[
      -1
  ].replace(".yaml", "")
  config = parse_internal_config_filename(config_yaml_name)

  # Get derived configuration
  cluster, cluster_region = get_cluster(config.HYPERCOMPUTER)
  docker_image = image_version
  values_name = f"{config.HYPERCOMPUTER}_{config.FRAMEWORK}_values"

  with tempfile.TemporaryDirectory() as tmpdir:
    hook = SubprocessHook()

    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                git_cookie_authdaemon()
                + clone_recipes_gob()
                + (() if test_run else clone_internal_recipes_gob())
            ),
        ],
        cwd=tmpdir,
    )

    recipe_repo_root = get_recipe_repo_path(tmpdir)

    # Update paths now that we have the repo paths
    internal_recipe_repo_root = (
        "/home/airflow/gcs/dags/dags/map_reproducibility"
    )
    if not test_run:
      internal_recipe_repo_root = get_internal_recipe_repo_path(tmpdir)
    values_file_path = f"{internal_recipe_repo_root}/values/{values_name}.yaml"
    model_specific_values_file_path = (
        f"{internal_recipe_repo_root}/values/{config_yaml_name}_values.yaml"
    )
    if os.path.exists(model_specific_values_file_path):
      # Use model-specific values file
      values_file_path = model_specific_values_file_path
      print(
          f"Using model-specific values file: {model_specific_values_file_path}"
      )
    else:
      print(
          f"Model-specific values file not found, using general values file: {values_file_path}"
      )

    full_config_yaml_path = (
        f"{internal_recipe_repo_root}/{relative_config_yaml_path}"
    )
    print(f"values_file_path is {values_file_path}")
    print(f"full_config_yaml_path is {full_config_yaml_path}")

    # Parse the config content now that we have the file path
    config = parse_internal_config_content(full_config_yaml_path, config=config)
    job_name = get_internal_pre_workload_job_name(
        config.MODEL_ID, config.FRAMEWORK
    )

    container_timeout = int(timeout) - 4
    print(f"container timeout is {container_timeout}")
    result = hook.run_command(
        [
            "bash",
            "-c",
            ";".join(
                configure_project_and_cluster(cluster, cluster_region)
                + get_gpu_recipe_cmd(
                    config.HYPERCOMPUTER,
                    config.MODEL_ID,
                    config.FRAMEWORK,
                    recipe_repo_root,
                )
                + install_helm_cmds()
                + namespace_cmds()
                + get_internal_pre_workload_cmds(job_name=job_name)
                + helm_apply_cmds_internal_run_inference(
                    config.FRAMEWORK,
                    config.HYPERCOMPUTER,
                    full_config_yaml_path,
                    internal_recipe_repo_root,
                    values_file_path,
                    docker_image,
                    cluster_name=cluster,
                    kueue_name=KUEUE_NAME,
                    additional_cmds=f" --set workload.gpus={config.NUM_GPUS} ",
                    test_run=test_run,
                )
                + internal_wait_for_jobs_cmds(timeout=container_timeout)
                + copy_inference_output_cmds(tmpdir)
                + cleanup_cmds()
            ),
        ],
        cwd=tmpdir,
    )
    assert result.exit_code == 0, f"Command failed with code {result.exit_code}"

    output_location = os.path.join(tmpdir, "output.txt")
    jsonl_location = os.path.join(tmpdir, "output.jsonl")
    extract_autoregressive_write_to_jsonl(
        job_name, output_location, jsonl_location
    )
    write_jsonl_to_bigquery(jsonl_location)
    gcs_output_location = get_gcs_output_location()
    print(f"GCS output location is {gcs_output_location}")

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    destination_file_name = f"{job_name}_results.jsonl"
    destination_blob_name = os.path.join(
        gcs_output_location, destination_file_name
    )
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(jsonl_location)
    print(f"File {jsonl_location} uploaded to {gcs_output_location}.")
