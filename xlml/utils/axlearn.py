# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0 #
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to run workloads with AXLearn."""

import datetime as dt
import textwrap

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook

from dags import composer_env
from xlml.utils import gke
from xlml.utils import composer


@task(
    retry_delay=dt.timedelta(seconds=15),
    retries=4,
)
def update_image_tag_cmd(image_name: str, workload_id: str):
  """
  AXLearn pulls this particular image {docker_image}:{workload_id} when
  creating the pod.

  Tag the image with {workload_id} before submitting the workload via the
  AXLearn CLI.
  """

  hook = SubprocessHook()
  result = hook.run_command([
      "bash",
      "-c",
      (
          "gcloud container images add-tag "
          f"{image_name} "
          f"{image_name}:{workload_id} "
          "--quiet"
      ),
  ])
  assert (
      result.exit_code == 0
  ), f"Failed to update image tag; exit code {result.exit_code}"


@task(retries=0)
def generate_workload_id() -> str:
  """Generates a unique run name for a AXLearn run."""

  run_time = dt.datetime.now().strftime("%Y%m%d%H%M")
  env = "prod" if composer_env.is_prod_env() else "dev"
  return f"automation-{env}-{run_time}"


@task(retries=0)
def generate_axlearn_cli_command(
    task_id: str,
    project_id: str,
    zone: str,
    cluster_name: str,
    workload_id: str,
    docker_image_name: str,
    docker_image_repo: str,
    docker_image_full_url: str,
    accelerator_type: str = "",
    module: str = "",
    model_config: str = "",
    trainer_dir: str = "",
    num_slices: int = 1,
    trace_steps: list[int] = None,
    label: str = "tpu-v5p",
) -> str:
  # Log required info for XLML PLX Dashboard
  composer.log_metadata_for_xlml_dashboard({
      "cluster_project": project_id,
      "zone": zone,
      "cluster_name": cluster_name,
      "task_id": task_id,
      "workload_id": workload_id,
      "docker_image": docker_image_full_url,
      "accelerator_type": accelerator_type,
      "num_slices": num_slices,
  })

  cfg_file = "~/.axlearn/axlearn.default.config"
  cfg_content = textwrap.dedent(
      f"""
      [gcp]
      _active = "{project_id}:{zone}"

      [gcp."{project_id}:{zone}"]
      project = "{project_id}"
      region = "{gke.zone_to_region(zone)}"
      zone = "{zone}"
      gke_cluster = "{cluster_name}"
      cluster = "{cluster_name}"
      labels = "{label}"
      docker_repo = "{docker_image_repo}"
      default_dockerfile = "Dockerfile"
      permanent_bucket = "axlearn-bucket-multipod"
      private_bucket = "axlearn-bucket-multipod"
      ttl_bucket = "axlearn-bucket-multipod"
      """
  ).strip()

  axlearn_cli = (
      f"axlearn gcp launch run --cluster={cluster_name} "
      f"--runner_name gke_tpu_single "
      f"--name={workload_id} "
      f"--instance_type={accelerator_type} "
      f"--max_tries=10 "
      f"--num_replicas={num_slices} "
      f"--bundler_spec=allow_dirty=True "
      f"--bundler_type=artifactregistry "
      f"--bundler_spec=image={docker_image_name} "
      f'-- "'
      f"ulimit -n 1048576; ulimit -c 0; "
      f"python3 -c 'import jax; jax.devices()'; "
      f"python3 -m axlearn.common.launch_trainer_main"
      f'" '
      f"--module={module} "
      f"--config={model_config} "
      f"--trainer_dir={trainer_dir}/{workload_id} "
      f"--data_dir=gs://axlearn-public/tensorflow_datasets "
      f"--mesh_selector={accelerator_type} "
      f"--jax_backend=tpu "
      f"--initialization_timeout=1200 "
  )
  if trace_steps:
    axlearn_cli += f"--trace_at_steps={','.join(map(str, trace_steps))}"

  # fmt: off
  return " && ".join([
      # Generating the configuration file for AXLearn workload.
      "mkdir -p ~/.axlearn",
      f"cat > {cfg_file} <<'EOF'\n{cfg_content}\nEOF\necho 'file created'",

      # Setups for the AXLearn CLI.
      "export PYTHONPATH=$PYTHONPATH:/root",
      "axlearn gcp config activate",
      "apt-get install -y kubectl google-cloud-sdk-gke-gcloud-auth-plugin",
      f"gcloud container clusters get-credentials {cluster_name} \
            --region {gke.zone_to_region(zone)} --project {project_id}",

      # Starting the AXLearn CLI.
      "export BASTION_TIER=disabled",
      f"export PROJECT_ID={project_id}",
      axlearn_cli,
  ])
  # fmt: on
