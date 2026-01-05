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
import os
from absl import logging
import textwrap

from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from airflow.models.taskmixin import DAGNode
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.utils.task_group import TaskGroup

from dags import composer_env
from xlml.utils import gke
from xlml.utils import composer


_KPO_LABEL = {"axlearn-kpo-label": "axlearn_cli_kpo_worker"}

# pylint: disable=line-too-long
_KPO_NAMESPACE = "composer-user-workloads"
"""
**MUST** use this fixed namespace for Cloud Composer 2.
See: https://cloud.google.com/composer/docs/composer-2/use-kubernetes-pod-operator#composer-2-kpo-access-project-resources
"""
# pylint: enable=line-too-long

_COMPOSER_KUBECONFIG_PATH = "/home/airflow/composer_kube_config"


@task(
    retry_delay=dt.timedelta(seconds=15),
    retries=4,
)
def reset_kube_config() -> None:
  """Get credential for in-cluster to setup CLI AXLearn command."""

  cluster_name = os.environ["COMPOSER_GKE_NAME"]
  project_id = os.environ["GCP_PROJECT"]
  region = os.environ["COMPOSER_LOCATION"]

  logging.info(f"{' LOGGING AIRFLOW CLUSTER ':=^80}")
  logging.info("CLUSTER_NAME: %s", cluster_name)
  logging.info("PROJECT_ID: %s", project_id)
  logging.info("REGION: %s", region)

  hook = SubprocessHook()
  result = hook.run_command([
      "bash",
      "-c",
      f"sudo chown -R airflow:airflow {_COMPOSER_KUBECONFIG_PATH} && "
      f"gcloud container clusters get-credentials {cluster_name} "
      f"--region {region} --project {project_id}",
  ])

  assert (
      result.exit_code == 0
  ), f"XPK clean-up failed with code {result.exit_code}"


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


def start_cli_in_kpo(
    start_axlearn_cli_command: str,
    workload_id: str,
    task_owner: str,
    provisioning_timeout: dt.timedelta,
    workload_run_timeout: dt.timedelta,
    image_full_url: str,
) -> DAGNode:
  """
  Launch AXLearn CLI in an isolated Kubernetes Pod (via @task.kubernetes) and
  wait until the workload's Pods become Ready (or the wait times out).

  Airflow injects this function as a temporary Python script into the Pod and
  executes it there. Because only the Python standard library is guaranteed to
  be available at runtime (unless baked into the image), this function avoids
  third-party imports.

  Args:
    start_axlearn_cli_command: Full shell command that starts AXLearn (the CLI
      will submit the workload to GKE).
    workload_id: Workload/JobSet identifier used to discover Pods (e.g., via a
      label selector).
    task_owner: The owner of the task, used for Airflow metadata and
      pod labeling.
    provisioning_timeout: Timedelta object representing the time reserved for
      resource provisioning.
    workload_run_timeout: Timedelta object representing the maximum allowed
      execution time for the Airflow task.
    image_full_url: The full URL of the Docker image.

  Returns:
    None. The task completes when readiness is observed.

  Raises:
    AirflowFailException: If the AXLearn CLI command fails, or if Pods do not
      become Ready before the timeout.
  """

  airflow_task_timeout = workload_run_timeout.total_seconds()
  k8s_timeout = airflow_task_timeout - provisioning_timeout.total_seconds()

  with TaskGroup(group_id="start_cli_in_kpo") as group:
    # The default worker pod's kube_config may contain leftover
    # contexts/configs from other tasks (e.g., after `gcloud container clusters
    # get-credentials`) that point to a different GKE cluster.
    # Reset it so kubectl/gcloud—and the KPO we launch—target the Composer
    # cluster and use the correct Workload Identity–backed SA.
    reset_kube_config_task = reset_kube_config.override(owner=task_owner)()

    kpo = KubernetesPodOperator(
        task_id="run_axlearn-cli",
        name="axlearn-cli-kpo",
        namespace=_KPO_NAMESPACE,
        config_file=_COMPOSER_KUBECONFIG_PATH,
        image=image_full_url,
        cmds=["bash", "-cx", start_axlearn_cli_command],
        labels={
            **_KPO_LABEL,
            "workload_id": workload_id,
            "owner": task_owner,
        },
        active_deadline_seconds=int(k8s_timeout),
        execution_timeout=workload_run_timeout,
        retries=0,
    )

    _ = reset_kube_config_task >> kpo

  return group


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
