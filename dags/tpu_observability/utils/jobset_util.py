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

"""Utilities for managing JobSets in GKE clusters for TPU observability."""

import dataclasses
import datetime
import json
import logging
import os
import random
import string
import tempfile
import textwrap
from typing import Final

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud.monitoring_v3 import types
import kubernetes

from dags.tpu_observability.utils.node_pool_util import Info as node_pool_info
from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.gcp_util import query_time_series
from dags.tpu_observability.utils.time_util import TimeUtil
from xlml.utils import gke


class Workload:
  """A library of predefined workload scripts for JobSet.

  Each workload is a JSON-escaped string, ready to be used as a shell argument.
  """

  JAX_TPU_BENCHMARK = json.dumps(
      textwrap.dedent(
          """
          pip install jax[k8s] libtpu
          python -c '
          import jax
          import jax.numpy as jnp
          import time
          import os
          from jax.sharding import Mesh, NamedSharding
          from jax.experimental.pjit import pjit

          os.environ.setdefault("JAX_USE_PJIT", "true")
          jax.distributed.initialize()

          global_devices = jax.devices()
          print(
              f"[Host {jax.process_index()}] "
              f"Got {len(global_devices)} global devices"
          )
          mesh = Mesh(global_devices, ("x",))

          print(f"[Host {jax.process_index()}] Allocating data...")
          size = 32832
          x_global = jnp.ones((size, size), dtype=jnp.float32)
          y_global = jnp.ones((size, size), dtype=jnp.float32)

          print(f"[Host {jax.process_index()}] Sharding data...")
          sharding = NamedSharding(mesh, jax.sharding.PartitionSpec("x", None))
          x = jax.device_put(x_global, sharding)
          y = jax.device_put(y_global, sharding)
          print(f"[Host {jax.process_index()}] Data on device")

          # ========= Define heavy workload =========
          @pjit
          def matmul_ultra_heavy(x, y):
              tmp1 = jnp.dot(x, y)
              tmp2 = jnp.dot(tmp1, y.T)
              tmp3 = jnp.dot(tmp2, x.T)
              tmp4 = jnp.dot(tmp3, x)
              tmp5 = jnp.dot(tmp4, y)
              return tmp5

          print(f"[Host {jax.process_index()}] Warming up...")
          matmul_ultra_heavy(x, y).block_until_ready()

          # ========= Benchmark =========
          print(f"[Host {jax.process_index()}] Starting benchmark...")

          start = time.time()
          # Remember to control loop time to control experiment time
          for i in range(1_000_000):
              result = matmul_ultra_heavy(x, y)
          result.block_until_ready()
          end = time.time()

          if jax.process_index() == 0:
              print(f"Total time: {end - start:.2f} seconds (on full v6e-16)")
          ' &&
          echo "Workload finished, sleeping now..." &&
          sleep 10000
          """
      ),
      ensure_ascii=False,
  )


# pylint: disable=line-too-long
_TEMPLATE = string.Template(
    textwrap.dedent(
        """
        apiVersion: jobset.x-k8s.io/v1alpha2
        kind: JobSet
        metadata:
          name: $jobset_name
          annotations:
            alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
          namespace: $namespace
        spec:
          failurePolicy:
            maxRestarts: $max_restarts
          replicatedJobs:
          - name: $replicated_job_name
            replicas: $replicas
            template:
              spec:
                backoffLimit: $backoff_limit
                completions: $completions
                parallelism: $parallelism
                template:
                  spec:
                    nodeSelector:
                      cloud.google.com/gke-tpu-accelerator: $tpu_accelerator_type
                      cloud.google.com/gke-tpu-topology: $tpu_topology
                    containers:
                    - name: $container_name
                      image: $image
                      command: $command
                      args:
                        - $args
                      stdin: true
                      tty: true
                      resources:
                        requests:
                          google.com/tpu: $tpu_cores_per_pod
                        limits:
                          google.com/tpu: $tpu_cores_per_pod
        """
    )
)
# pylint: enable=line-too-long


@dataclasses.dataclass
class JobSet:
  """
  Generates YAML configurations for Kubernetes JobSets.

  This class helps in creating JobSet YAMLs by providing a template and allowing
  customization of various parameters like jobset name, replicas, TPU
  configuration, and the workload script to be executed.

  Attributes:
    jobset_name: The name of the JobSet.
    namespace: The Kubernetes namespace for the JobSet.
    max_restarts: The maximum number of restarts for the JobSet.
    replicated_job_name: The name for the replicated Job within the JobSet.
    replicas: The number of replicas for the replicated Job.
    backoff_limit: The number of failed pods to tolerate before marking the
      Job as failed.
    completions: The number of pods that must complete successfully.
    parallelism: The number of pods to run in parallel.
    tpu_accelerator_type: The type of TPU accelerator (e.g.,
      "tpu-v6e-slice").
    tpu_topology: The TPU topology (e.g., "4x4").
    container_name: The name of the container in the pod.
    image: The container image to use.
    tpu_cores_per_pod: The number of TPU cores requested per pod.
  """

  jobset_name: str
  namespace: str
  max_restarts: int
  replicated_job_name: str
  replicas: int
  backoff_limit: int
  completions: int
  parallelism: int
  tpu_accelerator_type: str
  tpu_topology: str
  container_name: str
  image: str
  tpu_cores_per_pod: int

  def generate_yaml(self, workload_script: Workload) -> str:
    """Generates the final JobSet YAML content.

    Args:
        workload_script: A pre-formatted, JSON-escaped string from the Workload
          class.

    Returns:
        A string containing the complete JobSet YAML.
    """
    params = dataclasses.asdict(self)
    params["command"] = ["bash", "-c"]
    params["args"] = workload_script

    return _TEMPLATE.substitute(params)


class Command:
  """
  A collection of static methods to generate Kubernetes and gcloud commands.

  This class provides methods to construct shell commands for interacting with
  GKE clusters, including authentication, applying/deleting JobSets, and
  getting pod information.
  """

  @staticmethod
  def get_credentials_command(node_pool: node_pool_info) -> str:
    """
    Returns the command to authenticate `gcloud` with the specified GKE cluster.

    Args:
      node_pool: Configuration object with cluster details.

    Returns:
      A string containing the command to authenticate `gcloud` with the
        specified GKE cluster.
    """
    for attr_name in ["cluster_name", "region", "project_id"]:
      if not getattr(node_pool, attr_name):
        raise ValueError(f"{attr_name} must be set in the Info object.")

    return " ".join([
        "gcloud container clusters",
        f"get-credentials {node_pool.cluster_name}",
        f"--region={node_pool.region}",
        f"--project={node_pool.project_id}",
    ])

  @staticmethod
  def k8s_apply_jobset_command(
      kubeconfig: str, yaml_content: str, namespace: str
  ) -> str:
    return " ".join([
        f"kubectl --kubeconfig={kubeconfig} apply",
        f"-f - -n {namespace} <<EOF\n",
        f"{yaml_content}\nEOF",
    ])

  @staticmethod
  def k8s_delete_jobset_command(
      kubeconfig: str, jobset_name: str, namespace: str
  ) -> str:
    return " ".join([
        f"kubectl --kubeconfig={kubeconfig} delete jobsets {jobset_name}",
        f"-n {namespace} --timeout=60s --ignore-not-found=true",
    ])

  @staticmethod
  def k8s_get_pod_name_command(kubeconfig: str, namespace: str) -> str:
    return " ".join([
        f"kubectl --kubeconfig={kubeconfig} get pods",
        f"-n {namespace} -o jsonpath={{.items[*].metadata.name}}",
    ])


def get_replica_num(
    replica_type: str, job_name: str, node_pool: node_pool_info
) -> int:
  """
  Get the number of a certain type of replicas from a running jobset.

  This uses the Kubernetes API to connect to a desired cluster and returns
  the number of replicas in a certain status.

  Args:
    replica_type: The type of replica being searched for.
    job_name: The name of the job replica which is run from the jobset.
    node_pool: The Info object containing the cluster information needed for
    the kubernetes API to connect to it.
  Returns:
    The number of replicas of the specific type in the jobset.
  """
  api_client = gke.get_authenticated_client(
      node_pool.project_id,
      node_pool.region,
      node_pool.cluster_name,
  )

  api = kubernetes.client.CustomObjectsApi(api_client)

  jobsets = api.list_namespaced_custom_object(
      group="jobset.x-k8s.io",
      version="v1alpha2",
      namespace="default",
      plural="jobsets",
  )

  try:
    replica_job_status = jobsets["items"][0]["status"]["replicatedJobsStatus"]
    name = replica_job_status[0]["name"]
    replicas = replica_job_status[0][replica_type]
    logging.info("Found %s replicas", replicas)

  except (KeyError, IndexError, TypeError) as e:
    logging.error("Error in getting jobset satus: %s", e)
    return 0

  if name != job_name:
    raise AirflowFailException(
        f"Jobset found '{name}' does not match jobset name given '{job_name}'"
    )

  return replicas


def get_running_pods(
    node_pool: node_pool_info, namespace="default"
) -> list[str]:
  """
  Get a list of pods which are in the "running" state.

  Args:
    node_pool: The Info object containing the cluster information needed for
      the kubernetes API to connect to it.
    namespace: The kubernetes namespace which is being searched for running
      pods.
  Returns:
    A list containing the names of all the pods in the "running" state as
      strings.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    env = os.environ.copy()
    env["KUBECONFIG"] = tmpdir + "/kubeconfig"

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        f"kubectl get pods -n {namespace} -o json",
    ])

    stdout = subprocess.run_exec(cmd, env=env)

    data = json.loads(stdout)

    running_pods = [
        item["metadata"]["name"]
        for item in data.get("items", [])
        if item.get("status", {}).get("phase") == "Running"
    ]

    logging.info("Running pods: %s", running_pods)

  return running_pods


@task
def run_workload(
    node_pool: node_pool_info, yaml_config: str, namespace: str
) -> TimeUtil:
  """
  Applies the specified YAML file to the GKE cluster.

  Args:
    node_pool: Configuration object with cluster details.
    yaml_config: The JobSet object containing YAML configuration.
    namespace: The Kubernetes namespace to apply the JobSet.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_apply_jobset_command(
            temp_config_file.name, yaml_config, namespace
        ),
    ])

    subprocess.run_exec(cmd, env=env)

    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    return TimeUtil.from_datetime(current_time_utc)


@task
def end_workload(node_pool: node_pool_info, jobset_name: str, namespace: str):
  """
  Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    node_pool: Configuration object with cluster details.
    jobset_name: The name of the JobSet to delete.
    namespace: The Kubernetes namespace to delete the JobSet from.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_delete_jobset_command(
            temp_config_file.name, jobset_name, namespace
        ),
    ])

    subprocess.run_exec(cmd, env=env)


@task
def list_pod_names(node_pool: node_pool_info, namespace: str) -> list[str]:
  """
  Retrieves a list of active pod names from a specific GKE cluster namespace.

  This task executes a series of shell commands to:
  1. Authenticate `gcloud` and generate a temporary kubeconfig for the cluster.
  2. Query `kubectl` to fetch pod names filtered by the provided namespace.

  Args:
    node_pool: Configuration object with cluster details.
    namespace: The Kubernetes namespace to query for pods.

  Returns:
    A list of strings representing the names of the active pods.

  Raises:
    AirflowFailException: If the command returns an empty output or fails to
      retrieve any pod names.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_get_pod_name_command(temp_config_file.name, namespace),
    ])

    stdout = subprocess.run_exec(cmd, env=env)

    if not stdout or not stdout.strip():
      logging.warning("Received empty pod list from bash task.")
      raise AirflowFailException("Received empty pod list from bash task.")

    pod_list = stdout.strip().split()
    return pod_list


@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def wait_for_jobset_started(
    node_pool: node_pool_info,
    pod_name_list: str,
    job_apply_time: TimeUtil,
) -> bool:
  """
  Waits for the jobset to start by polling Cloud Logging for positive tensorcore
  utilization metrics.

  This task polls Cloud Logging for a specific log pattern that appears
  shortly after the TPU job begins execution within the specified container.
  It times out if no such log is found within a defined period.

  Args:
    node_pool: An Info dataclass instance containing project and cluster
      details.
    pod_name_list: A list of pod names.
    job_apply_time: The datetime object of the time the job was applied.
  """

  end_time_datatime = job_apply_time.to_datetime() + datetime.timedelta(
      minutes=10
  )
  start_time = job_apply_time
  end_time = TimeUtil.from_datetime(end_time_datatime)

  if not pod_name_list:
    raise AirflowFailException("pod_name_list is empty, sensor cannot proceed.")

  pod_name = random.choice(pod_name_list)
  metric_name = "kubernetes.io/container/accelerator/tensorcore_utilization"
  filter_string = [
      f'metric.type = "{metric_name}"',
      f'resource.labels.cluster_name = "{node_pool.cluster_name}"',
      f'resource.labels.pod_name = "{pod_name}"',
  ]
  time_series_data = query_time_series(
      project_id=node_pool.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
      view=types.ListTimeSeriesRequest.TimeSeriesView.FULL,
  )

  # The value of this metric means percentage of tensorcore utilization,
  # any positive values can represent that the jobset has started.
  threshold_value: Final[float] = 0.0

  # The minimum number of consecutive initial data points that must all exceed
  # 'threshold_value' to confirm that the jobset has successfully started and
  # is active.
  threshold_records_count: Final[int] = 3

  if (
      not time_series_data
      or len(time_series_data[0].points) < threshold_records_count
  ):
    return False
  last_n_data_points = [
      round(point.value.double_value, 2)
      for point in time_series_data[0].points[0:threshold_records_count]
  ]

  return all(p > threshold_value for p in last_n_data_points)


@task.sensor(poke_interval=60, timeout=3600, mode="reschedule")
def wait_for_jobset_ttr_to_be_found(node_pool: node_pool_info) -> bool:
  """
  Polls the jobset time_between_interruptions metric.

  A sensor task which polls the jobset time_between_interruptions metric
  every 60 seconds for 60 minutes. 60 minutes is used here since this
  metric does have a long latency before appearing in monitoring, typically
  between 30-45 minutes. While it may be possible for this latency to be
  longer than 60 minutes, it would be exceedingly rare, and it would be
  impractical for the test to run longer.

  Args:
    info(Info): An instance of the Info class that encapsulates
    the configuration and metadata of a GKE node pool and workload.
  """
  now = datetime.datetime.now()

  time_series = query_time_series(
      project_id=node_pool.project_id,
      filter_str=(
          'metric.type="kubernetes.io/jobset/times_to_recover" '
          f'resource.labels.cluster_name="{node_pool.cluster_name}" '
      ),
      start_time=TimeUtil.from_datetime(now - datetime.timedelta(minutes=60)),
      end_time=TimeUtil.from_datetime(now),
  )

  # This function checks whether the TTR metric is present;
  # it does not assess its value.
  logging.info("Time series: %s", time_series)
  return len(time_series) > 0


@task.sensor(poke_interval=30, timeout=600, mode="reschedule")
def wait_for_jobset_status_occurrence(
    replica_type: str, job_name: str, node_pool: node_pool_info
):
  """
  A sensor which checks if are any jobset replicas in a status type.

  Args:
    replica_type(str): The type of status being checked for.
    job_name(str): The name of the job replica which is run from the jobset.
    node_pool(Info): The Info object containing the cluster information needed
    for the kubernetes API to connect to it.
  """
  logging.info("Checking for number of replicas of type: %s", replica_type)
  ready_replicas = get_replica_num(
      replica_type=replica_type,
      job_name=job_name,
      node_pool=node_pool,
  )
  return ready_replicas > 0


@task.sensor(poke_interval=30, timeout=600, mode="reschedule")
def wait_for_all_pods_running(num_pods: int, node_pool: node_pool_info):
  num_running = len(get_running_pods(node_pool=node_pool, namespace="default"))
  return num_running == num_pods


def query_uptime_metrics(
    node_pool: node_pool_info,
    jobset_name: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
):
  """Queries the JobSet's uptime metric from Cloud Monitoring."""
  start_time = TimeUtil.from_datetime(start_time)
  end_time = TimeUtil.from_datetime(end_time)

  filter_string = [
      'metric.type="kubernetes.io/jobset/uptime"',
      f'resource.labels.project_id = "{node_pool.project_id}"',
      f'resource.labels.cluster_name = "{node_pool.cluster_name}"',
      f'resource.labels.entity_name = "{jobset_name}"',
  ]

  return query_time_series(
      project_id=node_pool.project_id,
      filter_str=" AND ".join(filter_string),
      start_time=start_time,
      end_time=end_time,
      view=types.ListTimeSeriesRequest.TimeSeriesView.FULL,
      log_enable=True,
  )


@task.sensor(poke_interval=30, timeout=3600, mode="reschedule")
def wait_for_jobset_uptime_data(
    node_pool: node_pool_info,
    jobset_name: str,
    jobset_apply_time: TimeUtil,
):
  """Verify uptime data exists after jobset application."""
  start_time = jobset_apply_time.to_datetime()
  end_time = datetime.datetime.now(datetime.timezone.utc)
  data = query_uptime_metrics(node_pool, jobset_name, start_time, end_time)

  logging.info(f"Uptime data query result: {data}")
  if len(data) > 0:
    return True
  return False


@task.sensor(poke_interval=30, timeout=360, mode="reschedule")
def ensure_no_jobset_uptime_data(
    node_pool: node_pool_info,
    jobset_name: str,
    jobset_clear_time: TimeUtil,
    wait_time_seconds: int,
):
  """Ensure no uptime data is recorded after jobset deletion."""
  start_time = jobset_clear_time.to_datetime()
  now = datetime.datetime.now(datetime.timezone.utc)
  data = query_uptime_metrics(node_pool, jobset_name, start_time, now)

  logging.info(f"Uptime data query result: {data}")
  if len(data) > 0:
    raise AirflowFailException(f"Data detected: {data}")

  if now - start_time >= datetime.timedelta(seconds=wait_time_seconds):
    logging.info("Stability period passed with no data detected.")
    return True
  return False


@task
def get_current_time() -> TimeUtil:
  """Get the current time in UTC."""
  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  return TimeUtil.from_datetime(current_time_utc)
