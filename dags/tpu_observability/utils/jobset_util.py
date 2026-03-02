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

import enum
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
from airflow.sensors.base import PokeReturnValue
from google.cloud.monitoring_v3 import types
import kubernetes

from dags.tpu_observability.utils import subprocess_util as subprocess
from dags.tpu_observability.utils.gcp_util import query_time_series
from dags.tpu_observability.utils.gcp_util import list_time_series
from dags.tpu_observability.utils.node_pool_util import Info as node_pool_info
from dags.tpu_observability.utils.node_pool_util import NODE_POOL_SELECTOR_KEY
from dags.tpu_observability.utils.time_util import TimeUtil
from xlml.apis import gcs
from xlml.utils import composer
from xlml.utils import gke


@task
def generate_node_pool_selector(prefix: str) -> str:
  """Generates a unique node_pool_selector value.

  Args:
    prefix: An identifier for the workload type (e.g., "resize", "rollback").

  Returns:
    The selector value string (e.g., "rollback-20260212123456").
  """
  run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
  return f"{prefix}-{run_id}"


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

          idx = jax.process_index()
          global_devices = jax.devices()
          print(f"[Host {idx}] Got {len(global_devices)} global devices")
          mesh = Mesh(global_devices, ("x",))

          print(f"[Host {idx}] Allocating data...")
          print(f"[Host {idx}] Defining sharding...")
          size = 32768
          global_shape = (size, size)
          sharding = NamedSharding(
            mesh, jax.sharding.PartitionSpec("x", None)
          )

          print(f"[Host {idx}] Creating sharded data directly on devices...")

          def ones_callback(index):
            resolved_indices = [
              s.indices(global_shape[i]) for i, s in enumerate(index)
            ]
            local_shape = tuple(
                stop - start for start, stop, step in resolved_indices
            )

            return jnp.ones(local_shape, dtype=jnp.float32)

          x = jax.make_array_from_callback(
              global_shape, sharding, ones_callback)
          y = jax.make_array_from_callback(
              global_shape, sharding, ones_callback)

          print(f"[Host {idx}] Data on device")

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
        f"""
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
                      {NODE_POOL_SELECTOR_KEY}: $node_pool_selector
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
  node_pool_selector: str

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
    params["node_pool_selector"] = self.node_pool_selector or ""

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
  def k8s_delete_pod_command(
      kubeconfig: str, pod_name: str, namespace: str
  ) -> str:
    return " ".join([
        f"kubectl --kubeconfig={kubeconfig} delete pod {pod_name}",
        f"-n {namespace} --wait=false",
    ])

  class K8sGetPodsOutput(enum.Enum):
    DEFAULT = "json"
    POD_NAME = "jsonpath={.items[*].metadata.name}"

  @staticmethod
  def k8s_get_pods(
      jobset_name: str,
      namespace: str,
      output: K8sGetPodsOutput = K8sGetPodsOutput.DEFAULT,
  ) -> str:
    """Generates the kubectl command to get pods for a specific JobSet."""
    # -l filters by the official JobSet label to catch all pods/slices
    return (
        f"kubectl get pods -n {namespace} "
        f"-l jobset.sigs.k8s.io/jobset-name={jobset_name} "
        f"-o {output.value}"
    )

  @staticmethod
  def k8s_get_pod_name_command(
      jobset_name: str, namespace: str, output: K8sGetPodsOutput
  ) -> str:
    """Alias for getting just the names, maintaining existing API."""
    return Command.k8s_get_pods(jobset_name, namespace, output)


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
    node_pool: node_pool_info, jobset_name: str, namespace: str = "default"
) -> list[str]:
  """
  Get a list of pods for a specific JobSet which are in the "Running" state.

  Args:
    node_pool: The Info object containing the cluster information needed for
      the kubernetes API to connect to it.
    jobset_name: The name of the JobSet to filter pods by.
    namespace: The kubernetes namespace which is being searched for running
      pods.
  Returns:
    A list containing the names of all the pods in the "running" state as
      strings.
  """
  with tempfile.TemporaryDirectory() as tmpdir:
    env = os.environ.copy()
    env["KUBECONFIG"] = os.path.join(tmpdir, "kubeconfig")

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_get_pod_name_command(
            jobset_name, namespace, Command.K8sGetPodsOutput.DEFAULT
        ),
    ])

    stdout = subprocess.run_exec(cmd, env=env)
    data = json.loads(stdout)

    running_pods = [
        item["metadata"]["name"]
        for item in data.get("items", [])
        if item.get("status", {}).get("phase") == "Running"
    ]

    logging.info("Running pods for JobSet '%s': %s", jobset_name, running_pods)

  return running_pods


def _generate_jobset_name(dag_id_prefix: str) -> str:
  """
  Generates a jobset name.

  Args:
    dag_id_prefix: The DAG ID to use as a prefix for the jobset name.
    (should be shorter than 40 characters to fit k8s naming 63 characters limit)
  Returns:
    A string representing the generated jobset name.
  """
  now_utc = datetime.datetime.now(datetime.timezone.utc)
  timestamp = now_utc.strftime("%Y%m%d%H%M%S")
  dag_id_prefix = dag_id_prefix.replace("_", "-").lower()

  return f"{dag_id_prefix}-{timestamp}"


@task
def build_jobset_from_gcs_yaml(
    gcs_path: str,
    dag_name: str,
    **overrides,
) -> JobSet:
  """
  Builds a JobSet instance by merging YAML defaults and generating
  a timestamped name based on dag_id_prefix.

  Args:
    gcs_path: The GCS path to the YAML configuration file.
    dag_name: The name of the DAG to extract specific configurations.
    **overrides: Additional parameters to override default configurations.
  """
  config = gcs.load_yaml_from_gcs(gcs_path)
  known_fields = {f.name for f in dataclasses.fields(JobSet)}
  merged = {
      k: v
      for k, v in config.get("jobset_defaults", {}).items()
      if k in known_fields
  }
  dag_cfg = config.get("dag", {}).get(dag_name, {})
  dag_id_prefix = dag_cfg.get("dag_id_prefix")

  for k, v in dag_cfg.items():
    if k in known_fields and v is not None:
      merged[k] = v

  merged.update({k: v for k, v in overrides.items() if k in known_fields})
  merged["jobset_name"] = _generate_jobset_name(dag_id_prefix)

  logging.info(
      f"Final JobSet '{merged['jobset_name']}' created for DAG '{dag_name}'"
  )
  return JobSet(**merged)


@task
def run_workload(
    node_pool: node_pool_info, jobset_config: JobSet, workload_type: Workload
) -> TimeUtil:
  """
  Applies the specified YAML file to the GKE cluster.

  Args:
    node_pool: Configuration object with cluster details.
    jobset_config: The JobSet object containing YAML configuration.
    workload_type: The workload script to execute.
  Returns:
    The UTC time when the workload was started.
  """
  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name
    yaml_config = jobset_config.generate_yaml(workload_script=workload_type)

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_apply_jobset_command(
            temp_config_file.name, yaml_config, jobset_config.namespace
        ),
    ])

    subprocess.run_exec(cmd, env=env)

    # Log metadata for XLML dashboard
    # Pod names follow the pattern:
    #   {jobset_name}-{replicated_job_name}-{job-index}-{pod-index}-{random}
    # The jobset_name prefix is stable across pod recreations, so a regex
    # pattern is more reliable than an exact pod name list.
    pod_name_pattern = f"{jobset_config.jobset_name}.*"
    jobset_metadata = {
        "project_id": node_pool.project_id,
        "cluster_name": node_pool.cluster_name,
        "node_pool_name": node_pool.node_pool_name,
        "jobset_name": jobset_config.jobset_name,
        "pod_name_pattern": pod_name_pattern,
    }
    composer.log_metadata_for_xlml_dashboard(jobset_metadata)
    logging.info(
        "Logged JobSet metadata to XLML dashboard: %s", jobset_metadata
    )

    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    return TimeUtil.from_datetime(current_time_utc)


@task
def end_workload(node_pool: node_pool_info, jobset_config: JobSet):
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
            temp_config_file.name,
            jobset_config.jobset_name,
            jobset_config.namespace,
        ),
    ])

    subprocess.run_exec(cmd, env=env)


@task
def list_pod_names(
    node_pool: node_pool_info, jobset_config: JobSet
) -> list[str]:
  """
  Lists the names of all active pods in the specified namespace for a given
  JobSet.

  This task executes a series of shell commands to:
  1. Authenticate `gcloud` and generate a temporary kubeconfig for the cluster.
  2. Query `kubectl` to fetch pod names filtered by the provided namespace.

  Args:
    node_pool: Configuration object with cluster details.
    jobset_config: The JobSet object containing configuration details.

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
        Command.k8s_get_pod_name_command(
            jobset_config.jobset_name,
            jobset_config.namespace,
            Command.K8sGetPodsOutput.POD_NAME,
        ),
    ])

    stdout = subprocess.run_exec(cmd, env=env)

    if not stdout or not stdout.strip():
      logging.warning("Received empty pod list from bash task.")
      raise AirflowFailException("Received empty pod list from bash task.")

    pod_list = stdout.strip().split()
    return pod_list


@task
def delete_one_random_pod(
    node_pool: node_pool_info,
    jobset_config: JobSet,
):
  """
  Randomly selects and deletes one pod that is currently in the "running" state.

  This task is used for fault injection to test the self-healing and recovery
  capabilities of a JobSet. It first retrieves all running pods in the
  specified namespace and then triggers a deletion via kubectl.

  Args:
    node_pool: The Info object containing the cluster information needed for
      the kubernetes API to connect to it.
    namespace: The kubernetes namespace where the target pod resides.
      Defaults to "default".

  Raises:
    AirflowFailException: If no running pods are found in the specified
      namespace.
  """
  running_pods = get_running_pods(
      node_pool=node_pool,
      jobset_name=jobset_config.jobset_name,
      namespace=jobset_config.namespace,
  )
  if not running_pods:
    logging.error(
        "No running pods found in namespace: %s", jobset_config.namespace
    )
    raise AirflowFailException(
        f"No running pods found in namespace: {jobset_config.namespace}"
    )

  target_pod = random.choice(running_pods)
  logging.info("Targeting pod for deletion: %s", target_pod)

  with tempfile.NamedTemporaryFile() as temp_config_file:
    env = os.environ.copy()
    env["KUBECONFIG"] = temp_config_file.name

    cmd = " && ".join([
        Command.get_credentials_command(node_pool),
        Command.k8s_delete_pod_command(
            temp_config_file.name, target_pod, jobset_config.namespace
        ),
    ])

    subprocess.run_exec(cmd, env=env)
    logging.info("Successfully initiated deletion for pod: %s", target_pod)


@task.sensor(poke_interval=30, timeout=900, mode="poke")
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
  time_series_data = list_time_series(
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


@task.sensor(poke_interval=60, timeout=3600, mode="poke")
def wait_for_jobset_ttr_to_be_found(
    node_pool: node_pool_info,
    jobset_config: JobSet,
    start_time: TimeUtil = None,
) -> bool:
  """Polls the jobset time-to-recover metric.

  A sensor task which polls the jobset time_between_interruptions metric
  every 60 seconds for 60 minutes. 60 minutes is used here since this
  metric does have a long latency before appearing in monitoring, typically
  between 30-45 minutes. While it may be possible for this latency to be
  longer than 60 minutes, it would be exceedingly rare, and it would be
  impractical for the test to run longer.

  Args:
    node_pool (Info): An instance of the Info class containing GKE metadata.
    jobset_config: An instance of the JobSet class representing the jobset
      configuration.
    start_time (TimeUtil, optional): The UTC timestamp to start polling from.
    If not provided, defaults to 60 minutes before the current time.

  Returns:
    bool: True if the TTR metric is found in Cloud Monitoring, False otherwise.
  """
  now = datetime.datetime.now()
  query_start = (
      start_time
      if start_time
      else TimeUtil.from_datetime(now - datetime.timedelta(minutes=60))
  )

  time_series = list_time_series(
      project_id=node_pool.project_id,
      filter_str=(
          'metric.type="kubernetes.io/jobset/times_to_recover" '
          f'resource.labels.cluster_name="{node_pool.cluster_name}" '
          f'resource.labels.entity_name="{jobset_config.jobset_name}"'
      ),
      start_time=query_start,
      end_time=TimeUtil.from_datetime(now),
  )

  logging.info("Time series: %s", time_series)
  return len(time_series) > 0


@task.sensor(poke_interval=30, timeout=600, mode="poke")
def wait_for_all_pods_running(
    node_pool: node_pool_info, jobset_config: JobSet
) -> PokeReturnValue:
  """Waits for all pods to be running and returns the pod names.

  Args:
    node_pool: The Info object containing the cluster information.
    jobset_config: The JobSet configuration.

  Returns:
    PokeReturnValue with is_done=True and pod names when all pods are running,
    or is_done=False to continue polling.
  """
  running_pods = get_running_pods(
      node_pool=node_pool,
      jobset_name=jobset_config.jobset_name,
      namespace="default",
  )
  num_pods = jobset_config.replicas * jobset_config.parallelism
  if len(running_pods) == num_pods:
    logging.info(
        "All %d pods are running for JobSet '%s': %s",
        num_pods,
        jobset_config.jobset_name,
        running_pods,
    )
    return PokeReturnValue(is_done=True, xcom_value=running_pods)
  return PokeReturnValue(is_done=False)


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

  return list_time_series(
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
    jobset_config: JobSet,
    jobset_apply_time: TimeUtil,
):
  """Verify uptime data exists after jobset application."""
  start_time = jobset_apply_time.to_datetime()
  end_time = datetime.datetime.now(datetime.timezone.utc)
  data = query_uptime_metrics(
      node_pool, jobset_config.jobset_name, start_time, end_time
  )

  logging.info(f"Uptime data query result: {data}")
  if len(data) > 0:
    return True
  return False


@task.sensor(poke_interval=30, timeout=360, mode="reschedule")
def ensure_no_jobset_uptime_data(
    node_pool: node_pool_info,
    jobset_config: JobSet,
    jobset_clear_time: TimeUtil,
    wait_time_seconds: int,
):
  """Ensure no uptime data is recorded after jobset deletion."""
  start_time = jobset_clear_time.to_datetime()
  now = datetime.datetime.now(datetime.timezone.utc)
  data = query_uptime_metrics(
      node_pool, jobset_config.jobset_name, start_time, now
  )

  logging.info(f"Uptime data query result: {data}")
  if len(data) > 0:
    raise AirflowFailException(f"Data detected: {data}")

  if now - start_time >= datetime.timedelta(seconds=wait_time_seconds):
    logging.info("Stability period passed with no data detected.")
    return True
  return False
