"""Utilities for managing JobSets in GKE clusters for TPU observability."""

import datetime
import dataclasses
import logging
import os
import random
import subprocess
from typing import Final
import json
import string
import textwrap

from airflow.decorators import task
from airflow.exceptions import AirflowFailException
from google.cloud.monitoring_v3 import types

from dags.tpu_observability.utils import node_pool_util as node_pool
from dags.tpu_observability.utils.monitoring import query_time_series
from dags.tpu_observability.utils.time_util import TimeUtil


class Workload:
  """A library of predefined workload scripts for JobSet.

  Each workload is a JSON-escaped string, ready to be used as a shell argument.
  """

  JAX_TPU_BENCHMARK = json.dumps(
      textwrap.dedent(
          """
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
          print(f"[Host {jax.process_index()}] Got {len(global_devices)} global devices")
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
          for i in range(1_000_000): # Remember to control loop time to control experiment time
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


@dataclasses.dataclass
class JobSet:
  """Generates YAML configurations for Kubernetes JobSets.

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


def _get_credentials_command(node_pool: node_pool.Info) -> str:
  """Returns the command to authenticate `gcloud` with the specified GKE cluster.

  Args:
    node_pool: Configuration object with cluster details.

  Returns:
    A string containing the command to authenticate `gcloud` with the specified
    GKE cluster.
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


def _k8s_apply_jobset_command(
    kubeconfig: str, yaml_content: str, namespace: str
) -> str:
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} apply",
      f"-f - -n {namespace} <<EOF\n",
      f"{yaml_content}\nEOF",
  ])


def _k8s_delete_jobset_command(
    kubeconfig: str, jobset_name: str, namespace: str
) -> str:
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} delete jobsets {jobset_name}",
      f"-n {namespace} --timeout=60s --ignore-not-found=true",
  ])


def _k8s_get_pod_name_command(kubeconfig: str, namespace: str) -> str:
  return " ".join([
      f"kubectl --kubeconfig={kubeconfig} get pods",
      f"-n {namespace} -o jsonpath={{.items[*].metadata.name}}",
  ])


@task
def run_workload(
    node_pool: node_pool.Info, kubeconfig: str, yaml_config: str, namespace: str
) -> TimeUtil:
  """Applies the specified YAML file to the GKE cluster.

  Args:
    node_pool: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    yaml_config: The JobSet object containing YAML configuration.
    namespace: The Kubernetes namespace to apply the JobSet.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  cmd = " && ".join([
      _get_credentials_command(node_pool),
      _k8s_apply_jobset_command(kubeconfig, yaml_config, namespace),
  ])

  result = subprocess.run(
      cmd,
      shell=True,
      check=False,
      env=env,
      capture_output=True,
      text=True,
  )
  logging.info("Command Execute:\n %s", cmd)

  if result.returncode != 0:
    raise AirflowFailException(
        f"Command failed with exit code {result.returncode}.\n ,STDERR"
        f" message: {result.stderr}"
    )
  logging.info("STDOUT message: %s", result.stdout)

  current_time_utc = datetime.datetime.now(datetime.timezone.utc)
  return current_time_utc


@task
def end_workload(
    node_pool: node_pool.Info, kubeconfig: str, jobset_name: str, namespace: str
):
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    node_pool: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    jobset_name: The name of the JobSet to delete.
    namespace: The Kubernetes namespace to delete the JobSet from.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  cmd = " && ".join([
      _get_credentials_command(node_pool),
      _k8s_delete_jobset_command(kubeconfig, jobset_name, namespace),
  ])

  result = subprocess.run(
      cmd,
      shell=True,
      check=False,
      env=env,
      capture_output=True,
      text=True,
  )
  logging.info("Command Execute:\n %s", cmd)

  if result.returncode != 0:
    logging.info("Command failed with exit code %s.", result.returncode)
    logging.info("STDERR message: %s", result.stderr)
  logging.info("STDOUT message: %s", result.stdout)


@task
def get_active_pods(
    node_pool: node_pool.Info, kubeconfig: str, namespace: str
) -> list[str]:
  """Deletes all JobSets from the GKE cluster to clean up resources.

  This task executes a bash script to:
  1. Authenticate `gcloud` with the specified GKE cluster.
  2. Delete all JobSets in the `default` namespace using `kubectl`.

  Args:
    node_pool: Configuration object with cluster details.
    kubeconfig: The path to the kubeconfig file.
    namespace: The YamlConfig object containing namespace information.

  Returns:
    A list of pod names.
  """
  env = os.environ.copy()
  env["KUBECONFIG"] = kubeconfig

  cmd = " && ".join([
      _get_credentials_command(node_pool),
      _k8s_get_pod_name_command(kubeconfig, namespace),
  ])

  process = subprocess.run(
      cmd,
      shell=True,
      check=True,
      env=env,
      capture_output=True,
      text=True,
  )
  logging.info("Command Execute:\n %s", cmd)

  if not process or not process.stdout.strip():
    logging.warning("Received empty pod list from bash task.")
    raise AirflowFailException("Received empty pod list from bash task.")

  pod_list = process.stdout.strip().split()
  return pod_list


@task.sensor(poke_interval=30, timeout=900, mode="reschedule")
def wait_for_jobset_started(
    node_pool: node_pool.Info,
    pod_name_list: str,
    job_apply_time: datetime.datetime,
) -> bool:
  """Waits for the jobset to start by polling Cloud Logging for positive tensorcore utilization metrics.

  This task polls Cloud Logging for a specific log pattern that appears
  shortly after the TPU job begins execution within the specified container.
  It times out if no such log is found within a defined period.

  Args:
    node_pool: An Info dataclass instance containing project and cluster
      details.
    pod_name_list: A list of pod names.
    job_apply_time: The datetime object of the time the job was applied.
  """

  end_time_datatime = job_apply_time + datetime.timedelta(minutes=10)
  start_time = TimeUtil.from_datetime(job_apply_time)
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
