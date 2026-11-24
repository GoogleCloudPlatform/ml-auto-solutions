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

"""Records the Pathways images that MaxText E2E post-training workloads run.

gcluster starts the Pathways containers of a workload from the `latest` tags of
the Pathways server and proxy server images, so two E2E runs can use different
Pathways versions without any change in MaxText. To make each run traceable:

1. `record_pathways_images` runs as soon as a post-training workload starts,
   while its pods exist, and reads the image digests that its Pathways
   containers started from.
2. `write_pathways_images` writes the digests of all the workloads of the run
   to one JSON file in GCS, with the ID of the post-training DAG run.
3. The E2E DAG reads the server digests back with
   `read_pathways_server_digests` and sends them to GitHub, where the
   post-training image gets a tag for each of them. It only reads a file with
   the ID of the post-training DAG run that it triggered: a GitHub re-run keeps
   the run name, so the file can be from an earlier attempt.

Usually all the workloads of a run use one server image. If `server:latest`
moves while the workloads start, which takes hours, there are more digests.

Limitations: the digests are read once per workload, when it starts. If its
JobSet restarts later (`max_restart`), the new pods may pull a newer image,
which isn't recorded. If a worker pod hasn't started after
`_READ_ATTEMPTS` reads, the record has only the head pod's containers.

None of these tasks fails the DAG run: a problem only skips them.
"""

import json
import re
import time
from typing import Any, Mapping, Sequence

from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.models.baseoperator import BaseOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
import kubernetes

from xlml.utils import gke

# The Pathways containers of a gcluster workload. The resource manager and the
# proxy run as sidecars (init containers) of the head pod, and every worker pod
# runs `pathways-worker`. The resource manager and the workers run the server
# image.
PATHWAYS_CONTAINERS = ("pathways-rm", "pathways-proxy", "pathways-worker")
SERVER_CONTAINERS = ("pathways-rm", "pathways-worker")

# One file per E2E run, under the run name that all the stages of the run share.
SUMMARY_BUCKET = "runner-maxtext-logs"
SUMMARY_OBJECT = "e2e_runs/{run_name}/pathways_images.json"

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")

# Reads the pods again while a Pathways container hasn't reported its image.
# `wait_for_workload_start` only waits for the head pod of a Pathways workload,
# and the worker pods can start minutes later, when the cluster has to add TPU
# nodes for them first.
_READ_ATTEMPTS = 10
_SECONDS_BETWEEN_READS = 60


def workload_start_of(gke_task_group: TaskGroup) -> BaseOperator:
  """Returns the task that waits for the workload of a `run()` group to start.

  `XpkTask` and `GclusterTask` wait in
  `<group>.run_model.launch_workload.wait_for_workload_start`.

  Args:
    gke_task_group: The task group that `BaseRunnerTask.run()` returned.

  Returns:
    The `wait_for_workload_start` task of the group.
  """
  run_model = gke_task_group.get_child_by_label("run_model")
  launch_workload = run_model.get_child_by_label("launch_workload")
  return launch_workload.get_child_by_label("wait_for_workload_start")


def images_of_pods(
    pods: Sequence[kubernetes.client.V1Pod],
) -> dict[str, dict[str, Any]]:
  """Returns the image and digests of each Pathways container in `pods`.

  Args:
    pods: The pods of a workload.

  Returns:
    For each Pathways container that has started, e.g. "pathways-rm", its image
    as the pod spec names it, e.g. ".../pathways/server:latest", and the sorted
    digests of the images that it started from. A container has more than one
    digest only if its pods pulled the image at different times and the tag
    moved in between.
  """
  images = {}
  for pod in pods:
    if pod.status is None:
      continue
    statuses = (pod.status.init_container_statuses or []) + (
        pod.status.container_statuses or []
    )
    for container in statuses:
      if container.name not in PATHWAYS_CONTAINERS:
        continue
      digest = _DIGEST_RE.search(container.image_id or "")
      if not digest:
        continue
      entry = images.setdefault(
          container.name, {"image": container.image, "digests": []}
      )
      if digest.group() not in entry["digests"]:
        entry["digests"].append(digest.group())
  for entry in images.values():
    entry["digests"].sort()
  return images


def server_digests(records: Mapping[str, Mapping[str, Any]]) -> list[str]:
  """Returns every server image digest in `records`, sorted.

  Usually there is one. There are more if `server:latest` moved while the
  workloads started, and then the post-training image gets a tag for each.

  Args:
    records: What `record_pathways_images` returned for each workload.

  Returns:
    The digests of the images that the resource managers and workers ran.
  """
  digests = set()
  for record in records.values():
    for container in SERVER_CONTAINERS:
      digests.update(record["images"].get(container, {}).get("digests", []))
  return sorted(digests)


@task
def record_pathways_images(
    workload_id: str,
    project_id: str,
    region: str,
    cluster_name: str,
    namespace: str,
) -> dict[str, Any]:
  """Records the Pathways images that a workload runs, once it has started.

  Runs after `wait_for_workload_start`: the pods are deleted when the workload
  ends. If no Pathways container reports its image, the task is skipped rather
  than failed.

  Args:
    workload_id: The workload ID, from `decode_check.workload_id_of`.
    project_id: The project of the workload's GKE cluster.
    region: The region of the workload's GKE cluster.
    cluster_name: The name of the workload's GKE cluster.
    namespace: The Kubernetes namespace of the workload.

  Returns:
    The workload ID, and the images of its Pathways containers as
    `images_of_pods` returns them.
  """
  images = {}
  for attempt in range(1, _READ_ATTEMPTS + 1):
    try:
      core_api = gke.get_core_api_client(project_id, region, cluster_name)
      # With `use_pathways=False`, this lists the pods of every job of the
      # JobSet, so the worker pods too, not only the head pod.
      pods = gke.list_workload_pods(
          core_api,
          workload_id,
          use_pathways=False,
          workload_kind=gke.WorkloadKind.JOBSET,
          namespace=namespace,
      )
      images = images_of_pods(pods.items)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("Could not read the pods of %s: %s", workload_id, e)
    if attempt == _READ_ATTEMPTS or set(images) == set(PATHWAYS_CONTAINERS):
      break
    logging.info(
        "Found the images of %s so far. Reading again in %d seconds.",
        sorted(images),
        _SECONDS_BETWEEN_READS,
    )
    time.sleep(_SECONDS_BETWEEN_READS)

  if not images:
    raise AirflowSkipException(
        f"Found no Pathways container image in workload {workload_id}."
    )
  missing = sorted(set(PATHWAYS_CONTAINERS) - set(images))
  if missing:
    logging.warning("Found no image of %s in %s.", missing, workload_id)
  logging.info("Pathways images of %s: %s", workload_id, images)
  return {"workload_id": workload_id, "images": images}


@task(trigger_rule=TriggerRule.ALL_DONE)
def write_pathways_images(
    records: Mapping[str, Mapping[str, Any] | None],
    run_name: str,
    dag_run_id: str,
) -> str:
  """Writes the Pathways images of all the workloads of a run to GCS.

  Runs once every `record_pathways_images` task is done, whether it succeeded
  or not, and writes the file to `SUMMARY_OBJECT` in `SUMMARY_BUCKET`. If no
  workload recorded its images, or the file can't be written, the task is
  skipped rather than failed.

  Args:
    records: For each test, e.g. "qwen3-4b sft", what `record_pathways_images`
      returned, or None if it didn't return.
    run_name: The run name that all the stages of the E2E run share.
    dag_run_id: The ID of this post-training DAG run, which
      `read_pathways_server_digests` checks.

  Returns:
    The server image digests, comma-separated.
  """
  records = {test: record for test, record in records.items() if record}
  if not records:
    raise AirflowSkipException("No workload recorded its Pathways images.")
  object_name = SUMMARY_OBJECT.format(run_name=run_name)
  digests = server_digests(records)
  summary = {
      "run_name": run_name,
      "dag_run_id": dag_run_id,
      "server_digests": digests,
      "workloads": records,
  }
  try:
    GCSHook().upload(
        bucket_name=SUMMARY_BUCKET,
        object_name=object_name,
        data=json.dumps(summary, indent=2, sort_keys=True),
        mime_type="application/json",
        num_max_attempts=3,
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    raise AirflowSkipException(
        f"Could not write gs://{SUMMARY_BUCKET}/{object_name}: {e}"
    ) from e
  logging.info(
      "Wrote the Pathways images of %d workloads to gs://%s/%s. Server"
      " digests: %s",
      len(records),
      SUMMARY_BUCKET,
      object_name,
      digests,
  )
  return ",".join(digests)


@task(trigger_rule=TriggerRule.ALL_DONE)
def read_pathways_server_digests(
    run_name: str, post_training_run_id: str
) -> str:
  """Returns the server image digests that `write_pathways_images` wrote.

  Runs whether the post-training stage passed or not, and returns "" if there
  is no file, or if the file is from another post-training DAG run, so the
  GitHub callback still fires. A GitHub re-run keeps the run name, so without
  the check, a re-run whose `write_pathways_images` skipped would send the
  digests of an earlier attempt.

  Args:
    run_name: The run name that all the stages of the E2E run share.
    post_training_run_id: The ID of the post-training DAG run that this E2E run
      triggered.

  Returns:
    Every server image digest that the run used, sorted and comma-separated.
    Usually there is one; see `server_digests`.
  """
  object_name = SUMMARY_OBJECT.format(run_name=run_name)
  try:
    summary = json.loads(
        GCSHook().download(bucket_name=SUMMARY_BUCKET, object_name=object_name)
    )
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning(
        "Could not read gs://%s/%s: %s", SUMMARY_BUCKET, object_name, e
    )
    return ""
  if summary.get("dag_run_id") != post_training_run_id:
    logging.warning(
        "gs://%s/%s is from post-training DAG run %s, not %s. Ignoring it.",
        SUMMARY_BUCKET,
        object_name,
        summary.get("dag_run_id"),
        post_training_run_id,
    )
    return ""
  return ",".join(summary.get("server_digests", []))
