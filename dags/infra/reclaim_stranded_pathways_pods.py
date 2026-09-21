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

"""Temporary DAG that reclaims Pathways worker pods stranded by a slow GC.

When a Pathways workload finishes, the JobSet controller deletes the worker
Jobs and the control plane garbage collector is supposed to cascade that down
to the pods. On `bodaborg-v5p-nap` the GC is chronically backlogged, and that
cascade has been observed to sit in its queue for up to 47 minutes. While it
does, the worker pods stay in phase Running and keep the v5p chips busy.

This DAG polls the cluster and deletes those pods directly. A direct
`DELETE /api/v1/namespaces/{ns}/pods/{name}` is handled by the API server
immediately -- it does not go through the GC, which only processes cascade
deletion driven by ownerReferences.

This is a workaround, not a fix. Delete this file once b/563221051 is
resolved.

Safety rails, in the order they are applied:

  1. Only the `default` namespace of `bodaborg-v5p-nap` is touched.
  2. The JobSet name must match `_WORKLOAD_ID_PATTERN`, i.e. it must look
     like an ID minted by `xlml.utils.gcluster.generate_workload_id`. The
     cluster is shared -- other teams run workloads in the same namespace and
     several of theirs sit in a terminal state for days at a time.
  3. The workload must have a `pathways-head` pod, and every head pod must
     have stopped. A live head means training is still in progress.
  4. At least one head pod must be `Succeeded`. The JobSet's successPolicy is
     `All` over `pathways-head`, so a succeeded head is proof the workload
     completed. Failed heads are deliberately left alone: the JobSet is
     configured with `max_restart=3`, and deleting workers underneath a
     restart would interfere with it.
  5. Pods already carrying a `deletionTimestamp` are skipped, so a retry of
     this task is idempotent.
"""

import collections
import datetime
import logging
import re
from typing import Dict, List

from airflow import models
from airflow.decorators import task
import kubernetes

from dags import composer_env
from dags.common.vm_resource import GkeClusters
from xlml.utils import gke


# Poll often enough to beat the sensor. `wait_for_workload_completion` gives
# up after 60 minutes, so a daily sweep would always arrive too late to matter
# and would usually find the GC had already caught up.
SCHEDULED_TIME = "*/10 * * * *" if composer_env.is_prod_env() else None

# `GkeClusters` is a plain namespace class, so this is the dataclass itself
# rather than an enum member.
_CLUSTER = GkeClusters.TPU_V5P_BODABORG_NAP_CLUSTER
_NAMESPACE = _CLUSTER.namespace

_JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"
_REPLICATED_JOB_LABEL = "jobset.sigs.k8s.io/replicatedjob-name"

# The replicated job that runs the Pathways head. gcluster names the other one
# `worker`, but this DAG treats "not the head" as "a worker" so that it stays
# correct if the blueprint grows another replicated job.
_HEAD_REPLICATED_JOB = "pathways-head"

_LIVE_POD_PHASES = frozenset({"Pending", "Running", "Unknown"})

# Matches the IDs minted by `xlml.utils.gcluster.generate_workload_id`, which
# are `f"{benchmark_id}-{uuid4().hex[:5]}"` with
# `benchmark_id = f"{test_name}-{accelerator.name}"`, e.g. `rl-v5p-64-84b46`.
#
# The alternation is the set of `test_name`s that
# dags/multipod/maxtext_e2e_tpu_post_training.py can produce: its
# `post_training` modes are sft / multimodal_sft / lora / rl, and the DAG
# shortens `multimodal_sft` to `multim`. Keep this in sync if a mode is added.
#
# Scope is deliberately post-training only. Other DAGs mint IDs in the same
# shape and run in the same namespace -- `pre-v5p-64-b586d` from
# maxtext_e2e_tpu_pre_training.py was live on this cluster while this was
# written -- and they are intentionally left alone.
_WORKLOAD_ID_PATTERN = re.compile(
    r"^(?:rl|sft|multim|lora)-v\d+[a-z]*-\d+-[0-9a-f]{5}$"
)


def _phase(pod: kubernetes.client.V1Pod) -> str:
  return (pod.status.phase or "") if pod.status else ""


def _replicated_job(pod: kubernetes.client.V1Pod) -> str:
  return (pod.metadata.labels or {}).get(_REPLICATED_JOB_LABEL, "")


def group_pods_by_workload(
    pods: List[kubernetes.client.V1Pod],
) -> Dict[str, List[kubernetes.client.V1Pod]]:
  """Bucket pods by JobSet name, dropping anything that is not ours."""
  by_workload = collections.defaultdict(list)
  for pod in pods:
    workload_id = (pod.metadata.labels or {}).get(_JOBSET_NAME_LABEL, "")
    if _WORKLOAD_ID_PATTERN.match(workload_id):
      by_workload[workload_id].append(pod)
  return dict(by_workload)


def select_stranded_pods(
    workload_id: str, pods: List[kubernetes.client.V1Pod]
) -> List[kubernetes.client.V1Pod]:
  """Return the worker pods of a completed workload that are still alive.

  Returns an empty list unless the workload is provably finished. See the
  module docstring for the rules.
  """
  heads = [p for p in pods if _replicated_job(p) == _HEAD_REPLICATED_JOB]
  if not heads:
    logging.info(
        "Skipping %s: no %s pod, cannot tell whether it finished.",
        workload_id,
        _HEAD_REPLICATED_JOB,
    )
    return []

  live_heads = [h for h in heads if _phase(h) in _LIVE_POD_PHASES]
  if live_heads:
    logging.info(
        "Skipping %s: head pod %s is %s, training is still running.",
        workload_id,
        live_heads[0].metadata.name,
        _phase(live_heads[0]),
    )
    return []

  if not any(_phase(h) == "Succeeded" for h in heads):
    logging.info(
        "Skipping %s: no succeeded head pod (phases: %s). Leaving the JobSet "
        "alone in case it is restarting.",
        workload_id,
        sorted({_phase(h) for h in heads}),
    )
    return []

  return [
      p
      for p in pods
      if _replicated_job(p) != _HEAD_REPLICATED_JOB
      and _phase(p) in _LIVE_POD_PHASES
      and p.metadata.deletion_timestamp is None
  ]


@task
def reclaim_stranded_pathways_pods(dry_run: str = "False") -> int:
  """Delete worker pods that outlived a completed Pathways workload."""
  # Airflow renders `{{ params.dry_run }}` to a string, not a bool.
  is_dry_run = str(dry_run).strip().lower() in ("true", "1", "yes")

  core_api = gke.get_core_api_client(
      project_id=_CLUSTER.project,
      region=_CLUSTER.zone,
      cluster_name=_CLUSTER.name,
  )
  # A bare key is an existence selector: every pod owned by any JobSet.
  pod_list = core_api.list_namespaced_pod(
      namespace=_NAMESPACE, label_selector=_JOBSET_NAME_LABEL
  )
  by_workload = group_pods_by_workload(pod_list.items)
  logging.info(
      "Found %d pod(s) under %d workload(s) matching %s.",
      sum(len(v) for v in by_workload.values()),
      len(by_workload),
      _WORKLOAD_ID_PATTERN.pattern,
  )

  deleted = 0
  for workload_id, pods in sorted(by_workload.items()):
    stranded = select_stranded_pods(workload_id, pods)
    if not stranded:
      continue
    logging.info(
        "Workload %s completed but %d worker pod(s) are still alive.",
        workload_id,
        len(stranded),
    )
    for pod in stranded:
      name = pod.metadata.name
      if is_dry_run:
        logging.info("[dry run] Would delete pod %s (%s).", name, _phase(pod))
        continue
      try:
        core_api.delete_namespaced_pod(name=name, namespace=_NAMESPACE)
      except kubernetes.client.exceptions.ApiException as e:
        # 404 means someone else -- the GC, or the sensor -- got there first.
        if e.status != 404:
          logging.warning("Could not delete pod %s: %s", name, e)
        continue
      logging.info("Deleted stranded pod %s (was %s).", name, _phase(pod))
      deleted += 1

  logging.info("Reclaimed %d pod(s).", deleted)
  return deleted


with models.DAG(
    dag_id="reclaim_stranded_pathways_pods",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "clean_up", "pathways", "TPU"],
    start_date=datetime.datetime(2026, 9, 21),
    catchup=False,
    # Overlapping sweeps would race each other onto the same pods.
    max_active_runs=1,
    params={"dry_run": False},
    doc_md=__doc__,
) as dag:
  reclaim_stranded_pathways_pods(dry_run="{{ params.dry_run }}")
