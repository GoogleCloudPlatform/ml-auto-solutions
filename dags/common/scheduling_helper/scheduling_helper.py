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

"""Helper module for scheduling DAGs across clusters."""
import datetime as dt
import enum
from typing import TypeAlias

from xlml.apis.xpk_cluster_config import XpkClusterConfig
from dags.common.vm_resource import TpuVersion, Zone


class DayOfWeek(enum.Enum):
  ALL = "*"
  WEEK_DAY = "1-5"
  WEEKEND = "0,6"


# Mock cluster to group TPU Observability DAGs
TPU_OBS_MOCK_CLUSTER = XpkClusterConfig(
    name="tpu-observability-automation-prod",
    device_version=TpuVersion.TRILLIUM,
    core_count=16,
    project="cienet-cmcs",
    zone=Zone.US_CENTRAL1_B.value,
)

DagIdToTimeout: TypeAlias = dict[str, dt.timedelta]

REGISTERED_DAGS: dict[str, DagIdToTimeout] = {
    TPU_OBS_MOCK_CLUSTER.name: {
        "gke_node_pool_label_update": dt.timedelta(minutes=30),
        "gke_node_pool_status": dt.timedelta(minutes=30),
        "jobset_rollback_ttr": dt.timedelta(minutes=90),
        "jobset_ttr_node_pool_resize": dt.timedelta(minutes=90),
        "jobset_ttr_pod_delete": dt.timedelta(minutes=90),
        "multi_host_nodepool_rollback": dt.timedelta(minutes=30),
        "node_pool_ttr_disk_size": dt.timedelta(minutes=90),
        "node_pool_ttr_update_label": dt.timedelta(minutes=90),
        "tpu_info_format_validation_dag": dt.timedelta(minutes=30),
        "tpu_sdk_monitoring_validation": dt.timedelta(minutes=30),
        "jobset_ttr_kill_process": dt.timedelta(minutes=90),
        "jobset_uptime_validation": dt.timedelta(minutes=90),
    },
}


def get_dag_timeout(dag_id: str) -> dt.timedelta:
  """Searches the registry and returns the specific timeout for a DAG."""
  for cluster_dags in REGISTERED_DAGS.values():
    if dag_id in cluster_dags:
      return cluster_dags[dag_id]
  raise ValueError(
      f"DAG '{dag_id}' is not registered. Please add it to REGISTERED_DAGS."
  )


class SchedulingHelper:
  """Manages DAG scheduling across different clusters."""

  DEFAULT_MARGIN = dt.timedelta(minutes=15)
  DEFAULT_ANCHOR = dt.datetime(2000, 1, 1, 8, 0, 0, tzinfo=dt.timezone.utc)

  @classmethod
  def arrange_schedule_time(
      cls,
      dag_id: str,
      day_of_week: DayOfWeek = DayOfWeek.ALL,
  ) -> str:
    """Calculates a cron schedule by stacking timeouts and margins."""
    anchor = cls.DEFAULT_ANCHOR

    for cluster_name, dags in REGISTERED_DAGS.items():
      if dag_id not in dags:
        continue

      offset = dt.timedelta(0)
      for current_dag_id, timeout in dags.items():
        if current_dag_id == dag_id:
          schedule = anchor + offset
          return f"{schedule.minute} {schedule.hour} * * {day_of_week.value}"
        offset += timeout + cls.DEFAULT_MARGIN

        if offset >= dt.timedelta(hours=24):
          raise ValueError(
              f"Schedule exceeds 24h window at '{dag_id}' in cluster '{cluster_name}'."
          )

    raise ValueError(
        f"DAG '{dag_id}' is not registered. Please add it to REGISTERED_DAGS."
    )
