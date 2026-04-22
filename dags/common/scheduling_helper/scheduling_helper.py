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

# Mock cluster to group TPU Interruption Validation DAGs
TPU_INTERRUPTION_MOCK_CLUSTER = XpkClusterConfig(
    name="tpu-interruption-validation-prod",
    device_version=TpuVersion.TRILLIUM,
    core_count=16,
    project="cienet-cmcs",
    zone=Zone.US_CENTRAL1_B.value,
)

DagIdToTimeout: TypeAlias = dict[str, dt.timedelta]
DefaultTimeout: dt.timedelta = dt.timedelta(minutes=30)
REGISTERED_DAGS: dict[str, DagIdToTimeout] = {
    TPU_OBS_MOCK_CLUSTER.name: {
        "gke_node_pool_label_update": DefaultTimeout,
        "gke_node_pool_status": DefaultTimeout,
        "jobset_rollback_ttr": dt.timedelta(minutes=90),
        "jobset_ttr_node_pool_resize": dt.timedelta(minutes=90),
        "jobset_ttr_pod_delete": dt.timedelta(minutes=90),
        "multi_host_nodepool_rollback": DefaultTimeout,
        "node_pool_ttr_disk_size": dt.timedelta(minutes=90),
        "node_pool_ttr_update_label": dt.timedelta(minutes=90),
        "tpu_info_format_validation_dag": DefaultTimeout,
        "tpu_sdk_monitoring_validation": DefaultTimeout,
        "jobset_ttr_kill_process": dt.timedelta(minutes=90),
        "jobset_uptime_validation": dt.timedelta(minutes=90),
        "tpu_info_metrics_verification": DefaultTimeout,
    },
    TPU_INTERRUPTION_MOCK_CLUSTER.name: {
        "validate_interruption_count_gce_bare_metal_preemption": DefaultTimeout,
        "validate_interruption_count_gce_host_error": DefaultTimeout,
        "validate_interruption_count_gke_bare_metal_preemption": DefaultTimeout,
        "validate_interruption_count_gke_hwsw_maintenance": DefaultTimeout,
        "validate_interruption_count_gke_defragmentation": DefaultTimeout,
        "validate_interruption_count_gce_hwsw_maintenance": DefaultTimeout,
        "validate_interruption_count_gke_host_error": DefaultTimeout,
        "validate_interruption_count_gce_defragmentation": DefaultTimeout,
        "validate_interruption_count_gce_other": DefaultTimeout,
        "validate_interruption_count_gce_migrate_on_hwsw_maintenance": DefaultTimeout,
        "validate_interruption_count_gce_eviction": DefaultTimeout,
        "validate_interruption_count_gke_other": DefaultTimeout,
        "validate_interruption_count_gke_migrate_on_hwsw_maintenance": DefaultTimeout,
        "validate_interruption_count_gke_eviction": DefaultTimeout,
    },
}


def get_dag_timeout(dag_id: str) -> dt.timedelta:
  """Searches the registry and returns the specific timeout for a DAG."""
  for cluster_dags in REGISTERED_DAGS.values():
    if dag_id in cluster_dags:
      return cluster_dags[dag_id]
  raise UnregisteredDagError(
      f"DAG '{dag_id}' is not registered. Please add it to REGISTERED_DAGS."
  )


class SchedulingError(ValueError):
  """Base class for scheduling errors."""

  pass


class UnregisteredDagError(SchedulingError):
  """Raised when a DAG is not found in REGISTERED_DAGS."""

  pass


class ScheduleWindowError(SchedulingError):
  """Raised when a schedule exceeds the 24-hour daily window."""

  pass


class StaleRegistrationError(SchedulingError):
  """
  Raised when a DAG is registered in the helper
  but does not exist in the DAG folder.
  """

  pass


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
        if offset >= dt.timedelta(hours=24) or timeout >= dt.timedelta(
            hours=24
        ):
          raise ScheduleWindowError(
              f"Schedule exceeds 24h window at '{dag_id} '"
              f"in cluster '{cluster_name}'."
          )

        if current_dag_id == dag_id:
          schedule = anchor + offset
          return f"{schedule.minute} {schedule.hour} * * {day_of_week.value}"
        offset += timeout + cls.DEFAULT_MARGIN

    raise UnregisteredDagError(
        f"DAG '{dag_id}' is not registered. Please add it to REGISTERED_DAGS."
    )
