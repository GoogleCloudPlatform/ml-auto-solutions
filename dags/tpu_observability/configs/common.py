from dataclasses import dataclass
import enum

from airflow.decorators import task
from dags.common.vm_resource import MachineVersion, TpuVersion
from xlml.utils import composer


@dataclass(frozen=True)
class TpuConfig:
  tpu_version: TpuVersion
  tpu_topology: str
  machine_version: MachineVersion


# Only one version of the machine is supported at the moment.
# Other versions (e.g., "ct5p-hightpu-4t") may be introduced later.
class MachineConfigMap(enum.Enum):
  V6E_16 = TpuConfig(
      tpu_version=TpuVersion.TRILLIUM,
      tpu_topology="4x4",
      machine_version=MachineVersion.CT6E_STAND_4T,
  )


@task(task_id="log_xlml_dashboard_metadata")
def log_metadata(
    cluster_project,
    region,
    zone,
    cluster_name,
    node_pool_name,
    workload_id,
    docker_image,
    accelerator_type,
    num_slices,
):
  composer.log_metadata_for_xlml_dashboard({
      "cluster_project": cluster_project,
      "region": region,
      "zone": zone,
      "cluster_name": cluster_name,
      "node_pool_name": node_pool_name,
      "workload_id": workload_id,
      "docker_image": docker_image,
      "accelerator_type": accelerator_type,
      "num_slices": num_slices,
  })

  return "Metadata Logged"
