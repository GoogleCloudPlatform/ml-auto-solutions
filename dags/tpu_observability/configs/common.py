from dataclasses import dataclass
import enum

from dags.common.vm_resource import MachineVersion, TpuVersion


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


GCS_CONFIG_PATH = (
    "gs://ml-auto-solutions-dag-configs/tpu_observability/dag_config.yaml"
)

GCS_JOBSET_CONFIG_PATH = (
    "gs://ml-auto-solutions-dag-configs/tpu_observability/jobset_config.yaml"
)
