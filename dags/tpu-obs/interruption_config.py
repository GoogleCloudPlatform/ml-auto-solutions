"""Configuration for interruption scenarios."""

import enum

# --- BASE CONFIGURATION ---
# Common default settings that can be overridden by specific scenarios.
BASE_CONFIG = {
    "project_id": "tpu-prod-env-one-vm",  # Default project ID
    "max_time_diff_sec": 150,
    "max_log_results": 1000,
    "metric_aggregation": None,  # Default to no aggregation
}


class ResourceType(enum.Enum):
  GKE = "GKE"
  GCE = "GCE"


class InterruptionReason(enum.Enum):
  DEFRAGMENTATION = "Defragmentation"
  EVICTION = "Eviction"
  HOST_ERROR = "HostError"
  MIGRATE_ON_HWSW_MAINTENANCE = "MigrateOnHWSWMaintenance"
  HWSW_MAINTENANCE = "HWSWMaintenance"
  OTHER = "Other"
  BARE_METAL_PREEMPTION = "BareMetalPreemption"
  AUTO_REPAIR = "AutoRepair"
  AUTO_UPGRADE = "AutoUpgrade"
  AUTO_RESIZE = "AutoResize"


# --- RESOURCE TYPE DEFINITIONS ---
# Defines common metric and log properties for different resource types.
RESOURCE_TYPES = {
    ResourceType.GKE: {
        "metric_type": "kubernetes.io/node/interruption_count",
        "resource_type": "k8s_node",
        "hint": "GKE",
    },
    ResourceType.GCE: {
        "metric_type": "tpu.googleapis.com/instance/interruption_count",
        "resource_type": "tpu.googleapis.com/GceTpuWorker",
        "hint": "GCE",
    },
}

# --- INTERRUPTION REASON DEFINITIONS ---
# Defines specific metric and log filter parts for different interruption
# reasons.
INTERRUPTION_REASONS = {
    InterruptionReason.DEFRAGMENTATION: {
        "metric_label": "Defragmentation",
        "log_filter_fragment": (
            'protoPayload.methodName="compute.instances.preempted" '
        ),
    },
    InterruptionReason.EVICTION: {
        "metric_label": "Eviction",
        "log_filter_fragment": (
            'protoPayload.methodName="compute.instances.preempted" '
        ),
    },
    InterruptionReason.HOST_ERROR: {
        "metric_label": "HostError",
        "log_filter_fragment": (
            'protoPayload.methodName="compute.instances.hostError" '
        ),
    },
    InterruptionReason.MIGRATE_ON_HWSW_MAINTENANCE: {
        "metric_label": "Migrate on HW/SW Maintenance",
        "log_filter_fragment": (
            'protoPayload.methodName="compute.instances.migrateOnHostMaintenance" '
        ),
    },
    InterruptionReason.HWSW_MAINTENANCE: {
        "metric_label": "HW/SW Maintenance",
        "log_filter_fragment": (
            'protoPayload.methodName="compute.instances.terminateOnHostMaintenance" '
        ),
    },
    InterruptionReason.OTHER: {
        "metric_label": "Other",
        "log_filter_fragment": (
            'protoPayload.methodName="compute.instances.guestTerminate" OR'
            ' protoPayload.methodName="compute.instances.instanceManagerHaltForRestart" OR'
            ' protoPayload.methodName="compute.instances.stoppedDueToPdDoubleServe" OR'
            ' protoPayload.methodName="compute.instances.kmsKeyError" OR'
            ' protoPayload.methodName="compute.instances.shredmillKeyError" OR'
            ' protoPayload.methodName="compute.instances.invalidVmImage" OR'
            ' protoPayload.methodName="compute.instances.scratchDiskCreationFailed" OR'
            ' protoPayload.methodName="compute.instances.localSsdInitializationError" OR'
            ' protoPayload.methodName="compute.instances.localSsdInitializationKeyError" OR'
            ' protoPayload.methodName="compute.instances.localSsdVerifyTarError" OR'
            ' protoPayload.methodName="compute.instances.localSsdRecoveryAttempting" OR'
            ' protoPayload.methodName="compute.instances.localSsdRecoveryTimeoutError" OR'
            ' protoPayload.methodName="compute.instances.localSsdRecoveryFailedError" '
        ),
    },
    InterruptionReason.BARE_METAL_PREEMPTION: {
        "metric_label": "Bare Metal Preemption",
        "log_filter_fragment": (
            'protoPayload.methodName="compute.instances.baremetalCaretakerPreempted" '
        ),
    },
    InterruptionReason.AUTO_REPAIR: {
        "metric_label": "",
        "log_filter_fragment": "",
    },
    InterruptionReason.AUTO_UPGRADE: {
        "metric_label": "",
        "log_filter_fragment": "",
    },
    InterruptionReason.AUTO_RESIZE: {
        "metric_label": "",
        "log_filter_fragment": "",
    },
}


# Combines BASE_CONFIG, RESOURCE_TYPES, and INTERRUPTION_REASONS to build
# the seleted scenario.
def get_scenario_config(platform, reason):
  return _generate_scenario_config(
      platform, reason, RESOURCE_TYPES[platform], INTERRUPTION_REASONS[reason]
  )


def _generate_scenario_config(
    platform_key: str,
    reason_key: str,
    resource_type_config: dict[str, str],
    reason_config: dict[str, str],
) -> dict[str, str]:
  """Generates a single scenario configuration dictionary."""
  scenario_description = (
      f"Validation of {platform_key} "
      f"{'nodes ' if platform_key == 'GKE' else 'instances'}"
      f"{reason_key} interruption's metrics and logs."
  )
  output_filename = f"{platform_key}_{reason_key}_validation_report.json"

  metric_query = (
      f'resource.labels.project_id = "{BASE_CONFIG["project_id"]}" '
      f'metric.type = "{resource_type_config["metric_type"]}" '
      f'resource.type = "{resource_type_config["resource_type"]}" '
      f'metric.labels.interruption_reason = "{reason_config["metric_label"]}" '
  )

  return {
      **BASE_CONFIG,
      "description": scenario_description,
      "output_filename": output_filename,
      "metric_query_filter": metric_query,
      "log_query_filter": reason_config["log_filter_fragment"],
      "resource_type_hint": platform_key,
      "interruption_reason": reason_config["metric_label"],
  }
