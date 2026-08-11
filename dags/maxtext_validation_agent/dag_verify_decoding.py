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

"""DAG to automate MaxText Checkpoint Decoding Validation (Sub-DAG D)."""

# pylint: disable=line-too-long

import datetime
from airflow import models
from dags.maxtext_validation_agent.lib import utils
from dags.maxtext_validation_agent.lib.utils import trigger_agent_on_failure
from dags.common.vm_resource import XpkClusters, TpuVersion

DEFAULT_PARAMS = {}

# Map over these clusters to automatically generate a DAG for each hardware type
SUPPORTED_CLUSTERS = [
    getattr(XpkClusters, attr) for attr in dir(XpkClusters)
    if not attr.startswith("__") and hasattr(getattr(XpkClusters, attr), "device_version") and isinstance(getattr(XpkClusters, attr).device_version, TpuVersion)
]

def create_decoding_dag(cluster_config):
  """Dynamically instantiates an Airflow DAG mapped to a specific XpkCluster."""
  dag_id = f"dag_verify_decoding_{cluster_config.name}"
  
  with models.DAG(
      dag_id=dag_id,
      schedule=None,
      tags=["maxtext", "checkpoint", "decoding", "validation"],
      start_date=datetime.datetime(2026, 6, 26),
      catchup=False,
      params=DEFAULT_PARAMS,
      default_args={
          "retries": 0,
          "on_failure_callback": trigger_agent_on_failure,
      },
  ) as dag:
    decoding_task = utils.get_decoding_validation_task(
        tpu_version=cluster_config.device_version.value,
        tpu_cores=cluster_config.core_count,
        tpu_zone=cluster_config.zone,
        tpu_project=cluster_config.project,
        time_out_in_min=45,
    ).run(skip_post_process=True)

    check_task = utils.get_upstream_failure_validator_task(dag)
    decoding_task >> check_task
    
  return dag

# Programmatically build the DAG graph objects required by Airflow Parser
for cluster in SUPPORTED_CLUSTERS:
  globals()[f"dag_verify_decoding_{cluster.name}"] = create_decoding_dag(cluster)

# Fallback basic DAG perfectly named backwards compatibility
globals()["dag_verify_decoding"] = create_decoding_dag(XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER)
