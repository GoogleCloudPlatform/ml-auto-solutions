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

import datetime
from airflow import models
from dags import composer_env
from dags.map_reproducibility.utils.constants import Schedule
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.internal_aotc_workload import cleanup_cml_workloads

TEST_RUN = False if composer_env.is_prod_env() else True
# Define common tags
DAG_TAGS = [
    "reproducibility",
    "experimental",
    "cleanup",
    "CPU",
]

dag_default_args = {
    "retries": 0,
}

for hypercomputer in ["a3mega", "a3ultra", "a4"]:
  cluster, cluster_region = get_cluster(hypercomputer)
  schedule = (
      Schedule.WEEKDAY_PDT_6AM_7AM_EXCEPT_THURSDAY if not TEST_RUN else None
  )
  with models.DAG(
      dag_id=f"new_internal_cleanup_{hypercomputer}",
      default_args=dag_default_args,
      schedule=schedule,
      tags=DAG_TAGS,
      start_date=datetime.datetime(2025, 1, 1),
      catchup=False,
  ) as dag:
    cleanup_cml_workloads(cluster=cluster, cluster_region=cluster_region)
