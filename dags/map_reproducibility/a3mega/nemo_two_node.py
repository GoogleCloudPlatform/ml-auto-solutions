# Copyright 2024 Google LLC
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

"""DAGs to run hypercomputer recipes"""

import datetime

from airflow import models
from dags import composer_env
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.common_utils import get_docker_image
from dags.map_reproducibility.utils.common_utils import run_nemo_workload


MODEL_ID = "mixtral-8x7b"
METRICS_MODEL_ID = "mixtral-7b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3mega"
FRAMEWORK = "nemo"

SCHEDULED_TIME = "0 6 * * *" if composer_env.is_prod_env() else None

VALUE_YAML_PATH = (
    f"training/{HYPERCOMPUTER}/{MODEL_ID}/nemo-pretraining-gke/values.yaml"
)
CLUSTER, CLUSTER_REGION = get_cluster(HYPERCOMPUTER)
SOFTWARE_ID = "pytorch_nemo"
IMAGE_VERSION = "nemo24.07"
DOCKER_IMAGE = get_docker_image(HYPERCOMPUTER, FRAMEWORK)


with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_two_node_{FRAMEWORK}",
    schedule=SCHEDULED_TIME,
    tags=[
        "reproducibility",
        "experimental",
        "xlml",
        "regressiontests",
        "a3ultra",
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  run_nemo_workload(
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      metrics_model_id=METRICS_MODEL_ID,
      two_node=True,
  )
