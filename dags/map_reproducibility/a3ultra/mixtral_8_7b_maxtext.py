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

"""DAGs to run Aotc reproducibility benchmarks."""

import datetime

from airflow import models
from dags import composer_env
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.common_utils import get_scheduled_time
from dags.map_reproducibility.utils.common_utils import get_docker_image
from dags.map_reproducibility.utils.common_utils import run_maxtext_workload


MODEL_ID = "mixtral-8x7b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3ultra"
FRAMEWORK = "maxtext"

SCHEDULED_TIME = (
    get_scheduled_time(HYPERCOMPUTER, MODEL_ID, FRAMEWORK)
    if composer_env.is_prod_env()
    else None
)

SOFTWARE_ID = "jax_maxtext"
CLUSTER, CLUSTER_REGION = get_cluster(HYPERCOMPUTER)
IMAGE_VERSION = "maxtext-nightly"
DOCKER_IMAGE = get_docker_image(HYPERCOMPUTER, FRAMEWORK)
KUEUE_NAME = "a3-ultra"

OPTIMIZER = "adam"
SEQUENCE_LENGTH = 2048
NUM_STEPS = 30
BATCH_SIZE_PER_DEVICE = 5

with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}",
    schedule=SCHEDULED_TIME,
    tags=[
        "reproducibility",
        "experimental",
        "xlml",
        "regressiontests",
        "a3ultra",
        "maxtext",
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  run_maxtext_workload(
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      num_steps=NUM_STEPS,
      batch_size_per_device=BATCH_SIZE_PER_DEVICE,
      kueue_name=KUEUE_NAME,
      optimizer=OPTIMIZER,
      sequence_length=SEQUENCE_LENGTH,
      helm_model_id=MODEL_ID,
  )
