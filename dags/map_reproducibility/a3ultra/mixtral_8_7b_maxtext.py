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
import os
import tempfile

from airflow import models
from airflow.decorators import task
from airflow.hooks.subprocess import SubprocessHook
from dags import composer_env
from dags.map_reproducibility.utils.common_utils import configure_project_and_cluster
from dags.map_reproducibility.utils.common_utils import install_helm_cmds
from dags.map_reproducibility.utils.common_utils import namespace_cmds
from dags.map_reproducibility.utils.common_utils import wait_for_jobs_cmds
from dags.map_reproducibility.utils.common_utils import cleanup_cmds
from dags.map_reproducibility.utils.common_utils import git_cookie_authdaemon
from dags.map_reproducibility.utils.common_utils import clone_recipes_gob
from dags.map_reproducibility.utils.common_utils import helm_apply_cmds
from dags.map_reproducibility.utils.common_utils import get_bq_writer_repo
from dags.map_reproducibility.utils.benchmarkdb_utils import write_run
from dags.map_reproducibility.utils.common_utils import extract_gpus
from dags.map_reproducibility.utils.common_utils import get_pre_workload_cmds
from dags.map_reproducibility.utils.common_utils import get_gpu_recipe_cmd
from dags.map_reproducibility.utils.common_utils import get_bq_writer_path
from dags.map_reproducibility.utils.common_utils import get_recipe_repo_path
from dags.map_reproducibility.utils.common_utils import get_cluster
from dags.map_reproducibility.utils.common_utils import get_scheduled_time
from dags.map_reproducibility.utils.common_utils import get_docker_image
from dags.map_reproducibility.utils.common_utils import calculate_maxtext_metrics
from dags.map_reproducibility.utils.common_utils import copy_bucket_cmds_maxtext
from dags.map_reproducibility.utils.common_utils import run_maxtext_workload


MODEL_ID = "mixtral-8x7b"
PRECISION = "bf16"
HYPERCOMPUTER = "a3ultra"
FRAMEWORK = "maxtext"
VALUE_YAML_PATH = (
    f"training/{HYPERCOMPUTER}/{MODEL_ID}/maxtext-pretraining-gke/values.yaml"
)

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
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
) as dag:
  run_maxtext_workload(
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      value_yaml_path=VALUE_YAML_PATH,
      num_steps=NUM_STEPS,
      batch_size_per_device=BATCH_SIZE_PER_DEVICE,
      kueue_name=KUEUE_NAME,
      optimizer=OPTIMIZER,
      sequence_length=SEQUENCE_LENGTH,
      helm_model_id=MODEL_ID,
  )
