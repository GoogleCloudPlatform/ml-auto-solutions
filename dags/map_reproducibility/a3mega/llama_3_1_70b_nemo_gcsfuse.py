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
from dags.map_reproducibility.utils.common_utils import run_nemo_workload


MODEL_ID = "llama3-1-70b"
METRICS_MODEL_ID = "llama3.1-70b"
CONFIG_MODEL_NAME = "llama3.1-70b-256gpus-bf16-gcsfuse-checkpointing.yaml"
PRECISION = "bf16"
HYPERCOMPUTER = "a3mega"
FRAMEWORK = "nemo"
KUEUE_NAME = "multislice-kueue"
STORAGE_PRODUCT = "gcs"
CHECKPOINT_BUCKET = "cmcs-checkpoint-sydney"
DATASET_BUCKET = "cmcs-benchmark-raw-metrics"
GCS_METRICS_BUCKET = "cmcs-benchmark-raw-metrics"
LOGS_BUCKET = "cmcs-benchmark-logs"
BENCHMARK_TYPE = "checkpointing"
WORKLOAD_TYPE = "system"
WORKLOAD_IMAGE = "us-docker.pkg.dev/supercomputer-testing/dlsl-metadata/recipe-release-patched"
RECIPE_BRANCH = "storage-next"

default_dag_args = {
    "retries": 0,
}


with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}_gcsfuse_ckpt",
    tags=[
        "experimental",
        "regressiontests",
        "a3mega",
        "storage-run"
    ],
    start_date=datetime.datetime(2024, 11, 15),
    catchup=False,
    default_args=default_dag_args,
) as dag:
  run_nemo_workload(
      hypercomputer=HYPERCOMPUTER,
      model_id=MODEL_ID,
      framework=FRAMEWORK,
      precision=PRECISION,
      kueue_name=KUEUE_NAME,
      metrics_model_id=METRICS_MODEL_ID,
      config_model_name=CONFIG_MODEL_NAME,
      user="lepan",
      git_name="lepan",
      git_email="lepan@google.com",
      gcs_results_generator=True,
      storage_product=STORAGE_PRODUCT,
      recipe_branch=RECIPE_BRANCH,
      recipes_repo_change_refs="refs/changes/00/3800/3",
      bq_writer_repo_change_refs="",
      gcs_automation_repo_change_refs="refs/changes/23/2023/13",
      gcs_source_bucket=CHECKPOINT_BUCKET,
      gcs_metrics_bucket=GCS_METRICS_BUCKET,
      benchmark_type=BENCHMARK_TYPE,
      logs_bucket=LOGS_BUCKET,
      workload_type=WORKLOAD_TYPE,
      workload_image=WORKLOAD_IMAGE,
  )
