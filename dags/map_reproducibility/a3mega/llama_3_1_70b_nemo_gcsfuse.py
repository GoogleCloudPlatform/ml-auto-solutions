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
PRECISION = "bf16"
HYPERCOMPUTER = "a3mega"
FRAMEWORK = "nemo"
KUEUE_NAME = "multislice-kueue"
STORAGE_PRODUCT = "gcs"
CHECKPOINT_BUCKET = "cmcs-checkpoint-sydney"
DATASET_BUCKET = "cmcs-storage-training-benchmark"
GCS_METRICS_BUCKET = "cmcs-benchmark-raw-metrics"
LOGS_BUCKET = "cmcs-benchmark-logs"
BENCHMARK_TYPE = "checkpointing"
WORKLOAD_TYPE = "system"
WORKLOAD_IMAGE = "us-docker.pkg.dev/supercomputer-testing/dlsl-metadata/recipe-release-patched"
# Please choose the following values based on the recipe branch.
CONFIG_MODEL_NAME_MAIN_BRANCH = "llama3-1-70b-256gpus-bf16-pile-checkpointing.yaml"
CONFIG_MODEL_NAME_STORAGE_NEXT_BRANCH = "llama3.1-70b-256gpus-bf16-gcsfuse-checkpointing.yaml"
HELM_TEMPLATE_FOLDER_NAME_MAIN_BRANCH = "nemo-training-v2"
STORAGE_NEXT_BRANCH = "storage-next"

default_dag_args = {
    "retries": 0,
}


with models.DAG(
    dag_id=f"{HYPERCOMPUTER}_recipes_{MODEL_ID}_{FRAMEWORK}_gcsfuse_ckpt",
    tags=["experimental", "regressiontests", "a3mega", "storage-run"],
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
      config_model_name=CONFIG_MODEL_NAME_STORAGE_NEXT_BRANCH,
      user="lepan",
      git_name="lepan",
      git_email="lepan@google.com",
      gcs_results_generator=True,
      storage_product=STORAGE_PRODUCT,
      # recipe_branch="",
      recipe_branch=STORAGE_NEXT_BRANCH,
      # recipes_repo_change_refs="refs/changes/20/4120/1",
      recipes_repo_change_refs="refs/changes/40/4140/1",
      bq_writer_repo_change_refs="",
      gcs_automation_repo_change_refs="refs/changes/81/2081/1",
      gcs_source_bucket=CHECKPOINT_BUCKET,
      gcs_metrics_bucket=GCS_METRICS_BUCKET,
      benchmark_type=BENCHMARK_TYPE,
      logs_bucket=LOGS_BUCKET,
      checkpoint_bucket=CHECKPOINT_BUCKET,
      dataset_bucket=DATASET_BUCKET,
      workload_type=WORKLOAD_TYPE,
      workload_image=WORKLOAD_IMAGE,
      # Leave helm_template_folder empty if using storage-next branch.
      # helm_template_folder=HELM_TEMPLATE_FOLDER,
  )
