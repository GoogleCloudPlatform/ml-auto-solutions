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

"""DAG to automate MaxText Checkpoint Structural Shape Validation."""

# pylint: disable=line-too-long

import datetime
from airflow import models
from dags.maxtext_validation_agent.lib import utils


DEFAULT_PARAMS = {
    "run_name": "qwen3-custom-shape-test",
    "checkpoint_gcs_path": "gs://maxtext-model-checkpoints/qwen3-8b/unscanned/0/items",
    "maxtext_model_name": "qwen3-8b",
    "maxtext_branch": "feat/mock-tensor-validation",
    "maxtext_commit_hash": "",
    "report_gcs_dir": "gs://maxtext-validation-agent-reports/",
    "maxtext_overrides": {
        "tokenizer_path": "Qwen/Qwen3-8B-Instruct",
        "tokenizer_type": "huggingface",
        "scan_layers": False,
        "max_target_length": 2048,
        "per_device_batch_size": 8.0,
        "attention": "dot_product",
    },
}

with models.DAG(
    dag_id="dag_verify_checkpoint_shape",
    schedule=None,
    tags=["maxtext", "checkpoint", "validation"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
) as dag:

  # Looks for keys in runtime conf first (from manual JSON or Master DAG),
  # falls back to defaults if run is standalone.

  checkpoint_task = utils.get_checkpoint_shape_validation_task(
      dag=dag,
      model_name="{{ dag_run.conf.get('maxtext_model_name', params['maxtext_model_name']) }}",
      checkpoint_gcs_path="{{ dag_run.conf.get('checkpoint_gcs_path', params['checkpoint_gcs_path']) }}",
      scan_layers="{{ dag_run.conf.get('maxtext_overrides', params['maxtext_overrides']).get('scan_layers', False) | lower }}",
  )

  # Execute Task A
  # pylint: disable=pointless-statement
  checkpoint_task
