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

"""DAG to automate MaxText Checkpoint Inspection (Task B)."""

# pylint: disable=line-too-long

import datetime
from airflow import models
from dags.maxtext_validation_agent.lib import utils
from dags.maxtext_validation_agent.lib.utils import trigger_agent_on_failure


DEFAULT_PARAMS = {
    "run_name": "qwen3-custom-inspection-test",
    "checkpoint_gcs_path": "gs://maxtext-model-checkpoints/qwen3-8b/unscanned/0/items",
    "maxtext_model_name": "qwen3-8b",
    "maxtext_branch": "{{ dag_run.conf.get('maxtext_branch', 'main') }}",
    "maxtext_commit_hash": "",
    "report_gcs_dir": "gs://maxtext-validation-agent-reports/",
    "hf_model_path": "Qwen/Qwen3-8B",
    "hf_config_url": "",
    "hf_ref_code_url": "",
    "maxtext_overrides": {
        "tokenizer_path": "Qwen/Qwen3-8B",
        "tokenizer_type": "huggingface",
        "scan_layers": False,
        "max_target_length": 2048,
        "per_device_batch_size": 8.0,
        "attention": "dot_product",
        "debug_tensors": True,
    },
}

with models.DAG(
    dag_id="dag_verify_forward_compile",
    schedule=None,
    tags=["maxtext", "checkpoint", "inspection"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
    default_args={
        "retries": 0,
        "on_failure_callback": trigger_agent_on_failure,
    },
) as dag:

  # Looks for keys in runtime conf first (from manual JSON or Master DAG),
  # falls back to defaults if run standalone.
  forward_compile_task = utils.get_forward_compile_validation_task(
      dag=dag,
  )

  # Execute Task B
  check_task = utils.get_upstream_failure_validator_task(dag)
  forward_compile_task >> check_task
