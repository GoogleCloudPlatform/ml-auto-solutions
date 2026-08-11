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
from dags.maxtext_validation_agent.lib.utils import trigger_agent_on_failure


DEFAULT_PARAMS = {
    "checkpoint_gcs_path": "gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items",
    "hf_config_url": "",
    "hf_model_path": "Qwen/Qwen3-0.6B",
    "hf_ref_code_url": "",
    "hf_token": "",
    "max_kl_div": "0.02",
    "maxtext_branch": "test-pipeline-ckpt-validation",
    "maxtext_commit_hash": "",
    "maxtext_model_name": "qwen3-0.6b",
    "forward_pass_maxtext_overrides": {
        "attention": "dot_product",
        "scan_layers": True,
        "weight_dtype": "float32",
        "tokenizer_path": "Qwen/Qwen3-0.6B",
        "tokenizer_type": "huggingface"
    },
    "report_gcs_dir": "gs://maxtext-validation-agent-reports/",
    "run_name": "qwen3-tiny-fp32-test",
    "xpk_cluster_name": "v5p-128-bodaborg-europe-west4-b",
    "xpk_project": "cloud-tpu-multipod-dev",
    "xpk_zone": "europe-west4-b"
}

with models.DAG(
    dag_id="dag_verify_checkpoint_shape",
    schedule=None,
    tags=["maxtext", "checkpoint", "validation"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
    default_args={
        "retries": 0,
        "on_failure_callback": trigger_agent_on_failure,
    },
) as dag:

  # Looks for keys in runtime conf first (from manual JSON or Master DAG),
  # falls back to defaults if run is standalone.

  checkpoint_task = utils.get_checkpoint_shape_validation_task(
      dag=dag,
      model_name="{{ dag_run.conf.get('maxtext_model_name', params['maxtext_model_name']) }}",
      checkpoint_gcs_path="{{ dag_run.conf.get('checkpoint_gcs_path', params['checkpoint_gcs_path']) }}",
      scan_layers="{{ dag_run.conf.get('forward_pass_maxtext_overrides', params['forward_pass_maxtext_overrides']).get('scan_layers', False) | lower }}",
  )

  # Execute Task A
  check_task = utils.get_upstream_failure_validator_task(dag)
  checkpoint_task >> check_task
