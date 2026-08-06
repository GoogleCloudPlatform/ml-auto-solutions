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

"""Master DAG to orchestrate nightly MaxText validations sequentially."""

# pylint: disable=line-too-long

import datetime
from airflow import models
from airflow.operators.python import PythonOperator
from airflow.api.common.trigger_dag import trigger_dag
from airflow.models import DagRun
from airflow.utils.session import provide_session
import time
import uuid

# Default payload passed to all downstream Sub-DAGs unless overridden in the UI.
# DEFAULT_PARAMS = {
#     "run_name": "qwen3-custom--test",
#     "checkpoint_gcs_path": "gs://maxtext-model-checkpoints/qwen3-8b/unscanned/0/items",
#     "maxtext_model_name": "qwen3-8b",
#     "maxtext_branch": "{{ dag_run.conf.get('maxtext_branch', 'main') }}",
#     "maxtext_commit_hash": "",
#     "report_gcs_dir": "gs://maxtext-validation-agent-reports/",
#     "hf_config_url": "",
#     "hf_ref_code_url": "",
#     "maxtext_overrides": {
#         "tokenizer_path": "Qwen/Qwen3-8B",
#         "tokenizer_type": "huggingface",
#         "scan_layers": False,
#         "max_target_length": 2048,
#         "per_device_batch_size": 8.0,
#         "attention": "dot_product",
#         "debug_tensors": True,
#     },
# }

DEFAULT_PARAMS = {
    "run_name": "qwen3-custom-test1",
    "xpk_project": "tpu-prod-env-multipod",
    "xpk_cluster_name": "v4-8-maxtext",
    "xpk_zone": "us-central2-b",
    "checkpoint_gcs_path": "gs://maxtext-model-checkpoints/qwen3-8b/unscanned/0/items",
    "maxtext_model_name": "qwen3-8b",
    "maxtext_branch": "feature/checkpoint-validation-clean",
    "maxtext_commit_hash": "",
    "report_gcs_dir": "gs://maxtext-validation-agent-reports/",
    "hf_model_path": "Qwen/Qwen3-8B",
    "hf_token": "",
    "hf_config_url": "https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json",
    "hf_ref_code_url": "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3/modeling_qwen3.py",
    "maxtext_overrides": {
        "attention": "dot_product",
        "debug_tensors": True,
        "max_target_length": 2048,
        "per_device_batch_size": 8,
        "rope_interleave": False,
        "scan_layers": False,
        "tokenizer_path": "Qwen/Qwen3-8B",
        "tokenizer_type": "huggingface",
    },
}

def monitor_agent_subdag(sub_dag_id, conf, **context):
    run_name = conf.get("run_name", "default_run")
    base_run_id = f"manual__{uuid.uuid4().hex[:8]}"

    print(f"Triggering initial run {base_run_id} for sub-DAG {sub_dag_id}")
    trigger_dag(dag_id=sub_dag_id, run_id=base_run_id, conf=conf)

    @provide_session
    def check_runs(session=None):
        runs = session.query(DagRun).filter(DagRun.dag_id == sub_dag_id).all()
        related_runs = []
        for r in runs:
            # Check if this run belongs to our master DAG execution
            if r.conf and r.conf.get("run_name", "").startswith(run_name):
                related_runs.append(r)
        return related_runs

    max_runs = 5
    while True:
        related_runs = check_runs()
        if not related_runs:
            time.sleep(30)
            continue
            
        successes = sum(1 for r in related_runs if r.state == "success")
        if successes > 0:
            print(f"Sub-DAG {sub_dag_id} succeeded! Total runs detected: {len(related_runs)}")
            return True
            
        failures = sum(1 for r in related_runs if r.state == "failed")
        running = sum(1 for r in related_runs if r.state == "running")
        
        # If we've hit the limit and no runs are currently running, fail the master DAG
        if len(related_runs) >= max_runs and running == 0:
            raise RuntimeError(f"Sub-DAG {sub_dag_id} failed {max_runs} times. Moving Master DAG to FAILED state.")
            
        print(f"Waiting for sub-DAG {sub_dag_id}... Active runs: {running}, Failures: {failures}/{max_runs}")
        time.sleep(30)

with models.DAG(
    dag_id="maxtext_validation_master_dag",
    schedule="0 0 * * *",  # Run nightly at midnight
    tags=["maxtext", "master", "nightly"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
    default_args={
        "retries": 0,
        "retry_delay": datetime.timedelta(minutes=15),
    },
    render_template_as_native_obj=True,
) as dag:

  trigger_checkpoint_shape_validation = PythonOperator(
      task_id="trigger_checkpoint_shape_validation",
      python_callable=monitor_agent_subdag,
      op_kwargs={"sub_dag_id": "dag_verify_checkpoint_shape", "conf": "{{ params }}"},
  )

  trigger_mock_tensor_validation = PythonOperator(
      task_id="trigger_mock_tensor_validation",
      python_callable=monitor_agent_subdag,
      op_kwargs={"sub_dag_id": "dag_verify_forward_compile", "conf": "{{ params }}"},
  )

  trigger_forward_pass_validation = PythonOperator(
      task_id="trigger_forward_pass_validation",
      python_callable=monitor_agent_subdag,
      op_kwargs={"sub_dag_id": "dag_verify_forward_pass", "conf": "{{ params }}"},
  )

  trigger_decoding_validation = PythonOperator(
      task_id="trigger_decoding_validation",
      python_callable=monitor_agent_subdag,
      op_kwargs={"sub_dag_id": "dag_verify_decoding", "conf": "{{ params }}"},
  )

  # execution order: Shape Validation (A) -> Mock Tensor (B) -> Forward Pass (C) -> Decoding (D)
  trigger_checkpoint_shape_validation >> trigger_mock_tensor_validation >> trigger_forward_pass_validation >> trigger_decoding_validation
