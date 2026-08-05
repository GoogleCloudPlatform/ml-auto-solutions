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
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

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

with models.DAG(
    dag_id="maxtext_validation_master_dag",
    schedule="0 0 * * *",  # Run nightly at midnight
    tags=["maxtext", "master", "nightly"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
    default_args={
        "retries": 3,
        "retry_delay": datetime.timedelta(minutes=15),
    },
    render_template_as_native_obj=True,
) as dag:

  # trigger Sub-DAG A (checkpoint inspection using the checkpoint inspection tool)
  trigger_checkpoint_shape_validation = TriggerDagRunOperator(
      task_id="trigger_checkpoint_shape_validation",
      trigger_dag_id="dag_verify_checkpoint_shape",
      conf="{{ params }}",  # passes master DAG's UI config down to Sub-DAG A
      wait_for_completion=True,  # Halt pipeline immediately if mismatched
  )

  # trigger Sub-DAG B (mock tensor validation/ dry run)
  trigger_mock_tensor_validation = TriggerDagRunOperator(
      task_id="trigger_mock_tensor_validation",
      trigger_dag_id="dag_verify_forward_compile",
      conf="{{ params }}",  # passes master DAG's UI config down to Sub-DAG B
      wait_for_completion=True,
  )

  # trigger Sub-DAG C (forward pass validation)
  trigger_forward_pass_validation = TriggerDagRunOperator(
      task_id="trigger_forward_pass_validation",
      trigger_dag_id="dag_verify_forward_pass",
      conf="{{ params }}",  # passes master DAG's UI config down to Sub-DAG C
      wait_for_completion=True,
  )

  # trigger Sub-DAG D (decoding validation)
  trigger_decoding_validation = TriggerDagRunOperator(
      task_id="trigger_decoding_validation",
      trigger_dag_id="dag_verify_decoding",
      conf="{{ params }}",  # passes master DAG's UI config down to Sub-DAG D
      wait_for_completion=True,
  )

  # execution order: Shape Validation (A) -> Mock Tensor (B) -> Forward Pass (C) -> Decoding (D)
  # pylint: disable=pointless-statement
  trigger_checkpoint_shape_validation >> trigger_mock_tensor_validation >> trigger_forward_pass_validation >> trigger_decoding_validation
