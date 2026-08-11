from datetime import timedelta
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
#     "skip_decoding": models.Param(default=False, type="boolean", description="Skip broken decoding generation steps"),
#    "run_name": "qwen3-custom--test",
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

from airflow.models.param import Param
from dags.common.vm_resource import XpkClusters, TpuVersion

CLUSTER_NAMES = [
    getattr(XpkClusters, attr).name for attr in dir(XpkClusters)
    if not attr.startswith("__") and hasattr(getattr(XpkClusters, attr), "device_version") and isinstance(getattr(XpkClusters, attr).device_version, TpuVersion)
]

DEFAULT_PARAMS = {
    "email": Param(
        default="fiyinbenstowe@google.com",
        type="string",
        description="The email address to receive E2E validation failure/remediation reports."
    ),
    "xpk_cluster_name": Param(
        default="v5p-128-bodaborg-europe-west4-b",
        enum=CLUSTER_NAMES,
        description="Select the target XPK cluster hardware for this test."
    ),
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
    "decode_maxtext_overrides": {
        "attention": "dot_product",
        "scan_layers": True,
        "weight_dtype": "float32",
        "tokenizer_path": "Qwen/Qwen3-0.6B",
        "tokenizer_type": "huggingface"
    },
    "report_gcs_dir": "gs://maxtext-validation-agent-reports/",
    "skip_decoding": models.Param(default=False, type="boolean", description="Skip decoding validation steps"),
    "run_name": "qwen3-quick-test",
}


def monitor_agent_subdag(sub_dag_id, conf, **context):
    latest_conf = conf.copy() if isinstance(conf, dict) else dict(conf)
    ti = context.get("ti")
    
    if ti:
        task_order = [
            "trigger_checkpoint_shape_validation",
            "trigger_mock_tensor_validation",
            "trigger_forward_pass_validation",
            "trigger_decoding_validation"
        ]
        for prev_task in task_order:
            if prev_task == ti.task_id:
                break
            healed = ti.xcom_pull(task_ids=prev_task, key="healed_conf")
            if healed and isinstance(healed, dict):
                print(f"Pulling healed configuration from {prev_task}")
                # Recursively merge forward overrides just to be safe
                if "forward_pass_maxtext_overrides" in healed and "forward_pass_maxtext_overrides" in latest_conf:
                    merged_forward = latest_conf["forward_pass_maxtext_overrides"].copy()
                    merged_forward.update(healed["forward_pass_maxtext_overrides"])
                    latest_conf["forward_pass_maxtext_overrides"] = merged_forward
                    
                # Recursively merge decode overrides just to be safe
                if "decode_maxtext_overrides" in healed and "decode_maxtext_overrides" in latest_conf:
                    merged_decode = latest_conf["decode_maxtext_overrides"].copy()
                    merged_decode.update(healed["decode_maxtext_overrides"])
                    latest_conf["decode_maxtext_overrides"] = merged_decode

                # Merge any other top-level healed params natively
                for key, value in healed.items():
                    if key not in ("forward_pass_maxtext_overrides", "decode_maxtext_overrides"):
                        latest_conf[key] = value

    run_name = latest_conf.get("run_name", "default_run")
    # Check for dynamic DAG routing based on cluster for hardware-specific tasks
    if sub_dag_id in ("dag_verify_forward_pass", "dag_verify_decoding"):
        cluster_name = latest_conf.get("xpk_cluster_name")
        if cluster_name:
            sub_dag_id = f"{sub_dag_id}_{cluster_name}"

    master_run_id = context.get("run_id", "default")
    safe_master_run_id = "".join([c if c.isalnum() else "-" for c in master_run_id])
    if not run_name.endswith(safe_master_run_id):
        run_name = f"{run_name}-{safe_master_run_id}"
        latest_conf["run_name"] = run_name

    base_run_id = f"manual__{uuid.uuid4().hex[:8]}"

    print(f"Triggering initial run {base_run_id} for sub-DAG {sub_dag_id}")
    trigger_dag(dag_id=sub_dag_id, run_id=base_run_id, conf=latest_conf)

    @provide_session
    def check_runs(session=None):
        runs = session.query(DagRun).filter(DagRun.dag_id == sub_dag_id).all()
        related_runs = []
        for r in runs:
            # Check if this run belongs to our master DAG execution
            if r.conf and r.conf.get("run_name", "").startswith(run_name):
                related_runs.append(r)
        return related_runs

    max_runs = 25
    logged_runs = set()
    last_active_time = time.time()
    
    while True:
        related_runs = check_runs()
        if not related_runs:
            time.sleep(30)
            continue
            
        for r in related_runs:
            if r.run_id not in logged_runs:
                url = f"https://4bae0a6de8f94e92aa8ee3a6ffc8b278-dot-us-central1.composer.googleusercontent.com/dags/{sub_dag_id}/grid?dag_run_id={r.run_id}"
                print(f"Tracking new run spawned for {sub_dag_id}: {r.run_id}")
                print(f"View Run in Airflow UI: {url}")
                logged_runs.add(r.run_id)
            
        successes = [r for r in related_runs if r.state == "success"]
        if successes:
            successful_run = successes[-1]
            print(f"Sub-DAG {sub_dag_id} succeeded! Total runs detected: {len(related_runs)}")
            if successful_run.conf:
                ti.xcom_push(key="healed_conf", value=successful_run.conf)
            return True
            
        failures = sum(1 for r in related_runs if r.state == "failed")
        # An active run is anything that hasn't reached a terminal state (e.g. running, queued, scheduled)
        active_runs = sum(1 for r in related_runs if r.state not in ("success", "failed"))
        
        if active_runs > 0:
            last_active_time = time.time()
            
        # If we've hit the limit and no runs are currently active, fail the master DAG
        if len(related_runs) >= max_runs and active_runs == 0:
            cancel_active_agent_jobs()
            raise RuntimeError(f"Sub-DAG {sub_dag_id} failed {max_runs} times. Moving Master DAG to FAILED state.")
            
        # If it's been more than 60 minutes since the last run was active and we have failures, the agent likely crashed
        if active_runs == 0 and failures > 0 and (time.time() - last_active_time) > 3600:
            cancel_active_agent_jobs()
            raise RuntimeError(f"Sub-DAG {sub_dag_id} has a failure but no new runs have been triggered for 60 minutes. Agent likely crashed.")
            
        print(f"Waiting for sub-DAG {sub_dag_id}... Active runs: {active_runs}, Failures: {failures}/{max_runs}")
        time.sleep(30)


def cancel_active_agent_jobs(context=None):
    """Cancels any running Cloud Run agent jobs if the Master DAG fails or times out."""
    try:
        import subprocess
        print("Cancelling all active Cloud Run agent executions...")
        out = subprocess.check_output(
            ["gcloud", "run", "jobs", "executions", "list", "--job", "maxtext-validation-job", "--region", "us-central1", "--format", "value(name)"],
            text=True
        )
        for exec_name in out.strip().splitlines():
            if exec_name:
                subprocess.run(["gcloud", "run", "jobs", "executions", "cancel", exec_name.strip(), "--region", "us-central1", "--quiet"])
        print("Successfully sent cancellation signal to all active agent executions.")
    except Exception as e:
        print(f"Warning: Failed to cancel Cloud Run agent executions: {e}")


with models.DAG(
    dag_id="maxtext_validation_master_dag",
    schedule=None,  # Run manually
    tags=["maxtext", "master", "nightly"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
    default_args={
        "retries": 0,
        "execution_timeout": timedelta(hours=2),
        "retry_delay": datetime.timedelta(minutes=15),
        "on_failure_callback": cancel_active_agent_jobs,
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

  from airflow.operators.python import BranchPythonOperator
  from airflow.operators.empty import EmptyOperator

  def branch_decoding(**context):
      if context["params"].get("skip_decoding"):
          return "skip_decoding_step"
      return "trigger_decoding_validation"

  check_decoding_flag = BranchPythonOperator(
      task_id="check_decoding_flag",
      python_callable=branch_decoding,
  )

  trigger_decoding_validation = PythonOperator(
      task_id="trigger_decoding_validation",
      python_callable=monitor_agent_subdag,
      op_kwargs={"sub_dag_id": "dag_verify_decoding", "conf": "{{ params }}"},
  )

  skip_decoding_step = EmptyOperator(task_id="skip_decoding_step")

  check_decoding_flag >> trigger_decoding_validation
  check_decoding_flag >> skip_decoding_step

  # execution order: Shape Validation (A) -> Mock Tensor (B) -> Forward Pass (C) -> Decoding (D)
  trigger_checkpoint_shape_validation >> trigger_mock_tensor_validation >> trigger_forward_pass_validation >> check_decoding_flag
