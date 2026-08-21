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
from airflow.sensors.base import BaseSensorOperator
from airflow.exceptions import AirflowException
from airflow.api.common.trigger_dag import trigger_dag
from airflow.models import DagRun
from airflow.utils.session import create_session, provide_session
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

DEFAULT_PARAMS = {
    "email": "",
    "xpk_cluster_name": "",
    "xpk_project": "",
    "xpk_zone": "",
    "checkpoint_gcs_path": "",
    "decode_maxtext_overrides": {
        "attention": "",
        "per_device_batch_size": "",
        "scan_layers": "",
        "tokenizer_path": "",
        "tokenizer_type": "",
        "weight_dtype": "",
    },
    "forward_pass_maxtext_overrides": {
        "attention": "",
        "per_device_batch_size": "",
        "scan_layers": "",
        "tokenizer_path": "",
        "tokenizer_type": "",
        "weight_dtype": "",
    },
    "hf_config_url": "",
    "hf_model_path": "",
    "hf_ref_code_url": "",
    "hf_token": "",
    "max_kl_div": "",
    "maxtext_branch": "",
    "maxtext_commit_hash": "",
    "maxtext_model_name": "",
    "report_gcs_dir": "",
    "run_name": "",
    "skip_decoding": Param(
        default=False,
        type="boolean",
        description="Skip decoding validation steps",
    ),
}


class SubDagAgentMonitorSensor(BaseSensorOperator):
    """Asynchronous reschedule sensor that triggers and monitors a sub-DAG with self-healing agent retries."""

    template_fields = ("sub_dag_id", "conf")

    def __init__(self, sub_dag_id: str, conf: dict = None, max_runs: int = 6, **kwargs):
        kwargs.setdefault("mode", "reschedule")
        kwargs.setdefault("poke_interval", 30)
        kwargs.setdefault("timeout", 7200)
        super().__init__(**kwargs)
        self.sub_dag_id = sub_dag_id
        self.conf = conf or {}
        self.max_runs = max_runs

    def _resolve_conf_and_dag_id(self, context):
        ti = context.get("ti")
        dag_run = context.get("dag_run")
        params = context.get("params", {})

        latest_conf = {}
        if isinstance(params, dict):
            latest_conf.update(params)
        elif hasattr(params, "dump"):
            latest_conf.update(params.dump())

        if dag_run and dag_run.conf:
            latest_conf.update(dag_run.conf)

        if isinstance(self.conf, dict):
            latest_conf.update(self.conf)

        if ti:
            task_order = [
                "trigger_checkpoint_shape_validation",
                "trigger_mock_tensor_validation",
                "trigger_forward_pass_validation",
                "trigger_decoding_validation",
            ]
            for prev_task in task_order:
                if prev_task == ti.task_id:
                    break
                healed = ti.xcom_pull(task_ids=prev_task, key="healed_conf")
                if healed and isinstance(healed, dict):
                    if (
                        "forward_pass_maxtext_overrides" in healed
                        and "forward_pass_maxtext_overrides" in latest_conf
                    ):
                        merged = latest_conf["forward_pass_maxtext_overrides"].copy()
                        merged.update(healed["forward_pass_maxtext_overrides"])
                        latest_conf["forward_pass_maxtext_overrides"] = merged
                    if (
                        "decode_maxtext_overrides" in healed
                        and "decode_maxtext_overrides" in latest_conf
                    ):
                        merged = latest_conf["decode_maxtext_overrides"].copy()
                        merged.update(healed["decode_maxtext_overrides"])
                        latest_conf["decode_maxtext_overrides"] = merged
                    for key, value in healed.items():
                        if key not in (
                            "forward_pass_maxtext_overrides",
                            "decode_maxtext_overrides",
                        ):
                            latest_conf[key] = value

        target_dag_id = self.sub_dag_id

        run_name = latest_conf.get("run_name", "default_run")
        master_run_id = context.get("run_id", "default")
        safe_master_run_id = "".join([c if c.isalnum() else "-" for c in master_run_id])
        if not run_name.endswith(safe_master_run_id):
            run_name = f"{run_name}-{safe_master_run_id}"
            latest_conf["run_name"] = run_name

        return target_dag_id, latest_conf, run_name

    def poke(self, context) -> bool:
        ti = context.get("ti")
        target_dag_id, latest_conf, run_name = self._resolve_conf_and_dag_id(context)

        with create_session() as session:
            runs = session.query(DagRun).filter(DagRun.dag_id == target_dag_id).all()
            related_runs = [
                r
                for r in runs
                if r.conf and str(r.conf.get("run_name", "")).startswith(run_name)
            ]

        if not related_runs:
            base_run_id = f"manual__{uuid.uuid4().hex[:8]}"
            self.log.info(
                "Triggering initial run %s for sub-DAG %s with run_name %s",
                base_run_id,
                target_dag_id,
                run_name,
            )
            trigger_dag(dag_id=target_dag_id, run_id=base_run_id, conf=latest_conf)
            return False

        for r in related_runs:
            url = f"https://4bae0a6de8f94e92aa8ee3a6ffc8b278-dot-us-central1.composer.googleusercontent.com/dags/{target_dag_id}/grid?dag_run_id={r.run_id}"
            self.log.info(
                "Sub-DAG %s run %s state: %s | Airflow UI: %s",
                target_dag_id,
                r.run_id,
                r.state,
                url,
            )

        successes = [r for r in related_runs if r.state == "success"]
        if successes:
            successful_run = successes[-1]
            self.log.info(
                "Sub-DAG %s succeeded! Total runs detected: %d",
                target_dag_id,
                len(related_runs),
            )
            if successful_run.conf:
                ti.xcom_push(key="healed_conf", value=successful_run.conf)
            return True

        failures = sum(1 for r in related_runs if r.state == "failed")
        active_runs = sum(
            1 for r in related_runs if r.state not in ("success", "failed")
        )

        if len(related_runs) >= self.max_runs and active_runs == 0:
            cancel_active_agent_jobs()
            raise AirflowException(
                f"Sub-DAG {target_dag_id} failed {failures} times. Moving Master DAG to FAILED state."
            )

        self.log.info(
            "Waiting for sub-DAG %s... Active: %d, Failures: %d/%d",
            target_dag_id,
            active_runs,
            failures,
            self.max_runs,
        )
        return False


def cancel_active_agent_jobs(_context=None):
    """Cancels any running Cloud Run agent jobs if the Master DAG fails or times out."""
    logger = logging.getLogger(__name__)
    try:
        logger.info("Cancelling all active Cloud Run agent executions...")
        out = subprocess.run(
            [
                "gcloud",
                "run",
                "jobs",
                "executions",
                "list",
                "--job",
                "maxtext-validation-job",
                "--region",
                "us-central1",
                "--project",
                "tpu-prod-env-multipod",
                "--format",
                "value(name)",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout

        for exec_name in out.strip().splitlines():
            exec_name = exec_name.strip()
            if not exec_name:
                continue
            # Explicitly set check=False so we continue cancelling other executions even if one fails.
            res = subprocess.run(
                [
                    "gcloud",
                    "run",
                    "jobs",
                    "executions",
                    "cancel",
                    exec_name,
                    "--region",
                    "us-central1",
                    "--project",
                    "tpu-prod-env-multipod",
                    "--quiet",
                ],
                text=True,
                check=False,
                capture_output=True,
            )
            if res.returncode != 0:
                logger.warning(
                    "Failed to cancel execution %s: returncode=%d stdout=%s stderr=%s",
                    exec_name,
                    res.returncode,
                    res.stdout,
                    res.stderr,
                )

        logger.info(
            "Successfully sent cancellation signal to all active agent executions."
        )
    except subprocess.CalledProcessError as e:
        logger.exception("gcloud list command failed: %s", e)
    except OSError as e:
        logger.exception("OS error while invoking gcloud: %s", e)
    except Exception:
        # If you must catch a very broad exception, at least record full stack trace.
        logger.exception("Unexpected error while cancelling Cloud Run agent executions")


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
    trigger_checkpoint_shape_validation = SubDagAgentMonitorSensor(
        task_id="trigger_checkpoint_shape_validation",
        sub_dag_id="dag_verify_checkpoint_shape",
    )

    trigger_mock_tensor_validation = SubDagAgentMonitorSensor(
        task_id="trigger_mock_tensor_validation",
        sub_dag_id="dag_verify_forward_compile",
    )

    trigger_forward_pass_validation = SubDagAgentMonitorSensor(
        task_id="trigger_forward_pass_validation",
        sub_dag_id="dag_verify_forward_pass",
    )

    def branch_decoding(**context):
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run and dag_run.conf else context.get("params", {})
        if conf.get("skip_decoding"):
            return "skip_decoding_step"
        return "trigger_decoding_validation"

    check_decoding_flag = BranchPythonOperator(
        task_id="check_decoding_flag",
        python_callable=branch_decoding,
    )

    trigger_decoding_validation = SubDagAgentMonitorSensor(
        task_id="trigger_decoding_validation",
        sub_dag_id="dag_verify_decoding",
    )

    skip_decoding_step = EmptyOperator(task_id="skip_decoding_step")

    check_decoding_flag >> trigger_decoding_validation
    check_decoding_flag >> skip_decoding_step

    # execution order: Shape Validation (A) -> Mock Tensor (B) -> Forward Pass (C) -> Decoding (D)
    (
        trigger_checkpoint_shape_validation
        >> trigger_mock_tensor_validation
        >> trigger_forward_pass_validation
        >> check_decoding_flag
    )
