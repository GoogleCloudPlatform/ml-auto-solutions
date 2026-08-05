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

"""DAG to automate MaxText Checkpoint Forward Pass Validation (Task C)."""

# pylint: disable=line-too-long

import datetime
from airflow import models
from dags.maxtext_validation_agent.lib import utils
from dags.maxtext_validation_agent.lib.utils import trigger_agent_on_failure
from dags.common.vm_resource import XpkClusters, TpuVersion

DEFAULT_PARAMS = {
    "email": "",
    "xpk_cluster_name": "",
    "xpk_project": "",
    "xpk_zone": "",
    "checkpoint_gcs_path": "",
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
}

with models.DAG(
    dag_id="dag_verify_forward_pass",
    schedule=None,
    tags=["maxtext", "checkpoint", "forward_pass"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
    default_args={
        "retries": 0,
        "on_failure_callback": trigger_agent_on_failure,
    },
) as dag:
    cluster_name = DEFAULT_PARAMS.get("xpk_cluster_name", "v4-8-maxtext")
    cluster_config = utils.get_cluster_config(cluster_name)

    def check_golden_logits_exist(**context):
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run and dag_run.conf else context.get("params", {})
        overrides = conf.get("forward_pass_maxtext_overrides", {})
        hf_model = conf.get("hf_model_path", overrides.get("hf_model_path", ""))
        max_model = conf.get("maxtext_model_name", "")
        bucket_name = "maxtext-validation-golden-logits"
        blob_name = f"golden-logits/{hf_model}/{max_model}_golden_logits.jsonl"

        from airflow.providers.google.cloud.hooks.gcs import GCSHook

        hook = GCSHook()
        if hook.exists(bucket_name=bucket_name, object_name=blob_name):
            print(
                f"Golden logits found at gs://{bucket_name}/{blob_name}. Skipping generation."
            )
            return "skip_golden_logits"
        print(f"Golden logits missing at gs://{bucket_name}/{blob_name}. Generating...")
        return "start_golden_logits"

    from airflow.operators.python import BranchPythonOperator
    from airflow.operators.empty import EmptyOperator

    check_logits = BranchPythonOperator(
        task_id="check_golden_logits_exist",
        python_callable=check_golden_logits_exist,
    )

    skip_golden_logits = EmptyOperator(task_id="skip_golden_logits")

    # A dummy task is needed because golden_logits_task is a dynamically named TaskGroup.
    # The BranchPythonOperator must return a rigid, known task_id.
    start_golden_logits = EmptyOperator(task_id="start_golden_logits")

    golden_logits_task = utils.get_golden_logits_generation_task(
        time_out_in_min=120,
    ).run(skip_post_process=True)

    # Added to join the branch execution without skipping downstream tasks
    join_golden_logits = EmptyOperator(
        task_id="join_golden_logits", trigger_rule="none_failed"
    )

    forward_pass_task = utils.get_forward_pass_validation_task(
        cluster_config=cluster_config,
        time_out_in_min=45,
    ).run(skip_post_process=True)

    check_task = utils.get_upstream_failure_validator_task(dag)

    (check_logits >> start_golden_logits >> golden_logits_task >> join_golden_logits)
    check_logits >> skip_golden_logits >> join_golden_logits
    join_golden_logits >> forward_pass_task >> check_task
