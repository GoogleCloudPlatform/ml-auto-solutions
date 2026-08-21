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
    "checkpoint_gcs_path": "",
    "hf_config_url": "",
    "hf_model_path": "",
    "hf_ref_code_url": "",
    "hf_token": "",
    "max_kl_div": "",
    "maxtext_branch": "",
    "maxtext_commit_hash": "",
    "maxtext_model_name": "",
    "forward_pass_maxtext_overrides": {
        "attention": "",
        "scan_layers": "",
        "weight_dtype": "",
        "tokenizer_path": "",
        "tokenizer_type": "",
    },
    "report_gcs_dir": "",
    "run_name": "",
    "xpk_cluster_name": "",
    "xpk_project": "",
    "xpk_zone": "",
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
