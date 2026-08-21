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

"""DAG to automate MaxText Checkpoint Decoding Validation (Sub-DAG D)."""

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
    "decode_maxtext_overrides": {
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
    dag_id="dag_verify_decoding",
    schedule=None,
    tags=["maxtext", "checkpoint", "decoding", "validation"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
    default_args={
        "retries": 0,
        "on_failure_callback": trigger_agent_on_failure,
    },
) as dag:
    cluster_param = DEFAULT_PARAMS.get("xpk_cluster_name", "v4-8-maxtext")
    cluster_name = (
        cluster_param.default
        if hasattr(cluster_param, "default")
        else str(cluster_param)
    )
    cluster_config = utils.get_cluster_config(cluster_name)

    decoding_task = utils.get_decoding_validation_task(
        cluster_config=cluster_config,
        time_out_in_min=45,
    ).run(skip_post_process=True)

    check_task = utils.get_upstream_failure_validator_task(dag)
    decoding_task >> check_task
