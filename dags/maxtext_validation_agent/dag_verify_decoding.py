"""DAG to automate MaxText Checkpoint Decoding Validation (Sub-DAG D)."""

# pylint: disable=line-too-long

import datetime
from airflow import models
from dags.maxtext_validation_agent.lib import utils
from dags.maxtext_validation_agent.lib.utils import trigger_agent_on_failure


DEFAULT_PARAMS = {
    "run_name": "qwen3-custom-decoding-test",
    "xpk_project": "tpu-prod-env-multipod",
    "xpk_cluster_name": "v4-8-maxtext",
    "xpk_zone": "us-central2-b",
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
        "max_target_length": 128,
        "max_prefill_predict_length": 16,
        "per_device_batch_size": 8.0,
        "attention": "dot_product",
        "rope_interleave": False,
        "debug_tensors": True,
        "prompt": "I love to ",
        "autoregressive_decode_assert": "",
    },
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

  # Execute Sub-DAG D: End-to-End Decoding / Text Generation Verification
  # Uses defaults if run standalone, or conf if triggered by Master DAG.
  decoding_task = utils.get_decoding_validation_task(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone="us-central2-b",
      time_out_in_min=45,
  ).run(skip_post_process=True)

  check_task = utils.get_upstream_failure_validator_task(dag)
  decoding_task >> check_task
