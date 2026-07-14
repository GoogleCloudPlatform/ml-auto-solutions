"""Master DAG to orchestrate all nightly MaxText validations."""

import datetime
from airflow import models
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

# configuration for Gemma 3 4B on the v4-8 cluster
GEMMA_NIGHTLY_CONFIG = {
    "run_name": "nightly-gemma3-4b-validation",
    "xpk_cluster_name": "v4-8-maxtext", # running cluster
    "checkpoint_gcs_path": "gs://maxtext-model-checkpoints/gemma3-4b/2025-03-18-19-03/unscanned/checkpoints/0/items",
    "maxtext_model_name": "gemma3-4b",
    "maxtext_overrides": {
        "tokenizer_path": "",
        "hf_path": "google/gemma-3-4b-it",
        "tokenizer_type": "huggingface",
        "scan_layers": False,
        "max_target_length": 4096,
        "per_device_batch_size": 16.0
    }
}

with models.DAG(
    dag_id="maxtext_validation_master_dag",
    schedule="0 0 * * *",
    tags=["maxtext", "master", "nightly"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
) as dag:

    # trigger Sub-DAG A (fast fail tensor shape matching)
    trigger_checkpoint_shape_validation = TriggerDagRunOperator(
        task_id="trigger_checkpoint_shape_validation",
        trigger_dag_id="dag_verify_checkpoint_shape",
        conf=GEMMA_NIGHTLY_CONFIG,
        wait_for_completion=True, # Master DAG halts here until this turns green
    )

    #execution order: Shape Check -> tensor match check --> forward logits pass --> sft & decoding
    trigger_checkpoint_shape_validation
