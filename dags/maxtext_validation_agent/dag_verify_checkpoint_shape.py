"""DAG to automate MaxText Checkpoint Structural Shape Validation."""

import datetime
from airflow import models
from dags.maxtext_validation_agent.lib import utils


DEFAULT_PARAMS = {
  "run_name": "qwen3-custom-shape-test",
  "checkpoint_gcs_path": "gs://maxtext-model-checkpoints/qwen3-8b/unscanned/0/items",
  "maxtext_model_name": "qwen3-8b",
  "maxtext_overrides": {
    "tokenizer_path": "Qwen/Qwen3-8B-Instruct",
    "tokenizer_type": "huggingface",
    "scan_layers": False,
    "max_target_length": 2048,
    "per_device_batch_size": 8.0,
    "attention": "dot_product"
  }
}

with models.DAG(
    dag_id="dag_verify_checkpoint_shape",
    schedule=None,
    tags=["maxtext", "checkpoint", "validation"],
    start_date=datetime.datetime(2026, 6, 26),
    catchup=False,
    params=DEFAULT_PARAMS,
) as dag:

    # Looks for keys in runtime conf first (from manual JSON or Master DAG),
    # falls back to defaults if run standalone.
    checkpoint_task = utils.get_checkpoint_shape_validation_task(
        dag=dag,
        model_name="{{ dag_run.conf.get('maxtext_model_name', params['maxtext_model_name']) }}",
        checkpoint_gcs_path="{{ dag_run.conf.get('checkpoint_gcs_path', params['checkpoint_gcs_path']) }}",
        scan_layers="{{ dag_run.conf.get('maxtext_overrides', params['maxtext_overrides']).get('scan_layers', False) | lower }}"
    )

    mock_tensor_task = utils.get_mock_tensor_validation_task(
        dag=dag,
        model_name="{{ dag_run.conf.get('maxtext_model_name', params['maxtext_model_name']) }}",
        checkpoint_path="{{ dag_run.conf.get('checkpoint_gcs_path', params['checkpoint_gcs_path']) }}"
    )

    # Execution order: Shape check MUST pass before attempting the tensor dry-run
    checkpoint_task >> mock_tensor_task
