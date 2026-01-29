"""
Airflow DAG for automating Llama3.1-8B SFT training from Jupyter notebook.

This DAG automates the sft_llama3_demo.ipynb notebook, executing SFT
training on single-host TPU VMs.
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.post_training.util import notebook_util


SCHEDULE = "0 23 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_sft_notebook"

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2026, 1, 9),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "maxtext",
        "post-training",
        "sft",
        "notebook",
        "TPU",
        "v6e-8",
        "nightly",
    ],
    description="Automated Llama3.1-8B SFT training from Jupyter notebook.",
    doc_md="""
      # Llama3.1-8B SFT Training (Notebook Automation)

      ### Overview
      This DAG automates the `sft_llama3_demo.ipynb` notebook, which
      demonstrates Supervised Fine-Tuning (SFT) on Llama3.1-8B-Instruct.
      It executes SFT training on single-host TPU VMs.

      ### Prerequisites
      - MaxText checkpoint for Llama3.1-8B-Instruct model
      - HuggingFace access token with read permissions
      - Single-host TPU VM (v6e-8)

      ### Execution Flow
      1. **TPU Creation:** Create TPU VM with required specifications
      2. **Environment Setup:** Clone MaxText, install dependencies
      3. **SFT Training:** Execute SFT training notebook
      4. **Log Validation:** Verify training completion signals
      5. **Cleanup:** Delete TPU resources

      ### Success Criteria
      The test passes when:
      1. TPU VM is created successfully
      2. Training completes without errors
      3. "SFT Training Completed Successfully" appears in logs
      4. Checkpoints are saved to output directory
    """,
    concurrency=1,
) as dag:
  # HF token retrieved from Airflow Variables
  HF_TOKEN_LLAMA31 = models.Variable.get("HF_TOKEN_CIENET", None)

  # Test configuration
  test_run_name = "llama31_rl_notebook"
  current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

  # Setup commands for MaxText environment
  setup_script = notebook_util.build_maxtext_setup_script()

  # Test SFT training
  sft_notebook_test = notebook_util.initialize_notebook_test(
      test_name=f"{DAG_TEST_NAME}_sft",
      dag_name=DAG_TEST_NAME,
      notebook_path="src/MaxText/examples/sft_llama3_demo.ipynb",
      set_up_script=setup_script,
      parameters={},
      task_owner=test_owner.DEPP_L,
  )

  notebook_util.run_training(sft_notebook_test, HF_TOKEN_LLAMA31)
