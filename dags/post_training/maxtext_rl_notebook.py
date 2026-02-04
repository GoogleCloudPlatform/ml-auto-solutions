"""
Airflow DAG for automating Llama3.1-8B RL training from Jupyter notebooks.

This DAG automates the rl_llama3_demo.ipynb notebook, executing GRPO/GSPO
training on single-host TPU VMs.
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.post_training.util import notebook_util, test_config_util


SCHEDULE = "0 21 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_rl_notebook"

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2026, 1, 9),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "maxtext",
        "post-training",
        "rl",
        "grpo",
        "gspo",
        "notebook",
        "TPU",
        "v6e-8",
        "nightly",
    ],
    description="Automated Llama3.1-8B RL from Jupyter notebooks.",
    doc_md="""
      # Llama3.1-8B RL Training (Notebook Automation)

      ### Overview
      This DAG automates the rl_llama3_demo.ipynb notebook, which demonstrates
      reinforcement learning training on Llama3.1-8B-Instruct using GRPO
      (Group Relative Policy Optimization) or GSPO (Group Sequence Policy
      Optimization) algorithms.

      ### Prerequisites
      - MaxText checkpoint for Llama3.1-8B-Instruct model
      - HuggingFace access token with read permissions
      - Single-host TPU VM (v6e-8)

      ### Execution Flow
      1. **TPU Creation:** Create TPU VM with required specifications
      2. **Environment Setup:** Clone MaxText, install dependencies
      3. **RL Training:** Execute RL (GRPO/GSPO) training with reward model
      4. **Log Validation:** Verify training completion signals
      5. **Cleanup:** Delete TPU resources

      ### Success Criteria
      The test passes when:
      1. TPU VM is created successfully
      2. Training completes without errors
      3. "RL Training Completed Successfully" appears in logs
      4. Checkpoints are saved to output directory
    """,
    concurrency=1,
) as dag:
  # HF token retrieved from Airflow Variables
  HF_TOKEN_LLAMA31 = models.Variable.get("HF_TOKEN_CIENET", None)

  loss_algos = [
      test_config_util.LossAlgo.GRPO,
      test_config_util.LossAlgo.GSPO,
  ]
  # Test configuration
  test_run_name = "llama31_rl_notebook"
  current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

  # Setup commands for MaxText environment
  setup_script = notebook_util.build_maxtext_setup_script()

  # Test both GRPO and GSPO algorithms
  for loss_algo in loss_algos:
    rl_notebook_test = notebook_util.initialize_notebook_test(
        test_name=f"{DAG_TEST_NAME}_rl_{loss_algo.value}",
        dag_name=DAG_TEST_NAME,
        notebook_path="src/maxtext/examples/rl_llama3_demo.ipynb",
        set_up_script=setup_script,
        parameters={"LOSS_ALGO": loss_algo.loss_name},
        task_owner=test_owner.DEPP_L,
    )

    notebook_util.run_training(rl_notebook_test, HF_TOKEN_LLAMA31)
