"""
Airflow DAG for automating Llama3.1-8B RL training from Jupyter notebook.

This DAG automates the rl_llama3_demo.ipynb notebook, executing GRPO/GSPO
training on single-host TPU VMs.
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import (
    Project,
    RuntimeVersion,
    TpuVersion,
    V6E_GCE_NETWORK,
    V6E_GCE_SUBNETWORK,
    Zone,
)
from dags.post_training.util import notebook_util, test_config_util
from xlml.apis import gcp_config, metric_config, task, test_config

SCHEDULE = "0 21 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_rl_notebook"
DEFAULT_BUCKET = "gs://rl-automation"

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
    description="Automated Llama3.1-8B RL training from Jupyter notebook.",
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
      - Single-host TPU VM (v6e-8 or v5p-8)

      ### Execution Flow
      1. **TPU Creation:** Create TPU VM with required specifications
      2. **Environment Setup:** Clone MaxText, install dependencies
      3. **RL Training:** Execute GRPO/GSPO training with reward model
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
  # Test configuration
  notebook_config = test_config_util.RLTestConfig(
      cluster=None,  # Not used for TPU VM tests
      accelerator="v6e-8",
      slices=[1],
      model_name="llama3.1-8b",
      base_dir=f"{DEFAULT_BUCKET}/llama3.1-8b-Instruct/outputs",
      tokenizer_path="meta-llama/Llama-3.1-8B-Instruct",
      load_parameters_path=(
          f"{DEFAULT_BUCKET}/llama3.1-8b-Instruct/scanned-pathways/0/items"
      ),
      loss_algos=[
          test_config_util.LossAlgo.GRPO,
          test_config_util.LossAlgo.GSPO,
      ],
  )

  # HF token retrieved from Airflow Variables
  HF_TOKEN_LLAMA31 = models.Variable.get("HF_TOKEN_CIENET", None)

  # Test configuration
  test_run_name = "llama31_rl_notebook"
  current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

  # Setup commands for MaxText environment
  setup_script = notebook_util.build_maxtext_setup_script()

  # Path to the RL demo notebook
  notebook_path = "src/MaxText/examples/rl_llama3_demo.ipynb"

  # Test both GRPO and GSPO algorithms
  for loss_algo in notebook_config.loss_algos:
    run_name = f"{loss_algo.value}-{current_datetime}"

    # Parameters to inject into notebook
    notebook_params = {
        "MODEL_CHECKPOINT_PATH": notebook_config.load_parameters_path,
        "OUTPUT_DIRECTORY": notebook_config.base_dir,
        "LOSS_ALGO": loss_algo.loss_name,
    }

    # Build notebook execution command
    notebook_execution = notebook_util.build_notebook_execution_command(
        notebook_path=notebook_path,
        parameters=notebook_params,
        maxtext_path="maxtext",
        venv_path="maxtext_venv",
        env_params={"HF_TOKEN": HF_TOKEN_LLAMA31},
    )

    # Create TPU VM test configuration
    rl_notebook_test = test_config.TpuVmTest(
        test_config.Tpu(
            version=TpuVersion.TRILLIUM,
            cores=8,
            runtime_version=RuntimeVersion.V2_ALPHA_TPUV6.value,
            reserved=False,
            network=V6E_GCE_NETWORK,
            subnetwork=V6E_GCE_SUBNETWORK,
        ),
        test_name=f"{DAG_TEST_NAME}_{loss_algo.value}",
        set_up_cmds=[setup_script],
        run_model_cmds=[notebook_execution],
        timeout=datetime.timedelta(minutes=180),
        task_owner=test_owner.JACKY_F,
        num_slices=1,
        gcs_subfolder=f"{DEFAULT_BUCKET}/{DAG_TEST_NAME}",
    )

    # Run the training task
    training_task = task.run_queued_resource_test(
        task_test_config=rl_notebook_test,
        task_gcp_config=gcp_config.GCPConfig(
            project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
            zone=Zone.EUROPE_WEST4_A.value,
            dataset_name=metric_config.DatasetOption.XLML_DATASET,
        ),
        skip_post_process=True,
    )
