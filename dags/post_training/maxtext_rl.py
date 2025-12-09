"""
GRPO (Group Relative Policy Optimization) training DAG for Llama3.1 70B
model.

This DAG runs GRPO training validation to test the MaxText reinforcement
learning pipeline. The workflow deploys training jobs to GKE clusters
using Pathways, executes the GRPO algorithm, and validates successful
completion through comprehensive log monitoring of training signals.
"""

import datetime

from airflow import models

from dags import composer_env
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config
from dags.post_training.util import validation_util, test_config_util
from xlml.utils.xpk import MAIN_BRANCH
from xlml.utils.gke import zone_to_region

SCHEDULE = "0 20 * * *" if composer_env.is_prod_env() else None
DAG_TEST_NAME = "maxtext_rl"

with models.DAG(
    dag_id=DAG_TEST_NAME,
    start_date=datetime.datetime(2025, 9, 21),
    schedule_interval=SCHEDULE,
    catchup=False,
    tags=[
        "maxtext",
        "post-training",
        "rl",
        "grpo",
        "TPU",
        "v5p-128",
        "nightly",
    ],
    description="GRPO training for MaxText RL pipeline validation.",
    doc_md="""
      # GRPO MaxText RL Training

      ### Overview
      This DAG runs GRPO (Group Relative Policy Optimization) training 
      to validate the MaxText reinforcement learning pipeline. The workflow
      tests the complete RL training stack including infrastructure setup,
      model initialization, training execution, and result validation.

      ### Execution Flow
      1. **Job Launch:** Deploy GRPO training job to GKE cluster using Pathways infrastructure
      2. **Model Loading:** Initialize Llama3.1 70B model with HuggingFace authentication
      3. **Training Run:** Execute train_rl with JAX proxy/CPU platforms
      4. **Log Validation:** Monitor and check for "Post GRPO Training" completion signal
      5. **Success/Failure:** Report final status based on log validation and job completion

      ### Success Criteria
      The test passes when:
      1. Training job completes successfully without errors
      2. "Post GRPO Training" log message appears in jax-tpu container logs
      3. No infrastructure failures or container launch issues occur
      4. All training steps execute within expected parameters
    """,
    concurrency=1,
) as dag:
  training_config = test_config_util.RLTestConfig(
      cluster=XpkClusters.TPU_V5P_128_CLUSTER,
      accelerator="v5p-128",
      slices=[1],  # Single slice for RL training
      model_name="llama3.1-70b",
      short_id="max-rl",
      base_dir=(
          f"{test_config_util.DEFAULT_BUCKET}/llama3.1-70b-Instruct/outputs"
      ),
      tokenizer_path="meta-llama/Llama-3.1-70B-Instruct",
      load_parameters_path=(
          f"{test_config_util.DEFAULT_BUCKET}/llama3.1-70b-Instruct/"
          "scanned-pathways/0/items/"
      ),
      rl_config_path="src/MaxText/configs/rl.yml",
  )
  # HF token retrieved from Airflow Variables for secure credential management
  HF_TOKEN_LLAMA3_1 = models.Variable.get("HF_TOKEN_LLAMA3_1", None)

  for mode, image in test_config_util.DOCKER_IMAGES_RL:
    for slice_num in training_config.slices:
      run_name = validation_util.generate_run_name(
          short_id=training_config.short_id,
          checkpointing_type="rl",
          slice_number=slice_num,
          accelerator=training_config.accelerator,
      )

      rl_training_command = (
          f"export HF_TOKEN={HF_TOKEN_LLAMA3_1} && "
          "export TPU_MIN_LOG_LEVEL=0 && "
          "export TF_CPP_MIN_LOG_LEVEL=0 && "
          "export TPU_STDERR_LOG_LEVEL=0 && "
          "export JAX_PLATFORMS=proxy,cpu && "
          "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 && "
          "export ENABLE_PATHWAYS_PERSISTENCE='1' && "
          f"python -m src.MaxText.rl.train_rl "
          f"{training_config.rl_config_path} run_name={run_name} "
          f"model_name={training_config.model_name} "
          f"tokenizer_path={training_config.tokenizer_path} "
          f"load_parameters_path={training_config.load_parameters_path} "
          f"base_output_directory={training_config.base_dir}",
      )

      start_time = validation_util.generate_timestamp()

      grpo_training_task = gke_config.get_gke_config(
          num_slices=slice_num,
          cluster=training_config.cluster,
          time_out_in_min=30,
          test_name=f"{training_config.short_id}",
          run_model_cmds=rl_training_command,
          docker_image=image.value,
          test_owner=test_owner.JACKY_F,
      ).run(
          use_pathways=True,
          xpk_branch=MAIN_BRANCH,
          skip_post_process=True,
      )

      end_time = validation_util.generate_timestamp()

      validate_grpo_training = validation_util.validate_log_exist(
          project_id=training_config.cluster.project,
          location=zone_to_region(training_config.cluster.zone),
          cluster_name=training_config.cluster.name,
          text_filter="Post RL Training",
          namespace="default",
          container_name="jax-tpu",
          pod_pattern=f"{training_config.short_id}.*",
          start_time=start_time,
          end_time=end_time,
      )

      (
          run_name
          >> start_time
          >> grpo_training_task
          >> end_time
          >> validate_grpo_training
      )
