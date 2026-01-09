"""
RL training DAG for Llama3.1 70B model using GRPO and GSPO algorithms.

This DAG runs RL training validation to test the MaxText reinforcement
learning pipeline. The workflow deploys training jobs to GKE clusters
using Pathways, executes the GRPO and GSPO algorithms, and validates successful
completion through comprehensive log monitoring of training signals.
"""

import datetime

from airflow import models
from airflow.models.baseoperator import chain
from airflow.utils.task_group import TaskGroup

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
        "gspo",
        "TPU",
        "v5p-128",
        "nightly",
    ],
    description="RL training (GRPO/GSPO) for MaxText RL pipeline validation.",
    doc_md="""
      # RL Training (GRPO/GSPO) MaxText RL Training

      ### Overview
      This DAG runs RL training using GRPO (Group Relative Policy Optimization) 
      and GSPO algorithms to validate the MaxText reinforcement learning pipeline. 
      The workflow tests the complete RL training stack including infrastructure setup,
      model initialization, training execution, and result validation.

      ### Execution Flow
      1. **Job Launch:** Deploy RL training jobs to GKE cluster using Pathways infrastructure
      2. **Model Loading:** Initialize Llama3.1 70B model with HuggingFace authentication
      3. **Training Run:** Execute train_rl with JAX proxy/CPU platforms for GRPO and GSPO
      4. **Log Validation:** Monitor and check for "Post RL Training" completion signal
      5. **Success/Failure:** Report final status based on log validation and job completion

      ### Success Criteria
      The test passes when:
      1. Training jobs complete successfully without errors
      2. "Post RL Training" log message appears in jax-tpu container logs
      3. No infrastructure failures or container launch issues occur
      4. All training steps execute within expected parameters
    """,
    concurrency=2,
) as dag:
  training_config = test_config_util.RLTestConfig(
      cluster=XpkClusters.TPU_V5P_128_CLUSTER,
      accelerator="v5p-128",
      slices=[1],  # Single slice for RL training
      model_name="llama3.1-70b",
      base_dir=(
          f"{test_config_util.DEFAULT_BUCKET}/llama3.1-70b-Instruct/outputs"
      ),
      tokenizer_path="meta-llama/Llama-3.1-70B-Instruct",
      load_parameters_path=(
          f"{test_config_util.DEFAULT_BUCKET}/llama3.1-70b-Instruct/"
          "scanned-pathways/0/items/"
      ),
      rl_config_path="src/MaxText/configs/rl.yml",
      loss_algos=[
          test_config_util.LossAlgo.GRPO,
          test_config_util.LossAlgo.GSPO,
      ],
  )
  # HF token retrieved from Airflow Variables for secure credential management
  HF_TOKEN_LLAMA3_1 = models.Variable.get("HF_TOKEN_LLAMA3_1", None)

  for mode, image in test_config_util.POST_TRAINING_DOCKER_IMAGES:
    # TODO: Enable stable mode once a new version of MaxText is available
    if mode == test_config_util.SetupMode.STABLE:
      continue  # Skip stable for RL training tests

    for loss_algo in training_config.loss_algos:
      for slice_num in training_config.slices:
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        run_name = f"{loss_algo.value}-{mode.value}-{current_datetime}"

        rl_training_command = training_config.generate_rl_training_command(
            loss_algo=loss_algo,
            run_name=run_name,
            hf_token=HF_TOKEN_LLAMA3_1,
        )

        with TaskGroup(
            group_id=f"{loss_algo.value}-{mode.value}-{slice_num}x{training_config.accelerator}"
        ) as group:
          with TaskGroup(group_id="run_training") as training_group:
            start_time = validation_util.generate_timestamp.override(
                task_id="generate_start_time"
            )()

            training_task = gke_config.get_gke_config(
                num_slices=slice_num,
                cluster=training_config.cluster,
                time_out_in_min=30,
                test_name=loss_algo.value,
                run_model_cmds=rl_training_command,
                docker_image=image.value,
                test_owner=test_owner.JACKY_F,
            ).run_model(
                use_pathways=True,
                xpk_branch=MAIN_BRANCH,
            )

            end_time = validation_util.generate_timestamp.override(
                task_id="generate_end_time"
            )()

            chain(
                start_time,
                training_task,
                end_time,
            )

          with TaskGroup(group_id="validate_training") as validation_group:
            validate_loss_algo = validation_util.validate_log_exist.override(
                task_id="validate_loss_algo"
            )(
                project_id=training_config.cluster.project,
                location=zone_to_region(training_config.cluster.zone),
                cluster_name=training_config.cluster.name,
                text_filter=f'"Config param loss_algo: {loss_algo.loss_name}"',
                namespace="default",
                container_name="jax-tpu",
                pod_pattern=f"{loss_algo.value}.*",
                start_time=start_time,
                end_time=end_time,
            )

            validate_training_logs = (
                validation_util.validate_log_exist.override(
                    task_id="validate_training_logs"
                )(
                    project_id=training_config.cluster.project,
                    location=zone_to_region(training_config.cluster.zone),
                    cluster_name=training_config.cluster.name,
                    text_filter="Post RL Training",
                    namespace="default",
                    container_name="jax-tpu",
                    pod_pattern=f"{loss_algo.value}.*",
                    start_time=start_time,
                    end_time=end_time,
                )
            )

          chain(
              training_group,
              validation_group,
          )
