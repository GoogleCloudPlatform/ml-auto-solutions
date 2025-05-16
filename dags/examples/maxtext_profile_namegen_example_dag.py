# Copyright 2024 Google LLC
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

"""A DAG to run end-to-end MoE tests."""


import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags import composer_env
from dags.common.quarantined_tests import QuarantineTests
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config
from xlml.utils import name_format
from xlml.apis import metric_config
import os


# Run once a day at 1 am UTC (5 pm PST)
SCHEDULED_TIME = "0 1 * * *" if composer_env.is_prod_env() else None


docker_image = {
    "stable": "gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_stack:2025-05-09",
}

test_models_tpu = {
    "mixtral-8x7b_pretraining-megablox_config-true_upload-one": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "base_output_directory": "gs://runner-maxtext-logs",
        "train_command": [
            "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} run_name=${RUN_NAME} model_name=mixtral-8x7b tokenizer_path=assets/tokenizer.mistral-v1 dataset_path=gs://maxtext-dataset per_device_batch_size=4 enable_checkpointing=false ici_fsdp_parallelism=-1 max_target_length=1024 async_checkpointing=false attention=flash dtype=bfloat16 weight_dtype=bfloat16"
            " steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3",
        ],
    },
    "mixtral-8x7b_pretraining-megablox_config-false": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "base_output_directory": "gs://runner-maxtext-logs",
        "train_command": [
            "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} run_name=${RUN_NAME} model_name=mixtral-8x7b tokenizer_path=assets/tokenizer.mistral-v1 dataset_path=gs://maxtext-dataset per_device_batch_size=4 enable_checkpointing=false ici_fsdp_parallelism=-1 max_target_length=1024 async_checkpointing=false attention=flash dtype=bfloat16 weight_dtype=bfloat16"
            " steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3",
        ],
    },
    "mixtral-8x7b_pretraining-megablox_config-true_upload-all": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "base_output_directory": "gs://runner-maxtext-logs",
        "train_command": [
            "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} run_name=${RUN_NAME} model_name=mixtral-8x7b tokenizer_path=assets/tokenizer.mistral-v1 dataset_path=gs://maxtext-dataset per_device_batch_size=4 enable_checkpointing=false ici_fsdp_parallelism=-1 max_target_length=1024 async_checkpointing=false attention=flash dtype=bfloat16 weight_dtype=bfloat16"
            " steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 upload_all_profiler_results=True",
        ],
    },
    "mixtral-8x7b_pretraining-megablox_config-true_upload-none": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "base_output_directory": "gs://runner-maxtext-logs",
        "train_command": [
            "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} run_name=${RUN_NAME} model_name=mixtral-8x7b tokenizer_path=assets/tokenizer.mistral-v1 dataset_path=gs://maxtext-dataset per_device_batch_size=4 enable_checkpointing=false ici_fsdp_parallelism=-1 max_target_length=1024 async_checkpointing=false attention=flash dtype=bfloat16 weight_dtype=bfloat16"
            " steps=10",
        ],
    },
}


with models.DAG(
    dag_id="maxtext_profile_namegen_example_dag",
    schedule=SCHEDULED_TIME,
    tags=[
        "sparsity_diffusion_devx",
        "multipod_team",
        "maxtext",
        "tpu",
        "stable",
        "nightly",
        "mlscale_devx",
    ],
    start_date=datetime.datetime(2024, 11, 14),
    catchup=False,
) as dag:
  tests = []
  for run_name, test_scripts_details in test_models_tpu.items():
    for image in docker_image.keys():
      base_output_directory = test_scripts_details["base_output_directory"]
      test_scripts_details["train_command"] = [
          f"export BASE_OUTPUT_PATH={base_output_directory}"
      ] + test_scripts_details["train_command"]

      # file_location: pass in base_output_directory, will be altered in `run_with_run_name_generation`
      job_metric_config = metric_config.MetricConfig()
      job_metric_config.tensorboard_summary = metric_config.SummaryConfig(
          file_location=base_output_directory,  # init location
          aggregation_strategy=metric_config.AggregationStrategy.MEDIAN,
          use_regex_file_location=True,
      )
      # turn off for config-false testing
      if run_name != "mixtral-8x7b_pretraining-megablox_config-false":
        job_metric_config.profile = metric_config.ProfileConfig(
            file_location=base_output_directory,  # init location
        )

      tpu_task = gke_config.get_gke_config(
          time_out_in_min=test_scripts_details["time_out_in_min"],
          test_name=f"maxtext_{image}_{run_name}",
          run_model_cmds=test_scripts_details["train_command"],
          docker_image=docker_image[image],
          test_owner=test_owner.RAN_R,
          cluster=test_scripts_details["cluster"],
          user_specified_job_metric_config=job_metric_config,
      ).run_with_run_name_generation(run_name_env="RUN_NAME")

      tests.append(tpu_task)

  for test in tests:
    test
  # for i in range(len(tests) - 1):
  #   tests[i] >> tests[i + 1]
