# Copyright 2025 Google LLC
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

"""
An example DAG to extract profile metrics from pretraining mixtral-8x7b model on 1xv4-128.
Profile extraction can be easily integrated with gke_config + run_with_run_name_generation.
"""

import datetime
from airflow import models
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage
from dags.multipod.configs import gke_config
from xlml.apis import metric_config

SCHEDULED_TIME = None
BASE_OUTPUT_PATH = "gs://runner-maxtext-logs"

docker_image = {
    "stable": "gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_stack:2025-05-20",
}

base_command = (
    f"export BASE_OUTPUT_PATH={BASE_OUTPUT_PATH} && "
    + "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=gs://runner-maxtext-logs run_name=${RUN_NAME} model_name=mixtral-8x7b tokenizer_path=assets/tokenizer.mistral-v1 dataset_path=gs://maxtext-dataset per_device_batch_size=4 enable_checkpointing=false ici_fsdp_parallelism=-1 max_target_length=1024 async_checkpointing=false attention=flash dtype=bfloat16 weight_dtype=bfloat16"
)

test_models_tpu = {
    # use: upload single profile from the first host, extract profile
    # add profiler config: ensure steps > skip_first_n_steps_for_profiler + profiler_steps
    "mixtral-8x7b_pretraining-megablox_config-true_upload-one": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "train_command": [
            base_command
            + " steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3",
        ],
    },
    # use: upload profiles from all hosts, extract one of the profiles
    # add profiler config: ensure steps > skip_first_n_steps_for_profiler + profiler_steps
    "mixtral-8x7b_pretraining-megablox_config-true_upload-all": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "train_command": [
            base_command
            + " steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 upload_all_profiler_results=True",
        ],
    },
    # testing: handle edge case, attempt to extract, find no match, proceed to post_process without error
    "testing_config-true_upload-none": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "train_command": [
            base_command + " steps=10",
        ],
    },
    # testing: not generate profile location, not extract profile in post_process
    "testing_config-false_upload-one": {
        "cluster": XpkClusters.TPU_V4_128_CLUSTER,
        "time_out_in_min": 60,
        "train_command": [
            base_command
            + " steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3",
        ],
        "not_add_profile_config": True,
    },
}


with models.DAG(
    dag_id="maxtext_profile_namegen_example_dag",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext"],
    start_date=datetime.datetime(2025, 5, 20),
    catchup=False,
    concurrency=2,
) as dag:
  for run_name, test_scripts_details in test_models_tpu.items():
    for image in docker_image.keys():
      # file_location: pass in base_output_directory, will be altered in `run_with_run_name_generation`
      job_metric_config = metric_config.MetricConfig()
      # optionally, add tensorboard metrics
      job_metric_config.tensorboard_summary = metric_config.SummaryConfig(
          file_location=BASE_OUTPUT_PATH,  # init location
          aggregation_strategy=metric_config.AggregationStrategy.MEDIAN,
          use_regex_file_location=True,
      )
      # enable profile extraction
      job_metric_config.profile = metric_config.ProfileConfig(
          file_location=BASE_OUTPUT_PATH,  # init location
      )

      # testing: config-false
      if "not_add_profile_config" in test_scripts_details:
        job_metric_config.profile = None

      tpu_task = gke_config.get_gke_config(
          num_slices=1,
          time_out_in_min=test_scripts_details["time_out_in_min"],
          test_name=f"maxtext_{image}_{run_name}",
          run_model_cmds=test_scripts_details["train_command"],
          docker_image=docker_image[image],
          test_owner=test_owner.SHUNING_J,
          cluster=test_scripts_details["cluster"],
          user_specified_job_metric_config=job_metric_config,  # customize config
      ).run_with_run_name_generation(run_name_env="RUN_NAME")
