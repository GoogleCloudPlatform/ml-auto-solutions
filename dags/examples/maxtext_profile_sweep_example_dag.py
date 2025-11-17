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
An example DAG to extract profile metrics from pretraining mixtral-8x7b model on v6-256.
Profile extraction can be seamlessly integrated with maxtext_sweep_gke_config + run_with_name_gen_and_quarantine.
"""

import datetime
from airflow import models
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.vm_resource import XpkClusters, DockerImage, Project
from dags.multipod.configs import maxtext_sweep_gke_config
from xlml.apis import metric_config

SCHEDULED_TIME = None
BASE_OUTPUT_PATH = "gs://runner-maxtext-logs"


def dict_to_arg(param_dict):
  cmd = [f"{param}={value}" for param, value in param_dict.items()]
  return " ".join(cmd)


docker_image = {
    "stable": "gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_stack:2025-05-20",
}

# https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/maxtext_trillium_model_configs.py
test_models_tpu = {
    "mixtral_8x7b_dropped": {
        "time_out_in_min": 60,
        "cluster": XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
        "train_command": [
            f"export BASE_OUTPUT_PATH={BASE_OUTPUT_PATH} && "
            "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} model_name=mixtral-8x7b "
            # add profiler config: ensure steps > skip_first_n_steps_for_profiler + profiler_steps
            "steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 "
            + dict_to_arg({
                "per_device_batch_size": 12,
                "ici_fsdp_parallelism": -1,
                "max_target_length": 4096,
                "remat_policy": "custom",
                "decoder_layer_input": "offload",
                "out_proj": "offload",
                "query_proj": "offload",
                "key_proj": "offload",
                "value_proj": "offload",
                "attention": "flash",
                "gcs_metrics": True,
                "use_iota_embed": True,
                "dataset_path": "gs://max-datasets-rogue",
                "dataset_type": "synthetic",
                "reuse_example_batch": 1,
                "enable_checkpointing": False,
                "sa_block_q": 2048,
                "sa_block_q_dkv": 2048,
                "sa_block_q_dq": 2048,
                "megablox": False,
                "sparse_matmul": False,
                "capacity_factor": 1.25,
                "tokenizer_path": "assets/tokenizer.mistral-v1",
            })
        ],
    },
    "mixtral_8x7b_dropless": {
        "time_out_in_min": 60,
        "cluster": XpkClusters.TPU_V6E_256_MLPERF_CLUSTER,
        "base_output_directory": "gs://runner-maxtext-logs",
        "train_command": [
            f"export BASE_OUTPUT_PATH={BASE_OUTPUT_PATH} && "
            "python3 -m MaxText.train MaxText/configs/base.yml base_output_directory=${BASE_OUTPUT_PATH} model_name=mixtral-8x7b "
            # add profiler config: ensure steps > skip_first_n_steps_for_profiler + profiler_steps
            "steps=10 profiler=xplane skip_first_n_steps_for_profiler=5 profiler_steps=3 "
            + dict_to_arg({
                "per_device_batch_size": 12,
                "ici_fsdp_parallelism": -1,
                "max_target_length": 4096,
                "remat_policy": "full",
                "attention": "flash",
                "gcs_metrics": True,
                "use_iota_embed": True,
                "dataset_path": "gs://max-datasets-rogue",
                "dataset_type": "synthetic",
                "reuse_example_batch": 1,
                "enable_checkpointing": False,
                "sa_block_q": 2048,
                "sa_block_q_dkv": 2048,
                "sa_block_q_dq": 2048,
                "megablox": True,
                "sparse_matmul": True,
            })
        ],
    },
}


with models.DAG(
    dag_id="maxtext_profile_sweep_example_dag",
    schedule=SCHEDULED_TIME,
    tags=["multipod_team", "maxtext", "TPU", "v6e-256"],
    start_date=datetime.datetime(2025, 5, 20),
    catchup=False,
    concurrency=2,
) as dag:
  quarantine_task_group = TaskGroup(
      group_id="Quarantine", dag=dag, prefix_group_id=False
  )

  for run_name, test_scripts_details in test_models_tpu.items():
    for image in docker_image.keys():
      # sweep num_slices and other training params
      # generate run_name and tensorboard/profile location for extraction
      num_slices = [1]
      sweep_params = {}
      maxtext_sweep_gke_test = maxtext_sweep_gke_config.get_maxtext_sweep_gke_config(
          test_owner=test_owner.SHUNING_J,
          dataset_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
          composer_project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
          dataset_name=metric_config.DatasetOption.XLML_DATASET,
          cluster=test_scripts_details["cluster"],
          time_out_in_min=test_scripts_details["time_out_in_min"],
          base_output_directory=BASE_OUTPUT_PATH,
          num_slices=num_slices,
          docker_image=docker_image[image],
          run_name_prefix=f"maxtext_{image}_{run_name}",
          base_run_model_cmds=test_scripts_details["train_command"],
          sweep_params=sweep_params,
          enable_profile_config=True,  # add flag to enable profile extraction
      )

      for test in maxtext_sweep_gke_test:
        test.run_with_name_gen_and_quarantine(quarantine_task_group)
