# Copyright 2026 Google LLC
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
A DAG to run MaxText E2E TPU Post-Training tests.
"""
import datetime
import hashlib

from click import command
from airflow import models
from airflow.models.param import Param
from airflow.utils.task_group import TaskGroup
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import XpkClusters
from dags.multipod.configs import gke_config

# HF token retrieved from Airflow Variables for secure credential management
HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)


def get_workload_name(model, mode, length=6):
  hex_code = f"{mode}-{hashlib.sha256(model.encode()).hexdigest()}"
  return hex_code[:length]


with models.DAG(
    dag_id="maxtext_e2e_tpu_post_training",
    schedule=None,
    tags=[
        "maxtext",
        "post-training",
        "TPU",
    ],
    start_date=datetime.datetime(2026, 6, 10),
    catchup=False,
    params={
        "docker_image": Param(
            type="string",
            description="Docker image URI for the candidate to test",
        ),
    },
) as dag:
  # pylint: disable=line-too-long
  test_models = {
      "gemma3-4b": {
          "checkpoint_conversion": {
              "to_maxtext": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_mt.sh",
              "to_huggingface": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_to_hf.sh",
          },
          "post_training": {
              "sft": {
                  "command": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_sft.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma3-4b/sft/{run_name}/checkpoints/5/model_params",
              },
              "multimodal_sft": {
                  "command": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_multimodal_sft.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma3-4b/multimodal/sft/{run_name}/checkpoints/4/items",
                  "to_hf_flags": "true true",
              },
              "lora": {
                  "command": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_lora.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma3-4b/lora/{run_name}/checkpoints/5/model_params",
              },
              "rl": {
                  "command": "bash tests/end_to_end/tpu/gemma3/4b/test_gemma3_rl.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma3-4b/rl/{run_name}/checkpoints/actor/5/model_params",
              },
          },
      },
      "gemma4-26b": {
          "checkpoint_conversion": {
              "to_maxtext": "bash tests/end_to_end/tpu/gemma4/26b/test_gemma4_to_mt.sh",
              "to_huggingface": "bash tests/end_to_end/tpu/gemma4/26b/test_gemma4_to_hf.sh",
          },
          "post_training": {
              "sft": {
                  "command": "bash tests/end_to_end/tpu/gemma4/26b/test_gemma4_sft.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma4-26b/sft/{run_name}/checkpoints/5/model_params",
                  "to_hf_flags": "false false",
              },
              "rl": {
                  "command": "bash tests/end_to_end/tpu/gemma4/26b/test_gemma4_rl.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/gemma4-26b/rl/{run_name}/checkpoints/actor/5/model_params",
                  "to_hf_flags": "false false",
              },
          },
      },
      "llama3_1_70b": {
          "checkpoint_conversion": {
              "to_maxtext": "bash tests/end_to_end/tpu/llama3.1/70b/test_llama3.1_70b_to_mt.sh",
              "to_huggingface": "bash tests/end_to_end/tpu/llama3.1/70b/test_llama3.1_70b_to_hf.sh",
          },
          "post_training": {
              "sft": {
                  "command": "bash tests/end_to_end/tpu/llama3.1/70b/test_llama3.1_70b_sft.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/llama3.1-70b/sft/{run_name}/checkpoints/5/model_params",
              },
              "rl": {
                  "command": "bash tests/end_to_end/tpu/llama3.1/70b/test_llama3.1_70b_rl.sh",
                  "maxtext_ckpt_path": "gs://runner-maxtext-logs/llama3.1-70b/rl/{run_name}/checkpoints/actor/5/model_params",
              },
          },
      },
  }
  # pylint: enable=line-too-long

  for model, test_config in test_models.items():
    with TaskGroup(group_id=model) as model_group:
      run_name = "post-{{ ts_nodash }}"

      convert_to_maxtext_cmd = (
          f"export HF_TOKEN={HF_TOKEN}",
          'export HF_HOME="/dev/shm/hf_cache"',
          'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
      ) + (f"{test_config['checkpoint_conversion']['to_maxtext']} {run_name}",)
      convert_to_maxtext_task = gke_config.get_gke_config(
          time_out_in_min=60,
          test_name="convert-to-maxtext",
          run_model_cmds=convert_to_maxtext_cmd,
          docker_image="{{ params.docker_image }}",
          cluster=XpkClusters.TPU_V5P_8_CLUSTER_V2,
          test_owner=test_owner.SURBHI_J,
      ).run(skip_post_process=True, priority="very-high")

      for mode, mode_test_config in test_config["post_training"].items():
        with TaskGroup(group_id=f"{mode}-{model}") as model_group:
          environment_variables = [
              f"export HF_TOKEN={HF_TOKEN}",
              "export TPU_MIN_LOG_LEVEL=0",
              "export TF_CPP_MIN_LOG_LEVEL=0",
              "export TPU_STDERR_LOG_LEVEL=0",
              "export JAX_PLATFORMS=proxy,cpu",
              "export JAX_BACKEND_TARGET=grpc://127.0.0.1:29000",
              "export ENABLE_PATHWAYS_PERSISTENCE='1'",
          ]

          with TaskGroup(group_id=f"train-{mode}-{model}") as model_group:
            command = mode_test_config["command"]
            training_cmd = (
                " && ".join(
                    environment_variables + [f"{command} {run_name} true"]
                ),
            )
            training_task = gke_config.get_gke_config(
                time_out_in_min=60,
                num_slices=1,
                cluster=XpkClusters.TPU_V5P_128_CLUSTER,
                test_name=get_workload_name(model, mode),
                run_model_cmds=training_cmd,
                docker_image="{{ params.docker_image }}",
                test_owner=test_owner.SURBHI_J,
            ).run_model(
                use_pathways=True,
                priority="very-high",
            )

          to_hf_flags = mode_test_config.get("to_hf_flags", "false true")

          model_path = mode_test_config["maxtext_ckpt_path"].format(
              run_name=run_name
          )
          convert_to_huggingface_cmd = (
              f"export HF_TOKEN={HF_TOKEN}",
              'export HF_HOME="/dev/shm/hf_cache"',
              'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
          ) + (
              f"{test_config['checkpoint_conversion']['to_huggingface']} "
              f"{run_name} {model_path} {to_hf_flags}",
          )
          convert_to_huggingface_task = gke_config.get_gke_config(
              time_out_in_min=60,
              test_name="convert-to-huggingface",
              run_model_cmds=convert_to_huggingface_cmd,
              docker_image="{{ params.docker_image }}",
              cluster=XpkClusters.TPU_V5P_8_CLUSTER_V2,
              test_owner=test_owner.SURBHI_J,
          ).run(skip_post_process=True, priority="very-high")

          (
              convert_to_maxtext_task
              >> training_task
              >> convert_to_huggingface_task
          )
