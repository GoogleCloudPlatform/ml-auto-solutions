#  Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""DAG to run MaxText SFT Trainer tests."""

import datetime
from airflow import models
from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, XpkClusters
from dags.multipod.configs import gke_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 10 am UTC (2 am PST)
SCHEDULED_TIME = '0 10 * * *' if composer_env.is_prod_env() else None
HF_TOKEN = models.Variable.get('HF_TOKEN', None)

with models.DAG(
    dag_id='maxtext_sft_trainer',
    schedule=SCHEDULED_TIME,
    tags=['multipod_team', 'maxtext', 'stable', 'nightly', 'mlscale_devx'],
    start_date=datetime.datetime(2025, 3, 1),
    catchup=False,
    concurrency=2,
) as dag:
  base_output_directory = f'{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_sft_trainer'
  docker_images = [
      (SetupMode.STABLE, DockerImage.MAXTEXT_TPU_JAX_STABLE_STACK_CANDIDATE),
      (SetupMode.NIGHTLY, DockerImage.MAXTEXT_TPU_STABLE_STACK_NIGHTLY_JAX),
  ]

  for mode, image in docker_images:
    command = (
        f'export HF_TOKEN={HF_TOKEN}',
        'export PRE_TRAINED_MODEL=llama2-7b',
        'export PRE_TRAINED_MODEL_TOKENIZER=meta-llama/Llama-2-7b-chat-hf',
        'export PRE_TRAINED_MODEL_CKPT_PATH=gs://maxtext-model-checkpoints/llama2-7b-chat/scanned/0/items',
        f'export BASE_OUTPUT_DIRECTORY={base_output_directory}',
        'export STEPS=1000',
        'export PROMPT="Suggest some famous landmarks in London."',
        'bash end_to_end/tpu/test_sft_trainer.sh',
    )
    maxtext_v4_configs_test = gke_config.get_gke_config(
        cluster=XpkClusters.TPU_V5P_8_CLUSTER,
        time_out_in_min=60,
        test_name=f'sft-trainer-{mode.value}',
        run_model_cmds=command,
        docker_image=image.value,
        test_owner=test_owner.SURBHI_J,
    ).run()
