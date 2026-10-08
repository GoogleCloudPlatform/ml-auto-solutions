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

"""DAG to run MaxText Knowledge Distillation tests with Cluster Toolkit."""

import datetime

from airflow import models

from dags import composer_env, gcs_bucket
from dags.common import test_owner
from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import DockerImage, GkeClusters
from dags.multipod.configs import gke_config

DAG_ID = "maxtext_distillation"
# Run once a day at 11 am UTC (3 am PST)
SCHEDULED_TIME = "0 11 * * *" if composer_env.is_prod_env() else None
HF_TOKEN = safe_get_from_variable("HF_TOKEN", None)

with models.DAG(
    dag_id=DAG_ID,
    schedule=SCHEDULED_TIME,
    tags=[
        "multipod_team",
        "maxtext",
        "post-training",
        "distillation",
        "nightly",
        "mlscale_devx",
        "TPU",
        "v5p-32",
    ],
    start_date=datetime.datetime(2026, 4, 1),
    catchup=False,
    concurrency=1,
) as dag:
  base_output_directory = f"{gcs_bucket.BASE_OUTPUT_DIR}/maxtext_distillation"
  teacher_ckpt_path = (
      "gs://maxtext-model-checkpoints/llama3.1-8b/scanned/20261006/0/items"
  )
  run_name = "distill-nightly-{{ ts_nodash | lower }}"

  command = (
      f"export HF_TOKEN={HF_TOKEN}",
      "export PYTHONPATH=/deps/src:/app/src",
      'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=61440"',
      "export TMPDIR=/dev/shm",
      "export JAX_COMPILATION_CACHE_DIR=/dev/shm/jax_cache",
      "export HF_HOME=/dev/shm/hf",
      "export HF_DATASETS_CACHE=/dev/shm/hf/datasets",
      "mkdir -p /dev/shm/hf/datasets",
      (
          "python3 -m maxtext.trainers.post_train.distillation.train_distill"
          " src/maxtext/configs/post_train/distillation.yml"
          f" run_name={run_name}"
          f" base_output_directory={base_output_directory}"
          f" teacher_overrides.load_parameters_path={teacher_ckpt_path}"
          f" hf_access_token={HF_TOKEN}"
          " distill_alpha=0.5"
          " distill_temperature=1.0"
          " distill_beta=1.0"
          " 'distill_layer_indices=[0,1,2,3,4,5,6,7]'"
          " steps=2"
          " learning_rate_schedule_steps=2"
          " enable_goodput_recording=False"
          " monitor_goodput=False"
          " save_checkpoint_on_completion=True"
      ),
  )

  distill_nightly = gke_config.get_gke_config(
      time_out_in_min=60,
      num_slices=1,
      cluster=GkeClusters.TPU_V5P_BODABORG_NAP_CLUSTER.override(core_count=32),
      test_name="distill-nightly",
      run_model_cmds=command,
      docker_image=DockerImage.MAXTEXT_POST_TRAINING_NIGHTLY.value,
      test_owner=test_owner.EMMA_L,
      priority="medium",
      use_gcluster=True,
  ).run()
