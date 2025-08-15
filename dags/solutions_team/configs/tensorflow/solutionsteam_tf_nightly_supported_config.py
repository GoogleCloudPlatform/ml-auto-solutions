# Copyright 2023 Google LLC
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

"""Utilities to construct configs for solutionsteam_tf_nightly_supported DAG."""

import time
import datetime
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket
from dags.solutions_team.configs.tensorflow import common
from airflow.models import Variable
from dags.common.vm_resource import TpuVersion, Project, RuntimeVersion
from typing import List


def get_tf_keras_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    test_feature: str,
    test_name: str,
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    runtime_version: str = RuntimeVersion.TPU_VM_TF_NIGHTLY.value,
    is_pod: bool = False,
    is_pjrt: bool = True,
    network: str = "default",
    subnetwork: str = "default",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_keras() + common.install_tf()
  if not is_pjrt and is_pod:
    set_up_cmds += common.set_up_se_nightly()
  keras_test_name = f"tf_keras_api_{test_name}"
  benchmark_id = f"{keras_test_name}-v{tpu_version.value}-{tpu_cores}"
  # Add default_var to pass DAG check
  # TODO(ranran): replace Variable.get() to XCOM when it applies
  tpu_name = Variable.get(benchmark_id, default_var=None) if is_pod else "local"
  env_variable = (
      common.export_env_variables(tpu_name, is_pod, is_pjrt)
      + " TF_CPP_MIN_LOG_LEVEL=0  TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0 "
  )
  skipped_tag = "--tags=-failspod" if is_pod else ""
  run_model_cmds = (
      "sudo chmod -R 777 /tmp/",
      (
          "export PATH=$PATH:/home/ml-auto-solutions/.local/bin &&"
          f" cd /tmp/tf2-api-tests && {env_variable} &&"
          " behave -e ipynb_checkpoints"
          f" --tags=-fails {skipped_tag} -i {test_feature}"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=keras_test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.AIRFLOW,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      tpu_name_env_var=is_pod,
      all_workers=False,
  )


def get_tf_resnet_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    runtime_version: str = RuntimeVersion.TPU_VM_TF_NIGHTLY.value,
    network: str = "default",
    subnetwork: str = "default",
    is_pod: bool = False,
    is_pjrt: bool = True,
    imagenet_dir: str = gcs_bucket.IMAGENET_DIR,
    tfds_data_dir: str = gcs_bucket.TFDS_DATA_DIR,
    global_batch_size: int = 2048,
    train_steps: int = 320,
    validation_interval: int = 320,
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = common.set_up_tensorflow_models() + common.install_tf()
  if is_pod:
    if not is_pjrt:
      set_up_cmds += common.set_up_se()
    else:
      set_up_cmds += common.set_up_pjrt()

  # adjust global batch size based on num_cores
  global_batch_size = 128 * tpu_cores
  params_override = {
      "runtime": {"distribution_strategy": "tpu"},
      "task": {
          "train_data": {
              "input_path": imagenet_dir + "/train*",
              "tfds_data_dir": tfds_data_dir,
              "global_batch_size": global_batch_size,
          },
          "validation_data": {
              "input_path": imagenet_dir + "/valid*",
              "tfds_data_dir": tfds_data_dir,
              "global_batch_size": global_batch_size,
          },
      },
      "trainer": {
          "train_steps": train_steps,
          "validation_interval": validation_interval,
      },
  }

  test_name = "tf_resnet_imagenet"
  benchmark_id = f"{test_name}-v{tpu_version.value}-{tpu_cores}"
  # Add default_var to pass DAG check
  # TODO(ranran): replace Variable.get() to XCOM when it applies
  tpu_name = Variable.get(benchmark_id, default_var=None) if is_pod else "local"
  env_variable = common.export_env_variables(tpu_name, is_pod, is_pjrt)

  epoch = time.time()
  model_dir = f"{gcs_bucket.BASE_OUTPUT_DIR}/{test_owner.Team.SOLUTIONS_TEAM.value}/resnet/{benchmark_id}/{epoch}"
  run_model_cmds = (
      "sudo chmod -R 777 /tmp/",
      (
          f"cd /usr/share/tpu/models && {env_variable} &&"
          " python3 official/vision/train.py"
          f" --experiment=resnet_imagenet"
          f" --mode=train_and_eval --model_dir={model_dir}"
          f" --params_override='{params_override}'"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.CHANDRA_D,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      tpu_name_env_var=is_pod,
      all_workers=False,
  )


def get_tf_dlrm_v1_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    bottom_mlp: List[int],
    embedding_dim: int,
    train_steps: int,
    extraFlags: str = "",
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    runtime_version: str = RuntimeVersion.TPU_VM_TF_NIGHTLY.value,
    is_pod: bool = False,
    is_pjrt: bool = True,
    criteo_dir: str = gcs_bucket.CRITEO_DIR,
    network: str = "default",
    subnetwork: str = "default",
    global_batch_size=16384,
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  # Add default_var to pass DAG check
  # TODO(ranran): replace Variable.get() to XCOM when it applies
  test_name = "tf_dlrm_criteo"
  benchmark_id = f"{test_name}-v{tpu_version.value}-{tpu_cores}"
  tpu_name = Variable.get(benchmark_id, default_var=None) if is_pod else "local"
  is_v5p = tpu_version == TpuVersion.V5P
  env_variable = (
      common.export_env_variables(tpu_name, is_pod, is_pjrt, is_v5p_sc=is_v5p)
      + " TF_CPP_MIN_LOG_LEVEL=0  TPU_STDERR_LOG_LEVEL=0 TPU_MIN_LOG_LEVEL=0  "
  )

  set_up_cmds = common.set_up_tensorflow_models() + common.install_tf()
  if is_pod:
    if not is_pjrt:
      set_up_cmds += common.set_up_se()
    else:
      set_up_cmds += common.set_up_pjrt()
  set_up_cmds += (
      "sudo apt-get -y update",
      "sudo apt-get install numactl",
  )
  params_override = {
      "runtime": {"distribution_strategy": "tpu"},
      "task": {
          "use_synthetic_data": False,
          "train_data": {
              "input_path": "gs://agagik-dev/data/criteo/terabyte_processed_golden_shuffled/train/*",
              "global_batch_size": global_batch_size,
          },
          "validation_data": {
              "input_path": "gs://agagik-dev/data/criteo/terabyte_processed_golden_shuffled/eval/*",
              "global_batch_size": global_batch_size,
          },
          "model": {
              "num_dense_features": 13,
              "bottom_mlp": [512, 256, 16],
              "embedding_dim": 16,
              "top_mlp": [1024, 1024, 512, 256, 1],
              "use_partial_tpu_embedding": False,
              "interaction": "dot",
              "max_ids_per_chip_per_sample": 128,
              "max_ids_per_table": 4096,
              "max_unique_ids_per_table": 2048,
              "vocab_sizes": [
                  39884406,
                  39043,
                  17289,
                  7420,
                  20263,
                  3,
                  7120,
                  1543,
                  63,
                  38532951,
                  2953546,
                  403346,
                  10,
                  2208,
                  11938,
                  155,
                  4,
                  976,
                  14,
                  39979771,
                  25641295,
                  39664984,
                  585935,
                  12972,
                  108,
                  36,
              ],
          },
      },
      "trainer": {
          "optimizer_config": {
              "embedding_optimizer": "SGD",
              "dense_optimizer": "SGD",
              "lr_config": {
                  "decay_exp": 2,
                  "decay_start_steps": 70000,
                  "decay_steps": 30000,
                  "learning_rate": 0.025,
                  "warmup_steps": 8000,
              },
              "dense_sgd_config": {
                  "decay_exp": 2,
                  "decay_start_steps": 70000,
                  "decay_steps": 30000,
                  "learning_rate": 0.00025,
                  "warmup_steps": 8000,
              },
          },
          "use_orbit": True,
          "validation_interval": 1000,
          "checkpoint_interval": 2000,
          "validation_steps": 5000,
          "train_steps": 10000,
          "steps_per_loop": 1000,
          "summary_interval": 1000,
          "train_tf_function": True,
          "train_tf_while_loop": True,
          "pipeline_sparse_and_dense_execution": True,
      },
  }

  model_dir = "/tmp"

  params_override["trainer"]["pipeline_sparse_and_dense_execution"] = "true"
  tpu_id = Variable.get(benchmark_id, default_var=None)
  # TODO (ericlefort): Replace the model_dir with this line when the var is available
  # model_dir = metric_config.SshEnvVars.GCS_OUTPUT.value + f"/dlrm/v5p/{benchmark_id}"
  epoch = time.time()
  model_dir = f"{gcs_bucket.BASE_OUTPUT_DIR}/{test_owner.Team.SOLUTIONS_TEAM.value}/dlrm/{benchmark_id}/{epoch}"

  # Clean out the prior checkpoint if it exists
  run_model_cmds = (
      (
          f'cd /usr/share/tpu/models && {env_variable} && TF_XLA_FLAGS="--tf_mlir_enable_mlir_bridge=true  --tf_xla_sparse_core_disable_table_stacking=true --tf_mlir_enable_tpu_variable_runtime_reformatting_pass=false --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_merge_control_flow_pass=true --tf_xla_disable_full_embedding_pipelining=true" LIBTPU_INIT_ARGS="--xla_sc_splitting_along_feature_dimension=auto  --copy_with_dynamic_shape_op_output_pjrt_buffer=true"'
          " numactl --cpunodebind=0  --membind=0 python3 official/recommendation/ranking/train.py"
          f" --model_dir={model_dir} {extraFlags}"
          f" --params_override='{params_override}'"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.CHANDRA_D,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      tpu_name_env_var=is_pod,
      all_workers=False,
  )


def get_tf_dlrm_v2_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,
    bottom_mlp: List[int],
    embedding_dim: int,
    train_steps: int,
    extraFlags: str = "",
    project_name: str = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    runtime_version: str = RuntimeVersion.TPU_VM_TF_NIGHTLY.value,
    is_pod: bool = False,
    is_pjrt: bool = True,
    criteo_dir: str = gcs_bucket.CRITEO_DIR,
    network: str = "default",
    subnetwork: str = "default",
    global_batch_size=16384,
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  # Add default_var to pass DAG check
  # TODO(ranran): replace Variable.get() to XCOM when it applies
  test_name = "tf_dlrm_criteo"
  benchmark_id = f"{test_name}-v{tpu_version.value}-{tpu_cores}"
  tpu_name = Variable.get(benchmark_id, default_var=None) if is_pod else "local"
  is_v5p = tpu_version == TpuVersion.V5P
  env_variable = common.export_env_variables(
      tpu_name, is_pod, is_pjrt, is_v5p_sc=is_v5p
  )

  set_up_cmds = common.set_up_tensorflow_models() + common.install_tf()
  if is_pod:
    if not is_pjrt:
      set_up_cmds += common.set_up_se()
    else:
      set_up_cmds += common.set_up_pjrt()
  params_override = {
      "runtime": {
          "distribution_strategy": "tpu",
          "mixed_precision_dtype": "mixed_bfloat16",
      },
      "task": {
          "use_synthetic_data": "false",
          "use_tf_record_reader": "true",
          "train_data": {
              "input_path": "gs://zyc_dlrm/dataset/tb_tf_record_train_val/train/day_*/*",
              "global_batch_size": global_batch_size,
          },
          "validation_data": {
              "input_path": "gs://zyc_dlrm/dataset/tb_tf_record_train_val/eval/day_*/*",
              "global_batch_size": global_batch_size,
          },
          "model": {
              "interaction": "multi_layer_dcn",
              "dcn_num_layers": 3,
              "dcn_low_rank_dim": 512,
              "num_dense_features": 13,
              "bottom_mlp": bottom_mlp,
              "embedding_dim": embedding_dim,
              "top_mlp": [1024, 1024, 512, 256, 1],
              "vocab_sizes": [
                  40000000,
                  39060,
                  17295,
                  7424,
                  20265,
                  3,
                  7122,
                  1543,
                  63,
                  40000000,
                  3067956,
                  405282,
                  10,
                  2209,
                  11938,
                  155,
                  4,
                  976,
                  14,
                  40000000,
                  40000000,
                  40000000,
                  590152,
                  12973,
                  108,
                  36,
              ],
              "multi_hot_sizes": [
                  3,
                  2,
                  1,
                  2,
                  6,
                  1,
                  1,
                  1,
                  1,
                  7,
                  3,
                  8,
                  1,
                  6,
                  9,
                  5,
                  1,
                  1,
                  1,
                  12,
                  100,
                  27,
                  10,
                  3,
                  1,
                  1,
              ],
              "use_multi_hot": "true",
              "concat_dense": "false",
              "dcn_use_bias": "true",
              "max_ids_per_chip_per_sample": 128,
              "max_ids_per_table": 15000,
              "max_unique_ids_per_table": 4096,
              "initialize_tables_on_host": "false",
              "use_partial_tpu_embedding": "false",
              "size_threshold": 0,
          },
      },
      "trainer": {
          "use_orbit": "true",
          "validation_interval": 1000,
          "checkpoint_interval": 0,
          "validation_steps": 1000,
          "train_steps": train_steps,
          "optimizer_config": {
              "embedding_optimizer": "SGD",
              "lr_config": {
                  "decay_exp": 1.6,
                  "decay_start_steps": 150000,
                  "decay_steps": 136054,
                  "learning_rate": 30,
                  "warmup_steps": 8000,
              },
          },
      },
  }

  model_dir = "/tmp"

  params_override["trainer"]["pipeline_sparse_and_dense_execution"] = "true"
  tpu_id = Variable.get(benchmark_id, default_var=None)
  # TODO (ericlefort): Replace the model_dir with this line when the var is available
  # model_dir = metric_config.SshEnvVars.GCS_OUTPUT.value + f"/dlrm/v5p/{benchmark_id}"
  epoch = time.time()
  model_dir = f"{gcs_bucket.BASE_OUTPUT_DIR}/{test_owner.Team.SOLUTIONS_TEAM.value}/dlrm/{benchmark_id}/{epoch}"

  # Clean out the prior checkpoint if it exists
  run_model_cmds = (
      (
          f"cd /usr/share/tpu/models && {env_variable} &&"
          " python3 official/recommendation/ranking/train.py"
          f" --model_dir={model_dir} {extraFlags}"
          f" --params_override='{params_override}'"
      ),
  )

  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=True,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=test_owner.CHANDRA_D,
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
      tpu_name_env_var=is_pod,
      all_workers=False,
  )
