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

"""Utilities to construct common configs."""

from __future__ import annotations
import time


CMD_PRINT_TF_VERSION = "python3 -c \"import tensorflow; print('Running using TensorFlow Version: ' + tensorflow.__version__)\""
CMD_REMOVE_LIBTPU_LOCKFILE = "sudo rm -f /tmp/libtpu_lockfile"
CMD_INSTALL_KERAS_NIGHTLY = (
    "pip install --upgrade --no-deps --force-reinstall tf-keras-nightly"
)


def set_up_se(
    major: Optional[int] = None,
    minor: Optional[int] = None,
    patch: Optional[int] = None,
) -> tuple[str]:
  """Adjust grpc_tpu_worker for SE tests"""
  grpc_version = "se-nightly"
  if major is not None:
    grpc_version = f"tf-{major}.{minor}.{patch}-se"
  return (
      CMD_REMOVE_LIBTPU_LOCKFILE,
      f"sudo sed -i 's/TF_DOCKER_URL=.*/TF_DOCKER_URL=gcr.io\/cloud-tpu-v2-images\/grpc_tpu_worker:{grpc_version}\"/' /etc/systemd/system/tpu-runtime.service",
      "sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime",
      "cat /etc/systemd/system/tpu-runtime.service",
  )


def set_up_pjrt(
    major: Optional[int] = None,
    minor: Optional[int] = None,
    patch: Optional[int] = None,
) -> tuple[str]:
  """Adjust grpc_tpu_worker for PjRt tests"""
  grpc_version = "nightly"
  if major is not None:
    grpc_version = f"tf-{major}.{minor}.{patch}-pjrt"
  return (
      f"sudo sed -i 's/TF_DOCKER_URL=.*/TF_DOCKER_URL=gcr.io\/cloud-tpu-v2-images\/grpc_tpu_worker:{grpc_version}\"/' /etc/systemd/system/tpu-runtime.service",
      "sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime",
      "cat /etc/systemd/system/tpu-runtime.service",
  )


def install_tf(
    major: Optional[int] = None,
    minor: Optional[int] = None,
    patch: Optional[int] = None,
    release_candidate: Optional[int] = None,
    libtpu_version: Optional[str] = None,
) -> tuple[str]:
  """Install tf + libtpu.

  If the version numbers are set, installs that version. Otherwise just installs using nightly.
  Either all of the version numbers need to be set or none of them should be set.

  Args:
      major (Optional[int]): The major version number
      minor (Optional[int]): The minor version number
      patch (Optional[int]): The minor version number
      libtpu_version (Optional[str]): The libtpu version to install
  """
  tf_installation_command = f"pip install tf-nightly-tpu -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force"
  if release_candidate is None:
    release_candidate = ""
  if major is not None:
    tf_installation_command = f"pip install tensorflow-tpu=={major}.{minor}.{patch}{release_candidate} -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force"
  cmds_install_tf_whl = tf_installation_command

  return (
      cmds_install_tf_whl,
      CMD_PRINT_TF_VERSION,
  )


def set_up_tensorflow_models(
    models_branch: Optional[str] = None,
    keras_version: Optional[str] = None,
) -> tuple[str]:
  """Common set up for tensorflow models for the release.

  If any versions are not set, defaults to nightly.

  Args:
      models_branch (Optional[str]): The models branch to use
  """
  if models_branch is None:
    models_branch = "master"

  cmd_install_keras = CMD_INSTALL_KERAS_NIGHTLY
  if keras_version is not None:
    cmd_install_keras = f"pip install --upgrade --force-reinstall --no-deps tf-keras=={keras_version}"

  return (
      "sudo mkdir -p /usr/share/tpu && cd /usr/share/tpu",
      f'if [ ! -d "/usr/share/tpu/models" ]; then sudo git clone -b {models_branch} https://github.com/tensorflow/models.git; fi',
      "pip install -r /usr/share/tpu/models/official/requirements.txt",
      f'if [ ! -d "/usr/share/tpu/recommenders" ]; then sudo git clone -b main https://github.com/tensorflow/recommenders.git; fi',
      f"pip install gin-config && pip install tensorflow-datasets",
      cmd_install_keras,
  )


def export_env_variables(
    tpu_name: str,
    is_pod: bool,
    is_pjrt: bool,
    is_v5p_sc: Optional[bool] = False,
) -> str:
  """Export environment variables for training."""
  stmts = [
      "export WRAPT_DISABLE_EXTENSIONS=true",
      "export TF_USE_LEGACY_KERAS=1",
  ]

  stmts.append(f"export TPU_NAME={tpu_name}")
  stmts.append(
      "export TF_XLA_FLAGS='--tf_mlir_enable_mlir_bridge=true --tf_xla_sparse_core_disable_table_stacking=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_merge_control_flow_pass=true'"
  )
  stmts.append(
      "export PYTHONPATH='/usr/share/tpu/recommenders:/usr/share/tpu/models'"
  )
  if is_pod:
    stmts.append("export TPU_LOAD_LIBRARY=0")

  if not is_pod and is_pjrt:
    stmts.append("export NEXT_PLUGGABLE_DEVICE_USE_C_API=true")
    stmts.append(
        "export TF_PLUGGABLE_DEVICE_LIBRARY_PATH=/home/$USER/.local/lib/python3.10/site-packages/libtpu/libtpu.so"
    )

  return " && ".join(stmts)
