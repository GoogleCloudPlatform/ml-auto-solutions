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

"""Utilities to construct common configs."""

from typing import Tuple

UPGRADE_PIP = "pip install --upgrade pip"
INSTALL_DEPENDENCIES = (
    "pip install tensorboardX"
)
INSTALL_NIGHTLY_PYTORCH = (
    "pip install "
    "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp310-cp310-linux_x86_64.whl"
)
INSTALL_NIGHTLY_PYTORCH_XLA = (
    "pip install "
    "https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl"
)
INSTALL_SPMD_TRANSFORMERS = (
      "git clone -b llama2-google-next-training https://github.com/pytorch-tpu/transformers.git /tmp/transformers; "
      "pip install /tmp/transformers accelerate datasets evaluate scikit-learn",
)


def set_up_hugging_face_transformers_llama2_fork() -> Tuple[str]:
  """Common set up for SPMD hugging face transformer fork."""
  return (
      UPGRADE_PIP,
      INSTALL_DEPENDENCIES,
      INSTALL_NIGHTLY_PYTORCH,
      INSTALL_NIGHTLY_PYTORCH_XLA,
      INSTALL_SPMD_TRANSFORMERS,
  )
