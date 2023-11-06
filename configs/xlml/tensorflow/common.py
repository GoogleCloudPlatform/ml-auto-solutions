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


def set_up_google_tensorflow_models() -> Tuple[str]:
  """Common set up for tensorflow."""
  return (
      "pip install -r /usr/share/tpu/models/official/requirements.txt",
      "pip install tensorflow-text-nightly",
      "gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/tensorflow/tf-nightly/latest/*.whl /tmp/ && pip install /tmp/tf*.whl --force",
      "sudo gsutil -m cp gs://cloud-tpu-v2-images-dev-artifacts/libtpu/latest/libtpu.so /lib/",
      "sudo sed -i 's/TF_DOCKER_URL=.*/TF_DOCKER_URL=gcr.io\/cloud-tpu-v2-images-dev\/grpc_tpu_worker:nightly\"/' /etc/systemd/system/tpu-runtime.service",
      "sudo systemctl daemon-reload && sudo systemctl restart tpu-runtime",
      "pip install tensorflow-recommenders --no-deps",
  )
