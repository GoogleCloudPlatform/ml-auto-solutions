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

from typing import Tuple


UPGRADE_PIP = "pip install --upgrade pip"
UPGRADE_SETUPTOOLS = "python -m pip install --upgrade setuptools"
UPGRADE_PACKAGING = "python -m pip install --upgrade packaging"


def set_up_nightly_jax() -> Tuple[str]:
  return (
      (
          "pip install -U --pre libtpu-nightly -f"
          " https://storage.googleapis.com/jax-releases/libtpu_releases.html"
      ),
      (
          "pip install --pre -U jaxlib -f"
          " https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
      ),
      "pip install git+https://github.com/google/jax",
  )


def set_up_jax_version(version) -> Tuple[str]:
  return (
      (
          f"pip install jax[tpu]=={version}  -f "
          "https://storage.googleapis.com/jax-releases/libtpu_releases.html"
      ),
      (
          f"pip install --pre jaxlib=={version} -f"
          " https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html"
      ),
      f"pip install jax=={version}",
  )
