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

"""The file for helper functions and common constants."""

import os


# Environment names
PROD_COMPOSER_ENV_NAME = "ml-automation-solutions"
DEV_COMPOSER_ENV_NAME = "ml-automation-solutions-dev"


# Constants
COMPOSER_ENVIRONMENT = "COMPOSER_ENVIRONMENT"
COMPOSER_LOCATION = "COMPOSER_LOCATION"

DEV_COMPOSER_ENV_GS_BUCKET = "gs://us-central1-ml-automation-s-bc6c8818-bucket"
PROD_COMPOSER_ENV_GS_BUCKET = "gs://us-central1-ml-automation-s-bc954647-bucket"


def is_prod_env() -> bool:
  """Indicate if the composer environment is Prod."""
  return os.environ.get(COMPOSER_ENVIRONMENT) == PROD_COMPOSER_ENV_NAME


def is_dev_env() -> bool:
  """Indicate if the composer environment is Dev."""
  return os.environ.get(COMPOSER_ENVIRONMENT) == DEV_COMPOSER_ENV_NAME


def get_gs_bucket() -> str:
  if is_prod_env():
    return PROD_COMPOSER_ENV_GS_BUCKET
  return DEV_COMPOSER_ENV_GS_BUCKET
