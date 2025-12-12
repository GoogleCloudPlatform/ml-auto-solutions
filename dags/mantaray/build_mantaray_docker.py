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

"""DAG to build nightly mantaray docker image: gcr.io/tpu-prod-env-multipod/mantaray_maxtext_tpu:nightly"""

from airflow import models

from dags.common import test_owner
from xlml.utils import mantaray
import datetime
from dags import composer_env

# Run this script in prod env only to not duplicate image building in dev airflow.
if composer_env.is_prod_env():
  with models.DAG(
      dag_id="mantaray_nightly_docker",
      schedule="0 1 * * *",  # everyday at 1am (UTC). Borg job is scheduled at midnight to upload source code to GCS.
      tags=["mantaray", "CPU"],
      start_date=datetime.datetime(2024, 9, 4),
      catchup=False,
  ) as dag:
    run_workload = mantaray.build_docker_image.override(
        owner=test_owner.BHAVYA_B
    )()
    run_workload
