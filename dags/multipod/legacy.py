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

"""A DAG to run tests migrated from the legacy XL ML infrastructure"""

import datetime
from airflow import models
from dags import composer_env, gcs_bucket, test_owner
from dags.vm_resource import TpuVersion, Zone, Project, DockerImage, ClusterName
from dags.multipod.configs import legacy_unit_test, gke_config
from dags.multipod.configs.common import SetupMode, Platform

# Run once a day at 9 am UTC (1 am PST)
SCHEDULED_TIME = "0 9 * * *" if composer_env.is_prod_env() else None


for test_mode in [SetupMode.NIGHTLY, SetupMode.STABLE]:
  with models.DAG(
      dag_id=f"multipod_legacy_xlml_{test_mode.value}",
      schedule=SCHEDULED_TIME,
      tags=["multipod_team", "xlml", "legacy", test_mode.value],
      start_date=datetime.datetime(2024, 1, 10),
      catchup=False,
      concurrency=2,
  ) as dag:
    docker_image = (
        DockerImage.MAXTEXT_JAX_NIGHTLY
        if test_mode == SetupMode.NIGHTLY
        else DockerImage.MAXTEXT_JAX_STABLE
    )
    # Tests that require scripts from the `jax/unit_tests` folder should follow
    # this pattern.
    # TODO(jonbolin): Example for legacy unit test migration - evaluate whether
    # to remove gpt1-like tests once test migration is complete.
    for n_slice in [1, 2]:
      legacy_unit_test.get_legacy_unit_test_config(
          test_cmd=("python3 gpt1-like.py",),
          tpu_version=TpuVersion.V4,
          tpu_cores=16,
          tpu_zone=Zone.US_CENTRAL2_B.value,
          time_out_in_min=60,
          test_name=f"gpt1-like",
          test_mode=test_mode,
          docker_image=docker_image.value,
          test_owner=test_owner.JON_B,
          num_slices=n_slice,
          cluster_name=ClusterName.V4_16_MULTISLICE_CLUSTER.value,
      ).run()

    # Tests that run MaxText end_to_end tests should follow this pattern.
    gke_config.get_gke_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        test_name="maxtext-decode",
        run_model_cmds=(
            "bash end_to_end/test_decode.sh 10 gs://maxtext-xlml gs://maxtext-xlml/dataset xlml-decode-v4-8-1slice-stable",
        ),
        docker_image=docker_image.value,
        test_owner=test_owner.JON_B,
    ).run()
