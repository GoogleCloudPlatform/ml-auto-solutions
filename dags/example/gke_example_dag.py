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

"""A DAG to run all GKE examples."""

import datetime
from airflow import models
from configs import vm_resource
from configs.example import gke_example_config as config


# TODO(ranran): add following examples:
# 1) jax_resnet_tpu_example (gce example dag)
# 2) jax_vit_tpu_benchmark_example (gce example dag)
# 3) jax_vit_tpu_benchmark_example (same dag)
with models.DAG(
    dag_id="gke_example_dag",
    schedule=None,
    tags=["example", "gke", "xlml", "benchmark"],
    start_date=datetime.datetime(2023, 11, 29),
    catchup=False,
) as dag:
  jax_resnet_tpu_example = config.get_flax_resnet_gke_config(
      tpu_version="4",
      tpu_cores=8,
      tpu_zone=vm_resource.Zone.US_CENTRAL2_B.value,
      cluster_name=vm_resource.ClusterName.V4_8_CLUSTER.value,
      cluster_config=vm_resource.ClusterConfig.V5E_CONFIG.value,
      docker_image=vm_resource.DockerImage.DEMO_TEST.value,
      time_out_in_min=60,
  ).run()

  # Test dependencies
  jax_resnet_tpu_example
