# Copyright 2025 Google LLC
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

from airflow.models.param import Param
from dags.maxtext_pathways.configs import model_configs as model_cfg
from dags.common.vm_resource import TpuVersion

MODEL_FRAMEWORK = ["mcjax", "pathways"]

MODEL_NAME = []
V5E_MODEL_NAME = [config.value for config in model_cfg.MaxTextV5eModelConfigs]
V5P_MODEL_NAME = [config.value for config in model_cfg.MaxTextV5pModelConfigs]
V6E_MODEL_NAME = [
    config.value for config in model_cfg.MaxTextTrilliumModelConfigs
]
MODEL_NAME.extend(V5E_MODEL_NAME)
MODEL_NAME.extend(V5P_MODEL_NAME)
MODEL_NAME.extend(V6E_MODEL_NAME)

DEVICE_VERSION = ["v" + version.value for version in TpuVersion]

PARAMETERS = {
    "user": Param(
        "root",
        type="string",
        title="User",
        description="User name is used to confirm the first three characters of the pod in the cluster.",
    ),
    "cluster_name": Param(
        "pw-v6e-64",
        type="string",
        title="Cluster Name",
        description="GCP cluster name for training model.",
    ),
    "project": Param(
        "cienet-cmcs",
        type="string",
        title="Project",
        description="GCP project ID for training model.",
    ),
    "zone": Param(
        "us-central1-b",
        type="string",
        title="Zone",
        description="Cluster zone.",
    ),
    "device_version": Param(
        "v6e",
        type="string",
        title="Device Version",
        description='Device type for the cluster. ex: "v5litepod"-32',
        enum=DEVICE_VERSION,
    ),
    "core_count": Param(
        64,
        type="integer",
        title="Core Count",
        description='Device core count for the cluster. ex: v6e-"64"',
    ),
    "service_account": Param(
        None,
        type=["string", "null"],
        title="Service account",
        description="Service account of the project.",
    ),
    "benchmark_steps": Param(
        20,
        type="integer",
        title="Benchmark Steps",
        description="Number of benchmark steps.",
    ),
    "num_slices_list": Param(
        1,
        type="integer",
        title="Number Slices",
        description="Number of slices",
    ),
    "server_image": Param(
        # TODO(b/451750407): Replace this temporary image with a formal one.
        "gcr.io/cienet-cmcs/lidanny/unsanitized_server:latest",
        type="string",
        title="Server Image",
        description="Server image for pathways.",
    ),
    "proxy_image": Param(
        # TODO(b/451750407): Replace this temporary image with a formal one.
        "gcr.io/cienet-cmcs/lidanny/unsanitized_proxy_server:latest",
        type="string",
        title="Proxy Image",
        description="Proxy image for pathways.",
    ),
    "runner": Param(
        # TODO(b/451750407): Replace this temporary image with a formal one.
        "gcr.io/cienet-cmcs/lidanny_latest:latest",
        type="string",
        title="Runner Image",
        description="Runner image for the cluster.",
    ),
    "selected_model_framework": Param(
        MODEL_FRAMEWORK[1],
        type="string",
        title="Model Framework",
        description="Select a model framework to run.",
        enum=MODEL_FRAMEWORK,  # dropdown menu
    ),
    "selected_model_names": Param(
        "default_basic_1",  # Available model for device type 'v6e'
        type="string",
        title="Model Name",
        description='Select a model name to run. Only when "customized_model_name" is selected for "Model Name", the input value of "customized_model_name" parameter will take effect.',
        enum=["customized_model_name"] + MODEL_NAME,
    ),
    "customized_model_name": Param(
        None,
        type=["null", "string"],
        title="Customized Model Name",
        description='Select a customized model name to run. Only when "customized_model_name" is selected for "Model Name", the input value of "customized_model_name" parameter will take effect.',
    ),
    "priority": Param(
        "medium",
        type="string",
        title="Priority",
        description="Priority for the workload",
        enum=["very high", "high", "medium", "low"],
    ),
    "max_restarts": Param(
        1,
        type="integer",
        title="Max Restarts",
        description="Max restarts for the workload",
    ),
    "bq_enable": Param(
        False,
        type="boolean",
        title="BigQuery Enable",
        description="Enable BigQuery to store metrics data",
    ),
    "bq_db_project": Param(
        "cloud-tpu-multipod-dev",
        type="string",
        title="BigQuery Database Project",
        description="The Project of BigQuery Database",
    ),
    "bq_db_dataset": Param(
        # TODO(b/451750407): Replace this temporary image with a formal one.
        "chzheng_test_100steps",
        type="string",
        title="BigQuery Database Dataset",
        description="The Dataset of BigQuery Database",
    ),
    "override_timeout_in_min": Param(
        None,
        type=["null", "integer"],
        title="Override Timeout In Minutes",
        description=(
            "Timeout in minutes for the workload task. "
            "Adjust it when your meet (airflow.exceptions.AirflowException: Timed out after ...) issue. "
            "The default value `None` means automatic calculation of timeout."
        ),
    ),
}
