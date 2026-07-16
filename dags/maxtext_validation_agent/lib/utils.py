# Copyright 2026 Google LLC
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

"""Utilities to construct MaxText decoding validation configs."""

import datetime
from xlml.apis import gcp_config, task, test_config, metric_config
from dags.common import vm_resource

# airflow imports
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s

def get_maxtext_validation_config(
    tpu_version: str,
    tpu_cores: int,
    tpu_zone: str,
    time_out_in_min: int,

) -> task.XpkTask:
    """Generates the XPK task configuration for MaxText validation."""

    job_gcp_config = gcp_config.GCPConfig(
        project_name=vm_resource.Project.TPU_PROD_ENV_MULTIPOD.value,
        zone=tpu_zone,
        dataset_name=metric_config.DatasetOption.XLML_DATASET,
        composer_project=vm_resource.Project.TPU_PROD_ENV_MULTIPOD.value,
    )

    run_model_cmds = (
        # clone the MaxText repository dynamically using Airflow Jinja templating.
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        # check out a specific commit_hash if provided (for reproducible testing of PRs), 
        # otherwise fallback to checking out the specified branch name, defaulting to 'main'.
        "cd /tmp/maxtext && git checkout {{ dag_run.conf.get('maxtext_commit_hash', params.get('maxtext_commit_hash')) or dag_run.conf.get('maxtext_branch', params.get('maxtext_branch', 'main')) }}",
        "cd /tmp/maxtext && pip install --no-deps -e .",
        "cd /tmp/maxtext && echo '\\''{{ dag_run.conf | tojson }}'\\'' > config.json",
        "cd /tmp/maxtext && python3 src/maxtext/experimental/agent/checkpoint_validation_agent/main.py --config config.json",
    )

    job_test_config = test_config.TpuGkeTest(
        accelerator=test_config.Tpu(
            version=test_config.TpuVersion(str(tpu_version)),
            cores=tpu_cores,
            runtime_version="tpu-ubuntu2204-base",
            reserved=True,
        ),
        test_name="maxtext_decoding_checkpoint_validation",
        set_up_cmds=(
            "pip install --upgrade pip",
            "google-cloud-sdk/bin/gcloud components update --quiet",
        ),
        run_model_cmds=run_model_cmds,
        timeout=datetime.timedelta(minutes=time_out_in_min),
        task_owner="airflow",
        cluster_name="{{ dag_run.conf['xpk_cluster_name'] }}",
        docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-06",
        num_slices=1,
    )

    return task.XpkTask(
        task_test_config=job_test_config,
        task_gcp_config=job_gcp_config,
    )

def get_checkpoint_shape_validation_task(
    dag,
    model_name: str,
    checkpoint_gcs_path: str,
    scan_layers: str = "false",
):
    """
    Sub-DAG A: Post-Conversion Shape Validation.
    Executes existing scripts in the maxtext repo to validate checkpoint metadata.
    """
    compute_resources = k8s.V1ResourceRequirements(
        requests={"memory": "4Gi", "ephemeral-storage": "10Gi", "cpu": "2"},
        limits={"memory": "8Gi", "ephemeral-storage": "10Gi", "cpu": "4"}
    )

    cmds = [
        "set -e",
        # Clone the MaxText repository.
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        # Check out a specific commit_hash if provided (for reproducible testing of PRs), 
        # otherwise fallback to checking out the specified branch name, defaulting to 'main'.
        "cd /tmp/maxtext && git checkout {{ dag_run.conf.get('maxtext_commit_hash', params.get('maxtext_commit_hash')) or dag_run.conf.get('maxtext_branch', params.get('maxtext_branch', 'main')) }}",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",

        # Dump theoretical (ideal) MaxText parameter shapes into a text file.
        "python3 -m maxtext.checkpoint_conversion.inspect_checkpoint maxtext "
        f"model_name={model_name} scan_layers={scan_layers} --output_file=/tmp/ideal_raw.txt",

        # Dump actual Orbax checkpoint parameter shapes from GCS into a text file.
        "python3 -m maxtext.checkpoint_conversion.inspect_checkpoint orbax "
        f"--path {checkpoint_gcs_path} --output_file=/tmp/actual_raw.txt",

        # Filter the raw dumps to only extract the dictionary keys mapping to tensor shapes.
        "grep '^key:' /tmp/ideal_raw.txt > /tmp/ideal_shapes.txt",
        "grep '^key:' /tmp/actual_raw.txt > /tmp/actual_shapes.txt",

        # Execute the shape validator script to compare the two shape files.
        "python3 /tmp/maxtext/src/maxtext/experimental/agent/checkpoint_validation_agent/checkpoint_shape_validator.py "
        "--report_gcs_dir={{ dag_run.conf.get('report_gcs_dir', params.get('report_gcs_dir', '')) }}"
    ]

    return KubernetesPodOperator(
        task_id="checkpoint_shape_validation",
        name="checkpoint-shape-validation-pod",
        namespace="composer-user-workloads",
        config_file="/home/airflow/composer_kube_config",
        image="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-06",
        cmds=["bash", "-c"],
        arguments=[" && ".join(cmds)],
        container_resources=compute_resources,
        get_logs=True,
        startup_timeout_seconds=600,
        dag=dag,
    )

def get_mock_tensor_validation_task(dag, model_name, checkpoint_path):
    """
    Sub-DAG B: Mock Tensor Dry Run.
    Verifies that the model can run a forward pass without crashing.
    """
    compute_resources = k8s.V1ResourceRequirements(
        requests={"memory": "4Gi", "cpu": "2"},
        limits={"memory": "8Gi", "cpu": "4"}
    )

    cmds = [
        "set -e",
        # Clone the MaxText repository.
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        # Check out a specific commit_hash if provided (for reproducible testing of PRs), 
        # otherwise fallback to checking out the specified branch name, defaulting to 'main'.
        "cd /tmp/maxtext && git checkout {{ dag_run.conf.get('maxtext_commit_hash', params.get('maxtext_commit_hash')) or dag_run.conf.get('maxtext_branch', params.get('maxtext_branch', 'main')) }}",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",
        
        # Execute the mock tensor forward pass (dry run).
        # Note: This is a fast verification step that uses mock/synthetic tensors to ensure 
        # the model's shapes and tensor layouts are correct without burning heavy TPU compute. 
        # It does NOT catch logic/math bugs (that is handled by the downstream logit verification task).
        # This dynamically unpacks the 'maxtext_overrides' dictionary from the DAG runtime config 
        # (or fallback params) into command-line arguments using Airflow Jinja templating.
        (
            f"python3 /tmp/maxtext/src/maxtext/experimental/agent/checkpoint_validation_agent/mock_tensor_test.py "
            f"{checkpoint_path} {model_name} "
            "--report_gcs_dir={{ dag_run.conf.get('report_gcs_dir', params.get('report_gcs_dir', '')) }} "
            "{% for k, v in dag_run.conf.get('maxtext_overrides', params.get('maxtext_overrides', {})).items() %}"
            "{{ k }}={{ v }} "
            "{% endfor %}"
        )
    ]

    return KubernetesPodOperator(
        task_id="mock_tensor_validation",
        name="mock-tensor-validation-pod",
        namespace="composer-user-workloads",
        config_file="/home/airflow/composer_kube_config",
        image="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-06",
        cmds=["bash", "-c"],
        arguments=[" && ".join(cmds)],
        container_resources=compute_resources,
        get_logs=True,
        dag=dag,
    )
