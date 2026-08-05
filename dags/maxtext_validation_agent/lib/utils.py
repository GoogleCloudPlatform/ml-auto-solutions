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

# pylint: disable=line-too-long

import datetime
import json
import logging
import time

from xlml.apis import gcp_config, task, test_config, metric_config
from dags.common import vm_resource
from dags.common.vm_resource import XpkClusters

# airflow imports
from airflow.utils.session import create_session
from airflow.models import DagRun
from google.cloud import run_v2
from google.cloud import storage
from google.api_core.exceptions import GoogleAPICallError

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from kubernetes.client import models as k8s


class DynamicTpuGkeTest(test_config.TpuGkeTest):

    @property
    def benchmark_id(self) -> str:
        return self.test_name


def trigger_agent_on_failure(context):
    """Publishes complete Airflow failure context and invokes the Cloud Run Job."""
    import logging

    from google.cloud import run_v2, storage

    logger = logging.getLogger(__name__)
    task_instance = context.get("task_instance") if context else None
    task_id = str(getattr(task_instance, "task_id", "unknown_task"))
    task_state = str(getattr(task_instance, "state", ""))

    # Skip remediation if task was manually killed, removed, shut down, or failed due to an upstream task.
    if task_state in ("upstream_failed", "killed", "removed", "shutdown"):
        logger.info(
            "Skipping agent remediation for task %s with non-executable state '%s'",
            task_id,
            task_state,
        )
        return

    # The guard task mirrors an upstream failure and must not spawn a second agent.
    if task_id == "check_upstream_failures":
        logger.info(
            "Skipping duplicate remediation callback for guard task %s", task_id
        )
        return

    try:
        dag_run = context.get("dag_run") if context else None
        conf = dict(getattr(dag_run, "conf", None) or {})
        dag_id = str(getattr(task_instance, "dag_id", "unknown_dag"))
        airflow_run_id = str(getattr(dag_run, "run_id", "unknown_run"))
        logical_date = getattr(dag_run, "logical_date", None)
        run_name = str(conf.get("run_name", "default_run"))
        remediation_key = f"{dag_id}:{airflow_run_id}:{task_id}"

        # Prevent infinite agent retries by capping at 5 runs for this specific run_name.

        with create_session() as session:
            runs = session.query(DagRun).filter(DagRun.dag_id == dag_id).all()
            related_runs = [
                r
                for r in runs
                if r.conf and r.conf.get("run_name", "").startswith(run_name)
            ]
            if len(related_runs) >= 5:
                logger.info(
                    "Maximum agent retries reached (5). Skipping agent trigger."
                )
                return

        error_msg = str(context.get("exception", "")) if context else ""
        if len(error_msg) > 15000:
            error_msg = "...[TRUNCATED]...\n" + error_msg[-15000:]

        trigger_data = {
            "schema_version": 2,
            "remediation_key": remediation_key,
            "airflow_error_message": error_msg,
            "airflow_task_id": task_id,
            "airflow_dag_id": dag_id,
            "airflow_run_id": airflow_run_id,
            "airflow_logical_date": logical_date.isoformat() if logical_date else "",
            "run_name": f"{run_name}_{airflow_run_id}",
            "dag_conf": conf,
            "maxtext_branch": str(conf.get("maxtext_branch", "main")),
            "maxtext_commit_hash": str(conf.get("maxtext_commit_hash", "")),
            "maxtext_model_name": str(conf.get("maxtext_model_name", "unknown_model")),
            "forward_pass_maxtext_overrides": conf.get(
                "forward_pass_maxtext_overrides", {}
            ),
            "decode_maxtext_overrides": conf.get("decode_maxtext_overrides", {}),
            "checkpoint_gcs_path": str(conf.get("checkpoint_gcs_path", "")),
            "report_gcs_dir": str(conf.get("report_gcs_dir", "")),
            "hf_model_path": str(conf.get("hf_model_path", "")),
            "hf_ref_code_url": str(conf.get("hf_ref_code_url", "")),
            "hf_config_url": str(conf.get("hf_config_url", "")),
            "alert_recipient": str(
                conf.get("email") or conf.get("alert_recipient", "")
            ),
        }

        bucket_name = conf.get(
            "agent_trigger_bucket", "maxtext-validation-agent-reports"
        )
        blob_name = f"airflow_direct_failure_{int(time.time())}_{task_id}.json"
        storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
            json.dumps(trigger_data, indent=2, default=str),
            content_type="application/json",
        )
        logger.info("Published remediation trigger gs://%s/%s", bucket_name, blob_name)

        job_name = conf.get(
            "overwatch_cloud_run_job",
            "projects/tpu-prod-env-multipod/locations/us-central1/jobs/maxtext-validation-job",
        )
        operation = run_v2.JobsClient().run_job(name=job_name)
        logger.info("Triggered Cloud Run Job operation %s", operation.operation.name)
    except Exception:
        logger.exception("Failed to publish or invoke Overwatch remediation")


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
        limits={"memory": "8Gi", "ephemeral-storage": "10Gi", "cpu": "4"},
    )

    cmds = [
        "set -e",
        # Clone the MaxText repository.
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        # Check out a specific commit_hash if provided (for reproducible testing of PRs),
        # otherwise fallback to checking out the specified branch name, defaulting to 'main'.
        "cd /tmp/maxtext && git checkout {% set target_commit = dag_run.conf.get('maxtext_commit_hash') or params.get('maxtext_commit_hash') %}{% set target_branch = dag_run.conf.get('maxtext_branch') or params.get('maxtext_branch') or 'main' %}{{ var.value.get('OVERRIDE_BRANCH_' ~ (dag_run.conf.get('run_name', params.get('run_name', 'default_run'))), target_commit or target_branch) }}",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",
        # Dump theoretical (ideal) MaxText parameter shapes into a text file.
        "python3 /tmp/maxtext/src/maxtext/checkpoint_conversion/inspect_checkpoint.py maxtext "
        f"model_name={model_name} scan_layers={scan_layers} --output_file=/tmp/ideal_raw.txt",
        # Dump actual Orbax checkpoint parameter shapes from GCS into a text file.
        "python3 /tmp/maxtext/src/maxtext/checkpoint_conversion/inspect_checkpoint.py orbax "
        f"--path {checkpoint_gcs_path} --output_file=/tmp/actual_raw.txt",
        # Filter the raw dumps to only extract the dictionary keys mapping to tensor shapes.
        "grep '^key:' /tmp/ideal_raw.txt > /tmp/ideal_shapes.txt",
        "grep '^key:' /tmp/actual_raw.txt > /tmp/actual_shapes.txt",
        "python3 /tmp/maxtext/src/maxtext/experimental/agent/ckpt_validation_pipeline/checkpoint_shape_validator.py "
        "--run_name={{ dag_run.conf.get('run_name', params.get('run_name', 'default_run')) }}_{{ run_id }} "
        "--report_gcs_dir={{ dag_run.conf.get('report_gcs_dir', params.get('report_gcs_dir', '')) | trim('/') }}/{{ dag_run.conf.get('run_name', run_id) if dag_run and dag_run.conf else run_id }}",
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
        on_finish_action="keep_pod",
        startup_timeout_seconds=600,
        dag=dag,
    )


def get_forward_compile_validation_task(dag):
    """
    Sub-DAG B: Mock Tensor Dry Run.
    Verifies that the model can run a forward pass without crashing.
    """
    compute_resources = k8s.V1ResourceRequirements(
        requests={"memory": "4Gi", "cpu": "2"},
        limits={"memory": "8Gi", "cpu": "4"},
    )

    cmds = [
        "set -e",
        # Clone the MaxText repository.
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        # Check out a specific commit_hash if provided (for reproducible testing of PRs),
        # otherwise fallback to checking out the specified branch name, defaulting to 'main'.
        "cd /tmp/maxtext && git checkout {% set target_commit = dag_run.conf.get('maxtext_commit_hash') or params.get('maxtext_commit_hash') %}{% set target_branch = dag_run.conf.get('maxtext_branch') or params.get('maxtext_branch') or 'main' %}{{ var.value.get('OVERRIDE_BRANCH_' ~ (dag_run.conf.get('run_name', params.get('run_name', 'default_run'))), target_commit or target_branch) }}",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",
        # Execute the mock tensor forward pass (dry run).
        # Note: This is a fast verification step that uses mock/synthetic tensors to ensure
        # the model's shapes and tensor layouts are correct without burning heavy TPU compute.
        # It does NOT catch logic/math bugs (that is handled by the downstream logit verification task).
        # This dynamically unpacks the 'maxtext_overrides' dictionary from the DAG runtime config
        # (or fallback params) into command-line arguments using Airflow Jinja templating.
        (
            "python3 /tmp/maxtext/src/maxtext/experimental/agent/ckpt_validation_pipeline/forward_compile_validator.py "
            "--report_gcs_dir={{ dag_run.conf.get('report_gcs_dir', params.get('report_gcs_dir', '')) | trim('/') }}/{{ dag_run.conf.get('run_name', run_id) if dag_run and dag_run.conf else run_id }} "
            "run_name={{ dag_run.conf.get('run_name', params.get('run_name', 'default_run')) }}_{{ run_id }} "
            "load_parameters_path={{ dag_run.conf.get('checkpoint_gcs_path', params.get('checkpoint_gcs_path', '')) }} "
            "model_name={{ dag_run.conf.get('maxtext_model_name', params.get('maxtext_model_name', '')) }} "
            "{% for k, v in dag_run.conf.get('forward_pass_maxtext_overrides', params.get('forward_pass_maxtext_overrides', {})).items() %}"
            '{{ k }}="{{ v }}" '
            "{% endfor %}"
        ),
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
        on_finish_action="keep_pod",
        dag=dag,
    )


def get_golden_logits_generation_task(
    tpu_project: str = None,
    tpu_zone: str = None,
    time_out_in_min: int = 120,
) -> task.XpkTask:
    """
    Sub-DAG Pre-step: Compute Golden Logits on High-RAM CPU.
    Downloads HuggingFace weights to a mega-memory CPU node, generates reference logits,
    uploads them to GCS, and deletes the cache since it's an ephemeral pod.
    """

    cpu_cluster = XpkClusters.CPU_M1_MEGAMEM_96_CLUSTER
    job_gcp_config = gcp_config.GCPConfig(
        project_name=cpu_cluster.project,
        zone=cpu_cluster.zone,
        dataset_name=metric_config.DatasetOption.XLML_DATASET,
        composer_project=tpu_project or cpu_cluster.project,
    )

    run_model_cmds = (
        "set -e",
        "export HF_TOKEN=\"{{ dag_run.conf.get('hf_token', params.get('hf_token', '')) }}\"",
        'if [[ "$HF_TOKEN" == "None" || -z "$HF_TOKEN" ]]; then unset HF_TOKEN; fi',
        "export HF_MODEL=\"{{ dag_run.conf.get('hf_model_path', params.get('hf_model_path', dag_run.conf.get('forward_pass_maxtext_overrides', params.get('forward_pass_maxtext_overrides', {})).get('hf_model_path', ''))) }}\"",
        "export MAXTEXT_MODEL=\"{{ dag_run.conf.get('maxtext_model_name', params.get('maxtext_model_name')) }}\"",
        'if gcloud storage ls "gs://maxtext-validation-golden-logits/golden-logits/${HF_MODEL}/${MAXTEXT_MODEL}_golden_logits.jsonl" >/dev/null 2>&1; then echo "Golden logits already exist. Skipping generation!"; exit 0; fi',
        # Clone the repository and checkout the targeted branch
        "cd /tmp && git clone https://github.com/AI-Hypercomputer/maxtext.git",
        "cd /tmp/maxtext && git checkout {{ var.value.get('OVERRIDE_BRANCH_' ~ (dag_run.conf.get('run_name', params.get('run_name', 'default_run'))), dag_run.conf.get('maxtext_commit_hash', params.get('maxtext_commit_hash')) or dag_run.conf.get('maxtext_branch', params.get('maxtext_branch', 'main'))) }}",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",
        # Install PyTorch and HF stuff here, where we have 1.4TB of RAM!
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        "pip install accelerate jsonlines huggingface_hub transformers numpy sentencepiece bs4",
        (
            "cd /tmp/maxtext && python3 -m tests.assets.logits_generation.generate_hf_golden_logits "
            '--model-id="${HF_MODEL}" '
            "--prompts=\"{{ dag_run.conf.get('prompts', params.get('prompts', 'I love to;Today is a;What is the')) }}\" "
            '--output-path="${MAXTEXT_MODEL}_golden_logits.jsonl" '
            "--gcs-bucket=maxtext-validation-golden-logits"
        ),
    )

    job_test_config = test_config.CpuGkeTest(
        accelerator=test_config.Cpu(
            device_type=vm_resource.CpuVersion.M1_MEGAMEM,
            machine_count=1,
        ),
        test_name="maxtext_golden_logits_generation",
        set_up_cmds=(
            "pip install --upgrade pip",
            "google-cloud-sdk/bin/gcloud components update --quiet",
        ),
        run_model_cmds=run_model_cmds,
        timeout=datetime.timedelta(minutes=time_out_in_min),
        task_owner="airflow",
        cluster_name="m1-megamem-96-shared",
        docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-06",
        num_slices=1,
    )

    return task.XpkTask(
        task_test_config=job_test_config,
        task_gcp_config=job_gcp_config,
    )


def get_cluster_config(cluster_name: str):
    """Finds an XpkClusterConfig from vm_resource by its cluster name, defaulting to v4-8."""
    from dags.common.vm_resource import XpkClusters, XpkClusterConfig

    for attr in dir(XpkClusters):
        val = getattr(XpkClusters, attr)
        if isinstance(val, XpkClusterConfig) and val.name == cluster_name:
            return val
    return XpkClusters.TPU_V4_8_MAXTEXT_CLUSTER


def get_forward_pass_validation_task(
    cluster_config=None,
    tpu_version: str = None,
    tpu_cores: int = None,
    tpu_zone: str = None,
    tpu_project: str = None,
    time_out_in_min: int = 45,
) -> task.XpkTask:
    """
    Sub-DAG C: Forward Pass Logits Verification.
    Executes a forward pass on a TPU cluster using Snehal's logit checker script with sow.
    This step ensures that the model is mathematically equivalent to its HuggingFace baseline.
    """
    if cluster_config is not None:
        tpu_version = cluster_config.device_version.value
        tpu_cores = cluster_config.core_count
        tpu_zone = f"{{{{ dag_run.conf.get('xpk_zone', params.get('xpk_zone', '{cluster_config.zone}')) }}}}"
        tpu_project = f"{{{{ dag_run.conf.get('xpk_project', params.get('xpk_project', '{cluster_config.project}')) }}}}"

    job_gcp_config = gcp_config.GCPConfig(
        project_name=tpu_project,
        zone=tpu_zone,
        dataset_name=metric_config.DatasetOption.XLML_DATASET,
        composer_project=tpu_project,
    )

    run_model_cmds = (
        "set -e",
        "export HF_TOKEN=\"{{ dag_run.conf.get('hf_token', params.get('hf_token', '')) }}\"",
        'if [[ "$HF_TOKEN" == "None" || -z "$HF_TOKEN" ]]; then unset HF_TOKEN; fi',
        # Clone the repository and checkout the targeted branch
        "cd /tmp && git clone https://github.com/AI-Hypercomputer/maxtext.git",
        "cd /tmp/maxtext && git checkout {% set target_commit = dag_run.conf.get('maxtext_commit_hash') or params.get('maxtext_commit_hash') %}{% set target_branch = dag_run.conf.get('maxtext_branch') or params.get('maxtext_branch') or 'main' %}{{ var.value.get('OVERRIDE_BRANCH_' ~ (dag_run.conf.get('run_name', params.get('run_name', 'default_run'))), target_commit or target_branch) }}",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        # Download the golden logits locally
        (
            "gcloud storage cp gs://maxtext-validation-golden-logits/golden-logits/"
            "{{ dag_run.conf.get('hf_model_path', params.get('hf_model_path', dag_run.conf.get('forward_pass_maxtext_overrides', params.get('forward_pass_maxtext_overrides', {})).get('hf_model_path', ''))) }}/"
            "{{ dag_run.conf.get('maxtext_model_name', params.get('maxtext_model_name')) }}_golden_logits.jsonl /tmp/golden_logits.jsonl"
        ),
        # Run our wrapper for Snehal's logit checker script.
        # This catches errors and writes a standard JSON report to GCS.
        # It pulls the pre-computed PyTorch golden logits from GCS to completely avoid downloading
        # HuggingFace weights or PyTorch onto the TPU memory directly.
        (
            "cd /tmp/maxtext && python3 src/maxtext/experimental/agent/ckpt_validation_pipeline/forward_pass_validator.py "
            "--run_name={{ dag_run.conf.get('run_name', params.get('run_name', 'default_run')) }}_{{ run_id }} "
            "--maxtext_model_name={{ dag_run.conf.get('maxtext_model_name', params.get('maxtext_model_name')) }} "
            "--checkpoint_gcs_path={{ dag_run.conf.get('checkpoint_gcs_path', params.get('checkpoint_gcs_path')) }} "
            "--report_gcs_dir={{ dag_run.conf.get('report_gcs_dir', params.get('report_gcs_dir', '')) | trim('/') }}/{{ dag_run.conf.get('run_name', run_id) if dag_run and dag_run.conf else run_id }} "
            "--golden_logits_path=/tmp/golden_logits.jsonl "
            "--max_kl_div={{ dag_run.conf.get('max_kl_div', params.get('max_kl_div', 0.02)) }} "
            "--atol=1e-02 "
            "--rtol=1e-02 "
            "{% set overrides = dag_run.conf.get('forward_pass_maxtext_overrides', params.get('forward_pass_maxtext_overrides', {})) %}"
            "--hf_model_path={{ dag_run.conf.get('hf_model_path', params.get('hf_model_path', overrides.get('hf_model_path', ''))) }} "
            "{% for k, v in overrides.items() %}"
            "{% if k != 'hf_model_path' %}{{ k }}=\"{{ v }}\" {% endif %}"
            "{% endfor %}"
            "{% if 'remat_policy' not in overrides %}remat_policy=none {% endif %}"
            "per_device_batch_size=1.0 "
        ),
    )

    job_test_config = DynamicTpuGkeTest(
        accelerator=test_config.Tpu(
            version=test_config.TpuVersion(str(tpu_version)),
            cores=tpu_cores,
            runtime_version="tpu-ubuntu2204-base",
            reserved=True,
        ),
        test_name="maxtext_forward_pass_validation",
        set_up_cmds=(
            "pip install --upgrade pip",
            "google-cloud-sdk/bin/gcloud components update --quiet",
        ),
        run_model_cmds=run_model_cmds,
        timeout=datetime.timedelta(minutes=time_out_in_min),
        task_owner="airflow",
        cluster_name="{{ dag_run.conf.get('xpk_cluster_name', params.get('xpk_cluster_name', ti.xcom_pull(task_ids='find_available_cluster') or 'v4-8-maxtext')) }}",
        docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-06",
        num_slices=1,
    )

    return task.XpkTask(
        task_test_config=job_test_config,
        task_gcp_config=job_gcp_config,
    )


def get_decoding_validation_task(
    cluster_config=None,
    tpu_version: str = None,
    tpu_cores: int = None,
    tpu_zone: str = None,
    tpu_project: str = None,
    time_out_in_min: int = 45,
) -> task.XpkTask:
    """
    Sub-DAG D: End-to-End Decoding / Text Generation Verification.
    Generates the XPK task configuration to run decode_validator.py on a TPU cluster.
    """
    if cluster_config is not None:
        tpu_version = cluster_config.device_version.value
        tpu_cores = cluster_config.core_count
        tpu_zone = f"{{{{ dag_run.conf.get('xpk_zone', params.get('xpk_zone', '{cluster_config.zone}')) }}}}"
        tpu_project = f"{{{{ dag_run.conf.get('xpk_project', params.get('xpk_project', '{cluster_config.project}')) }}}}"

    job_gcp_config = gcp_config.GCPConfig(
        project_name=tpu_project,
        zone=tpu_zone,
        dataset_name=metric_config.DatasetOption.XLML_DATASET,
        composer_project=tpu_project,
    )

    run_model_cmds = (
        "set -e",
        "export HF_TOKEN=\"{{ dag_run.conf.get('hf_token', params.get('hf_token', '')) }}\"",
        'if [[ "$HF_TOKEN" == "None" || -z "$HF_TOKEN" ]]; then unset HF_TOKEN; fi',
        # clone the MaxText repository dynamically using Airflow Jinja templating.
        "git clone https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        # check out a specific commit_hash if provided (for reproducible testing of PRs),
        # otherwise fallback to checking out the specified branch name, defaulting to 'main'.
        "cd /tmp/maxtext && git checkout {% set target_commit = dag_run.conf.get('maxtext_commit_hash') or params.get('maxtext_commit_hash') %}{% set target_branch = dag_run.conf.get('maxtext_branch') or params.get('maxtext_branch') or 'main' %}{{ var.value.get('OVERRIDE_BRANCH_' ~ (dag_run.conf.get('run_name', params.get('run_name', 'default_run'))), target_commit or target_branch) }}",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",
        "pip install torch --index-url https://download.pytorch.org/whl/cpu",
        (
            "cd /tmp/maxtext && python3 src/maxtext/experimental/agent/ckpt_validation_pipeline/decode_validator.py "
            "--report_gcs_dir={{ dag_run.conf.get('report_gcs_dir', params.get('report_gcs_dir', '')) | trim('/') }}/{{ dag_run.conf.get('run_name', run_id) if dag_run and dag_run.conf else run_id }} "
            "run_name={{ dag_run.conf.get('run_name', params.get('run_name', 'default_run')) }}_{{ run_id }} "
            "model_name={{ dag_run.conf.get('maxtext_model_name', params.get('maxtext_model_name', '')) }} "
            "load_parameters_path={{ dag_run.conf.get('checkpoint_gcs_path', params.get('checkpoint_gcs_path', '')) }} "
            "{% for k, v in dag_run.conf.get('decode_maxtext_overrides', params.get('decode_maxtext_overrides', {})).items() %}{{ k }}=\"{{ v }}\" {% endfor %}"
        ),
    )

    job_test_config = DynamicTpuGkeTest(
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
        cluster_name="{{ dag_run.conf.get('xpk_cluster_name', params.get('xpk_cluster_name', 'v4-8-maxtext')) }}",
        docker_image="gcr.io/tpu-prod-env-multipod/maxtext_jax_stable:2026-07-06",
        num_slices=1,
    )

    return task.XpkTask(
        task_test_config=job_test_config,
        task_gcp_config=job_gcp_config,
    )


# Backward-compatibility alias for legacy imports
get_maxtext_validation_config = get_decoding_validation_task


def check_upstream_failures(**context):
    """Guardrail task callable: raises AirflowFailException if any task in the DAG run failed."""
    from airflow.exceptions import AirflowFailException

    dag_run = context.get("dag_run")
    if not dag_run:
        return
    for ti in dag_run.get_task_instances():
        if (
            ti.state in ("failed", "upstream_failed")
            and ti.task_id != "check_upstream_failures"
        ):
            raise AirflowFailException(
                f"Task '{ti.task_id}' failed. Marking DAG as FAILED."
            )


def get_upstream_failure_validator_task(dag):
    """Returns a PythonOperator that runs with trigger_rule='all_done' to ensure DAG is marked failed if any upstream task failed."""
    from airflow.operators.python import PythonOperator

    return PythonOperator(
        task_id="check_upstream_failures",
        python_callable=check_upstream_failures,
        trigger_rule="all_done",
        dag=dag,
    )
