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
        # for testing, use a feature branch of maxtext that contains the mock tensor test script
        # "git clone -b feature/checkpoint-validation-clean https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        "git clone -b main https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
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
        # for testing, use a feature branch of maxtext that contains the mock tensor test script
        # "git clone -b feature/checkpoint-validation-clean https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        "git clone -b main https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",

        # dump ideal MaxText blueprint
        "python3 -m maxtext.checkpoint_conversion.inspect_checkpoint maxtext "
        f"model_name={model_name} scan_layers={scan_layers} --output_file=/tmp/ideal_raw.txt",

        # dump actual GCS Orbax checkpoint structures
        "python3 -m maxtext.checkpoint_conversion.inspect_checkpoint orbax "
        f"--path {checkpoint_gcs_path} --output_file=/tmp/actual_raw.txt",

        # filter for only the 'key:' lines
        "grep '^key:' /tmp/ideal_raw.txt > /tmp/ideal_shapes.txt",
        "grep '^key:' /tmp/actual_raw.txt > /tmp/actual_shapes.txt",

        # run the validation logic from the repo file
        "python3 /tmp/maxtext/src/maxtext/experimental/agent/checkpoint_validation_agent/checkpoint_shape_validator.py"
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
        # for testing, use a feature branch of maxtext that contains the mock tensor test script
        # "git clone -b feature/checkpoint-validation-clean https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        "git clone -b main https://github.com/AI-Hypercomputer/maxtext.git /tmp/maxtext",
        "cd /tmp/maxtext && pip install --no-cache-dir --no-deps -e .",
        "export PYTHONPATH=/tmp/maxtext/src:$PYTHONPATH",
        # pass the checkpoint path, model name, and dynamically unpack the overrides dictionary using Jinja
        (
            f"python3 /tmp/maxtext/src/maxtext/experimental/agent/checkpoint_validation_agent/mock_tensor_test.py "
            f"{checkpoint_path} {model_name} "
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
