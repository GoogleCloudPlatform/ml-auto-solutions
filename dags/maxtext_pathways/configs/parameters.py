from airflow.models.param import Param
from dags.maxtext_pathways.configs.model_configs import MaxTextV5eModelConfigs, MaxTextV5pModelConfigs, MaxTextTrilliumModelConfigs
from dags.common.vm_resource import TpuVersion

MODEL_FRAMEWORK = ['mcjax', 'pathways']

MODEL_NAME = []
V5E_MODEL_NAME = [config.value for config in MaxTextV5eModelConfigs]
V5P_MODEL_NAME = [config.value for config in MaxTextV5pModelConfigs]
V6E_MODEL_NAME = [config.value for config in MaxTextTrilliumModelConfigs]
MODEL_NAME.extend(V5E_MODEL_NAME)
MODEL_NAME.extend(V5P_MODEL_NAME)
MODEL_NAME.extend(V6E_MODEL_NAME)

DEVICE_VERSION = ['v' + version.value for version in TpuVersion]

PARAMETERS = {
    'user': Param(
        'username', type='string', title='User', description='User name.'
    ),
    'cluster_name': Param(
        'pw-scale-test-v5e-32',
        type='string',
        title='Cluster Name',
        description='GKE/GCP cluster name.',
    ),
    'project': Param(
        'cloud-tpu-multipod-dev',
        type='string',
        title='Project',
        description='GCP project ID.',
    ),
    'zone': Param(
        'us-south1-a', type='string', title='Zone', description='GCP zone.'
    ),
    'device_version': Param(
        'v5litepod',
        type='string',
        title='Device Version',
        description='Device type for the cluster. ex: "v5litepod"-32',
        enum=DEVICE_VERSION,
    ),
    'core_count': Param(
        32,
        type='integer',
        title='Core Count',
        description='Device core count for the cluster. ex: v5litepod-"32"',
    ),
    'service_account': Param(
        'one-click@cloud-tpu-multipod-dev.iam.gserviceaccount.com',
        type='string',
        title='Service account',
        description='Service account of the project.',
    ),
    'benchmark_steps': Param(
        20,
        type='integer',
        title='Benchmark Steps',
        description='Number of benchmark steps.',
    ),
    'num_slices_list': Param(
        2,
        type='string',
        title='Num Slices List',
        description='List of number of slices. ex: "2,4"',
    ),
    'server_image': Param(
        'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:latest',
        type='string',
        title='Server Image',
        description='Server image for the cluster.',
    ),
    'proxy_image': Param(
        'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:latest',
        type='string',
        title='Proxy Image',
        description='Proxy image for the cluster.',
    ),
    'runner': Param(
        'gcr.io/tpu-prod-env-one-vm/chzheng_latest:latest',
        type='string',
        title='Runner Image',
        description='Runner image for the cluster.',
    ),
    'selected_model_framework': Param(
        MODEL_FRAMEWORK[1],
        type='string',
        title='Model Framework',
        description='Select a model framework to run.',
        enum=MODEL_FRAMEWORK,  # dropdown menu
    ),
    'selected_model_names': Param(
        'llama3_1_8b_8192_v5e_256',
        type='string',
        title='Model Name',
        description='Select a model name to run.',
        enum=MODEL_NAME,
    ),
    'priority': Param(
        'medium',
        type='string',
        title='Priority',
        description='Priority for the workload',
        enum=['very high', 'high', 'medium', 'low'],
    ),
    'max_restarts': Param(
        1,
        type='integer',
        title='Max Restarts',
        description='Max restarts for the workload',
    ),
    'time_out_in_min': Param(
        60,
        type='integer',
        title='Time Out In Minutes',
        description='Time out in minutes for the workload task, adjust it when your meet (airflow.exceptions.AirflowSensorTimeout: Sensor has timed out).',
    ),
}
