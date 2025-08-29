from airflow.decorators import dag, task # Use decorator style to define DAG / Task
from airflow.models.param import Param # define DAG parameters
from airflow.operators.python import get_current_context # Get the execution context (including dag_run.conf and params)
from airflow.utils.dates import days_ago
from dags.maxtext_pw.configs.user_configs import UserConfig
from dags.maxtext_pw.configs.model_configs import MODEL_FRAMEWORK, MODEL_NAME


# Use the @dag decorator to declare a DAG
@dag(
  dag_id='dag_ui_params_demo',
  start_date=days_ago(1), # set in the 'past' to avoid backfilling
  schedule='45 6 * * *', # UTC+0: (6:45), UTC+8: (14:45)
  catchup=False, # Don't execute missed past schedules,
  default_view='graph',
  description='Demo: Using user configs with UI params.',
  # Define the parameter form to be displayed in the UI.
  params={
    'user': Param(
      'user_name',
      type='string',
      title='User',
      description='User name.'
    ),
    'cluster_name': Param(
      'test-v5e-32-cluster',
      type='string',
      title='Cluster Name',
      description='GKE/GCP cluster name.'
    ),
    'project': Param(
      'cloud-tpu-cluster',
      type='string',
      title='Project',
      description='GCP project ID.'
    ),
    'zone': Param(
      'us-south1-a',
      type='string',
      title='Zone',
      description='GCP zone.'
    ),
    'device_type': Param(
      'v5litepod-32',
      type='string',
      title='Device Type',
      description='Device type for the cluster.'
    ),
    'server_image': Param(
      'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:latest',
      type='string',
      title='Server Image',
      description='Server image for the cluster.'
    ),
    'proxy_image': Param(
      'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:latest',
      type='string',
      title='Proxy Image',
      description='Proxy image for the cluster.'
    ),
    'runner': Param(
      'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest',
      type='string',
      title='Runner Image',
      description='Runner image for the cluster.'
    ),
    'colocated_python_image': Param(
      'gcr.io/cloud-tpu-v2-images-dev/colocated_python_sidecar_latest:latest',
      type='string',
      title='Colocated Python Image',
      description='Colocated Python image for the cluster.'
    ),
    'benchmark_steps': Param(
      20,
      type='integer',
      title='Benchmark Steps',
      description='Number of benchmark steps.'
    ),
    'num_slices_list': Param(
      [2, 4],
      type='array',
      title='Num Slices List',
      description='List of number of slices.'
    ),
    'selected_model_framework': Param(
      MODEL_FRAMEWORK[0],
      type='string',
      title='Model Framework',
      description='Select a model framework to run.',
      enum=MODEL_FRAMEWORK # dropdown menu
    ),
    'selected_model_names': Param(
      MODEL_NAME[0],
      type='string',
      title='Model Name',
      description='Select a model name to run.',
      enum=MODEL_NAME
    ),
  }
  )
def test_recipes():
  @task
  def get_user_config():
    '''read and print the input and config parameters'''

    # Get the execution context, which contains the DAG run's configuration and parameters.
    ctx = get_current_context()
    params = ctx.get('params', {})

    # Access the parameters passed from the UI
    print(f'{"=" * 30} UI Dag parameters {"=" * 30}')
    for key, value in params.items():
      print(f'{key}: {value}')
    
    # create an UserConfig object
    user_config = UserConfig(
      user=params['user'],
      cluster_name=params['cluster_name'],
      project=params['project'],
      zone=params['zone'],
      device_type=params['device_type'],
      server_image=params['server_image'],
      proxy_image=params['proxy_image'],
      runner=params['runner'],
      colocated_python_image=params['colocated_python_image'],
      benchmark_steps=params['benchmark_steps'],
      num_slices_list=params['num_slices_list'],
      selected_model_framework=[params['selected_model_framework']],
      selected_model_names=[params['selected_model_names']]
    )

    # Access the generated attributes
    print(f'{"=" * 30} UserConfig parameters {"=" * 30}')
    for key, value in user_config.__dict__.items():
      print(f'{key}: {value}')

    return user_config
  
  @task
  def recipe_A(user_config: UserConfig):
    '''run a recipe'''

    print('Running recipe - A.')
    print('...')
    print('Finished recipe - A.')

  # Add the task to the DAG
  user_config = get_user_config()
  recipe_A(user_config)

# call function to create a DAG instance that the scheduler can load
dag = test_recipes()