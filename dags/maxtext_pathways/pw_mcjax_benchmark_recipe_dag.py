import datetime
from airflow.models.dag import DAG
from dags.maxtext_pathways.configs.parameters import PARAMETERS
from dags.maxtext_pathways.utils.tasks import build_recipe_command, run_workload, wait_for_workload_completion, clean_up_workload


with DAG(
  dag_id='pw_mcjax_benchmark_recipe_dag',
  start_date=datetime.datetime(2025, 1, 1),
  schedule_interval=None,
  catchup=False,
  default_args = {
      'retries': 0,
  },
  tags=[
      'maxtext',
      'pathways',
      'mcjax',
      'benchmark',
      'nightly',
  ],
  description='A DAG to run a MaxText pw_mcjax_benchmark_recipe on GKE.',
  params=PARAMETERS,
  doc_md="""
  # A DAG to run a MaxText pw_mcjax_benchmark_recipe on GKE.

  # Description
  Specify different models and number of slices to test the MaxText pw_mcjax_benchmark_recipe on different clusters.  
  The DAG first generates recipe command through UI parameters, then runs the workload, waits and monitors the workload logs, and finally cleans up the workload.

  ### Prerequisites
  This test requires an existing cluster.  
  This test requires that a bucket with the same name as the UI parameter "[User]-[Region]" exists in the UI parameter [Project].

  ### Procedures
  An Airflow Composer environment must be created, and the required DAG code must be deployed to the associated GCS bucket.  
  To initiate the recipe, the user must access the Airflow UI, locate the specific DAG, and trigger its execution.
  """
) as dag:
  # Define task dependencies by instantiating and linking tasks.
  build_recipe_command_task = build_recipe_command()
  run_workload_task = run_workload(build_recipe_command_task)
  wait_for_workload_completion_task = wait_for_workload_completion()
  clean_up_workload_task = clean_up_workload().as_teardown(setups=run_workload_task)

  # Set the execution order.
  (
    build_recipe_command_task
    >> run_workload_task
    >> wait_for_workload_completion_task
    >> clean_up_workload_task
  )
  