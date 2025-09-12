import datetime
from airflow.models.dag import DAG
from dags.maxtext_pathways.configs.parameters import PARAMETERS
from dags.maxtext_pathways.utils.tasks import build_recipe_command, run_workload, wait_for_workload_completion, clean_up_workload


with DAG(
  dag_id="pw_mcjax_benchmark_recipe_dag",
  start_date=datetime.datetime(2025, 1, 1),
  schedule_interval=None,
  catchup=False,
  tags=[
    'maxtext',
    'pathways',
    'mcjax',
    'benchmark'
    'nightly',
  ],
  description="A DAG to run a MaxText pw_mcjax_benchmark_recipe on GKE.",
  params=PARAMETERS
) as dag:
  # Define task dependencies by instantiating and linking tasks.
  build_recipe_command_task = build_recipe_command()
  run_workload_task = run_workload(build_recipe_command_task)
  wait_for_workload_completion_task = wait_for_workload_completion()
  clean_up_workload_task = clean_up_workload()

  # Set the execution order.
  (
    build_recipe_command_task
    >> run_workload_task
    >> wait_for_workload_completion_task
    >> clean_up_workload_task
  )