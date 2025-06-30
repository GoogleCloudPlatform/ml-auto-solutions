import logging

from airflow import models

from airflow.decorators import task
from airflow.exceptions import AirflowException


"""
A DAG that always fails when triggered. This DAG is an example DAG used to
trigger the on_failure_actions.py/DagRunListener plugin and post a Github
Issue.
"""


@task
def task_a():
  logging.info("task A")


# Add or override the owner of the task, in order to assign issue assignees.
@task(owner="severus-ho")
def task_b():
  logging.info("task B")
  raise AirflowException("task B failed")


@task
def task_c():
  logging.info("task C")


with models.DAG(
    dag_id="on_failure_actions_trigger",
    schedule=None,
    tags=[
        "test_dag",
        "on_failure_alert",  # Add this to enable DagRunListener
    ],
    catchup=False,
    default_args={
        "retries": 0,  # Set to 0 for throwing exception immediately
    },
) as dag:
  taskA = task_a()
  taskB = task_b()
  taskC = task_c()
  taskA >> taskB >> taskC
