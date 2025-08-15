import logging

from airflow import models

from airflow.decorators import task
from airflow.exceptions import AirflowException
from airflow.utils.task_group import TaskGroup

from dags.common import test_owner

"""
A DAG that always fails when triggered. This DAG is an example DAG used to
trigger the on_failure_actions.py/DagRunListener plugin and post a Github
Issue.
"""


@task
def task_a():
  logging.info("task A")


# Add or override the owner of the task, in order to assign issue assignees.
@task(owner=test_owner.AIRFLOW)
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
    ],
    catchup=False,
    default_args={
        "retries": 0,  # Set to 0 for throwing exception immediately
    },
) as dag:
  with TaskGroup(group_id="Test1") as group_a:
    with TaskGroup(group_id="Subgroup1") as group_b:
      taskA = task_a()

  with TaskGroup(group_id="Test2") as group_c:
    with TaskGroup(group_id="Subgroup2") as group_d:
      taskB = task_b()

  with TaskGroup(group_id="Test3") as group_e:
    with TaskGroup(group_id="Subgroup3") as group_f:
      taskC = task_c()

  group_a >> group_c >> group_e
