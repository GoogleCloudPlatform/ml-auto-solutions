import logging
import os

from airflow.exceptions import AirflowException
from airflow.listeners import hookimpl
from airflow.models import DagRun, TaskInstance
from airflow.plugins_manager import AirflowPlugin
from airflow.providers.github.hooks.github import GithubHook
from github import Github

from dags.common.vm_resource import Project
from xlml.utils import composer
from urllib import parse

_PROJECT_ID = Project.CLOUD_ML_AUTO_SOLUTIONS
_REPO_NAME = "GoogleCloudPlatform/ml-auto-solutions"


def generate_dag_run_link(
    proj_id: str,
    dag_id: str,
    dag_run_id: str,
    task_id: str,
):
  airflow_link = composer.get_airflow_url(
      proj_id,
      os.environ.get("COMPOSER_LOCATION"),
      os.environ.get("COMPOSER_ENVIRONMENT"),
  )
  airflow_dag_run_link = (
      f"{airflow_link}/dags/{dag_id}/"
      f"grid?dag_run_id={parse.quote(dag_run_id)}&task_id={task_id}&tab=logs"
  )
  return airflow_dag_run_link


"""
Airflow Listener class to handle DAG run failures.

This listener specifically triggers actions when a DAG run fails.
It checks for the 'on_failure_alert' tag on the failed DAG. If the tag is present,
it proceeds to create a GitHub issue with details about the failed DAG run
and its failed tasks. The issue is assigned to the owners of the failed tasks
(excluding 'airflow' as an owner).

TODO: Implement more sophisticated issue filing strategies beyond a single failure, such as:
-   Consecutive Failures: Only file an issue if a DAG has failed for two or more
    consecutive runs to reduce noise from transient issues.
-   Reduced Pass Rate: File an issue if the current run failed AND the overall
    pass rate of the past N (e.g., 10) runs for this DAG falls below a certain threshold
    (e.g., 50%).
"""


class DagRunListener:

  def __init__(self):
    self.log_prefix = self.__class__.__name__

  @hookimpl
  def on_dag_run_success(self, dag_run: DagRun, msg: str):
    self.on_dag_finished(dag_run, msg)

  @hookimpl
  def on_dag_run_failed(self, dag_run: DagRun, msg: str):
    self.on_dag_finished(dag_run, msg)

  def on_dag_finished(self, dag_run: DagRun, msg: str):
    logging.info(f"[{self.log_prefix}] DAG run: {dag_run.dag_id} finished")
    logging.info(f"[{self.log_prefix}] msg: {msg}")

    try:
      # Only DAGs with the 'on_failure_alert' tag will be processed.
      if "on_failure_alert" not in dag_run.dag.tags:
        logging.info(
            f"[{self.log_prefix}] DAG {dag_run.dag_id} isn't "
            f"'on_failure_alert' by tags. Return"
        )
        return
      logging.info(
          f"[{self.log_prefix}] DAG run {dag_run.dag_id} is 'on_failure_alert'"
      )

      failed_task_instances = [
          ti for ti in dag_run.task_instances if ti.state == "failed"
      ]
      if len(failed_task_instances) == 0:
        logging.info(
            f"[{self.log_prefix}] No failed tasks, GitHub Issue operation completed."
        )
        return

      body = (
          f"- **Run ID**: {dag_run.run_id}\n"
          f"- **Execution Date**: {dag_run.execution_date}\n"
      )
      group_dict = {}
      for task_instance in failed_task_instances:
        group_id = self.get_group_id(task_instance)
        if group_id in group_dict:
          group_dict[group_id].append(task_instance)
        else:
          group_dict[group_id] = [task_instance]

      client = self.get_github_client()
      for group_id, task_instances in group_dict.items():
        title = f"[{self.log_prefix}] {dag_run.dag_id} {group_id} failed"
        assignees = set()
        issue_body = body
        for task_instance in task_instances:
          link = generate_dag_run_link(
              proj_id=str(_PROJECT_ID),
              dag_id=dag_run.dag_id,
              dag_run_id=dag_run.run_id,
              task_id=task_instance.task_id,
          )
          issue_body += (
              f"- **Failed Task**: [{task_instance.task_id}](" f"{link})\n"
          )
          if task_instance.task.owner and task_instance.task.owner != "airflow":
            assignees.add(task_instance.task.owner)

        issue = self.query_latest_issues(client, title)
        try:
          if issue:
            self.add_issue_comment(issue, issue_body)
          else:
            self.create_issue(client, title, issue_body, list(assignees))
        except Exception as e:
          if "422" not in str(e):  # Invalid GitHub username as assignees
            raise e
          logging.error(
              f"[{self.log_prefix}] Invalid assignees, retrying without assignees. Original error: {e}"
          )
          if issue:
            self.add_issue_comment(issue, issue_body)
          else:
            self.create_issue(client, title, issue_body)

      logging.error(f"[{self.log_prefix}] GitHub Issue operation completed.")
    except AirflowException as airflow_e:
      logging.error(
          f"[{self.log_prefix}] Airflow exception: {airflow_e}", exc_info=True
      )
    except Exception as e:
      logging.error(
          f"[{self.log_prefix}] Unexpected exception: {e}", exc_info=True
      )

  @staticmethod
  def get_group_id(task_instance: TaskInstance):
    task = task_instance.task
    task_id = task.task_id
    group_id = task.task_group.group_id
    if group_id:
      # Benchmark ID would be the first section of group_id
      return group_id.split(".")[0]
    else:
      return task_id

  def get_github_client(self) -> Github:
    environment_name = os.environ.get(
        "COMPOSER_ENVIRONMENT", default="ml-automation-solutions"
    )
    logging.error(f"[{self.log_prefix}] env: {environment_name}")
    conn_id = environment_name + "-github_default"
    return GithubHook(github_conn_id=conn_id).get_conn()

  def query_latest_issues(self, client: Github, title: str):
    logging.error(f"[{self.log_prefix}] query open issues titled {title}")
    query_issues = f"{title} in:title state:open repo:{_REPO_NAME} is:issue"
    issues = list(
        client.search_issues(query=query_issues, sort="updated", order="desc")
    )
    return (
        sorted(issues, key=lambda i: i.updated_at, reverse=True)[0]
        if (len(issues) > 0)
        else None
    )

  def create_issue(self, client, title, body, assignees=None):
    if not assignees:
      assignees = []
    logging.error(f"[{self.log_prefix}] Create a new one")
    repo = client.get_repo(full_name_or_id=_REPO_NAME)
    repo.create_issue(title=f"{title}", body=f"{body}", assignees=assignees)

  def add_issue_comment(self, issue, body):
    logging.error(f"[{self.log_prefix}] Update the latest one")
    issue.create_comment(body=body)


class ListenerPlugins(AirflowPlugin):
  name = "listener_plugins"
  listeners = [DagRunListener()]
