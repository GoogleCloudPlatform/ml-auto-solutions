import json
import logging
import os
import time
from typing import List

import google.auth.transport.requests
import jwt
import requests
from airflow.exceptions import AirflowException
from airflow.listeners import hookimpl
from airflow.models import DagRun, TaskInstance
from airflow.plugins_manager import AirflowPlugin
from github import Github, Auth
from github.Issue import Issue
from google.cloud import secretmanager

from urllib import parse

PROJECT_ID = "cloud-ml-auto-solutions"
REPO_NAME = "GoogleCloudPlatform/ml-auto-solutions"
SECRET_MANAGER = (
    "airflow-connections-"
    + os.environ.get("COMPOSER_ENVIRONMENT", default="ml-automation-solutions")
    + "-github_app"
)


def get_github_client() -> Github:
  secret_value = fetch_json_secret()
  app_id = secret_value["app_id"]
  installation_id = secret_value["installation_id"]
  private_key = secret_value["private_key"]
  installation_token = get_installation_token(
      app_id, installation_id, private_key
  )
  return Github(auth=Auth.Token(installation_token))


def fetch_json_secret():
  """
  Fetches the latest version of a secret from Google Secret Manager,
  decodes it from UTF-8, and parses it as a JSON object.
  Assumes the secret is stored in JSON format.
  Returns:
      dict: The parsed JSON secret as a Python dictionary.
  """
  client = secretmanager.SecretManagerServiceClient()
  secret_path = (
      f"projects/{PROJECT_ID}/secrets/{SECRET_MANAGER}/versions/latest"
  )
  response = client.access_secret_version(request={"name": secret_path})
  secret_str = response.payload.data.decode("UTF-8")
  secret_dict = json.loads(secret_str)
  return secret_dict


def get_installation_token(
    github_app_id: str, installation_id: str, private_key: str
):
  """
  Retrieves a GitHub App installation access token using the App's ID and private key.

  This token is used to authenticate API requests on behalf of the GitHub App installation.
  Internally, it first generates a short-lived JWT for the app, then exchanges it
  for an installation access token.

  Args:
      github_app_id (str): The GitHub App's numeric identifier (found in the app's settings).
      installation_id (str): The installation ID for a specific GitHub organization or repository.
      private_key (str): The PEM-formatted private key associated with the GitHub App.

  Returns:
      str: The installation access token.
  """
  jwt_token = generate_jwt(github_app_id, private_key)
  token_url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
  headers = {
      "Authorization": f"Bearer {jwt_token}",
      "Accept": "application/vnd.github+json",
  }
  resp = requests.post(token_url, headers=headers)
  resp.raise_for_status()
  return resp.json()["token"]


def generate_jwt(github_app_id: str, private_key: str) -> str:
  now = int(time.time())
  jwt_payload = {"iat": now - 60, "exp": now + (10 * 60), "iss": github_app_id}
  return jwt.encode(jwt_payload, private_key, algorithm="RS256")


def query_latest_issues(client: Github, title: str):
  query_issues = f"{title} in:title state:open repo:{REPO_NAME} is:issue"
  issues = list(
      client.search_issues(query=query_issues, sort="updated", order="desc")
  )
  return (
      sorted(issues, key=lambda i: i.updated_at, reverse=True)[0]
      if (len(issues) > 0)
      else None
  )


def add_comment_or_create_issue(
    client: Github,
    issue: Issue,
    title: str,
    issue_body: str,
    assignees: List[str] = None,
):
  if not assignees:
    assignees = []
  if issue:
    logging.error(f"[DagRunListener] Update the latest one")
    create_comment(issue, issue_body)
  else:
    logging.error(f"[DagRunListener] Create a new one")
    create_issue(client, REPO_NAME, title, issue_body, assignees)


def create_issue(client, repo, title, body, assignees=None):
  if not assignees:
    assignees = []
  repo = client.get_repo(full_name_or_id=repo)
  repo.create_issue(title=f"{title}", body=f"{body}", assignees=assignees)


def create_comment(issue, body):
  issue.create_comment(body=body)


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
            f"[{self.log_prefix}] DAG {dag_run.dag_id} isn't enabled 'on_failure_alert' by tags. Return"
        )
        return
      logging.info(
          f"[{self.log_prefix}] DAG run {dag_run.dag_id} is enabled 'on_failure_alert'"
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
      test_name_dict = {}
      for task_instance in failed_task_instances:
        test_name = DagRunListener.get_test_name(task_instance)
        if test_name in test_name_dict:
          test_name_dict[test_name].append(task_instance)
        else:
          test_name_dict[test_name] = [task_instance]

      for test_name, task_instances in test_name_dict.items():
        title = f"[{self.log_prefix}] {dag_run.dag_id} {test_name} failed"
        assignees = set()
        issue_body = body
        for task_instance in task_instances:
          link = self.generate_dag_run_link(
              proj_id=str(PROJECT_ID),
              dag_id=dag_run.dag_id,
              dag_run_id=dag_run.run_id,
              task_id=task_instance.task_id,
          )
          issue_body += (
              f"- **Failed Task**: [{task_instance.task_id}](" f"{link})\n"
          )
          if task_instance.task.owner and task_instance.task.owner != "airflow":
            assignees.add(task_instance.task.owner)

        client = get_github_client()
        issue = query_latest_issues(client, title)
        try:
          add_comment_or_create_issue(
              client, issue, title, issue_body, list(assignees)
          )
        except Exception as e:
          if "422" not in str(e):  # Invalid GitHub username as assignees
            raise e
          logging.error(
              f"[{self.log_prefix}] Invalid assignees, retrying without assignees. Original error: {e}"
          )
          add_comment_or_create_issue(client, issue, title, issue_body)

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
  def get_test_name(task_instance: TaskInstance):
    task = task_instance.task
    task_id = task.task_id
    if task.task_group.group_id:
      # Benchmark ID would be the first section of group_id
      return task.task_group.group_id.split(".")[0]
    else:
      return task_id

  @staticmethod
  def generate_dag_run_link(
      proj_id: str,
      dag_id: str,
      dag_run_id: str,
      task_id: str,
  ):
    airflow_link = DagRunListener.get_airflow_url(
        proj_id,
        os.environ.get("COMPOSER_LOCATION"),
        os.environ.get("COMPOSER_ENVIRONMENT"),
    )
    airflow_dag_run_link = (
        f"{airflow_link}/dags/{dag_id}/"
        f"grid?dag_run_id={parse.quote(dag_run_id)}&task_id={task_id}&tab=logs"
    )
    return airflow_dag_run_link

  @staticmethod
  def get_airflow_url(project: str, region: str, env: str) -> str:
    """Get Airflow web UI.

    Args:
     project: The project name of the composer.
     region: The region of the composer.
     env: The environment name of the composer.

    Returns:
    The URL of Airflow.
    """
    request_endpoint = (
        "https://composer.googleapis.com/"
        f"v1beta1/projects/{project}/locations/"
        f"{region}/environments/{env}"
    )
    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    headers = {"Authorization": f"Bearer {creds.token}"}
    response = requests.get(request_endpoint, headers=headers)
    configs = response.json()
    return configs["config"]["airflowUri"]


class ListenerPlugins(AirflowPlugin):
  name = "listener_plugins"
  listeners = [DagRunListener()]
