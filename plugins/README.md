The "on_failure_actions" plugin listens to the DAGs run results and takes action for failed DAG runs. Currently, the action is to file a GitHub issue for the failed DAG run.

## Pre-requisites:
To leverage the "on_failure_actions" plugin, ensure the following conditions are met:

### 1.  **DAG Opt-In:** 
Each DAG intended to utilize this feature **must include the `"on_failure_alert"` tag** within its DAG definition. DAGs without this specific tag will be ignored by the plugin's failure-handling logic, and no GitHub issue will be filed for their failures.

    with DAG(
        dag_id='my_critical_dag',
        # ... other DAG parameters ...
        tags=['data_ingestion', 'critical', 'on_failure_alert'], # <--- Add this tag
    ) as dag:
        # ... tasks ...

### 2.  **GitHub Owner Mapping:** 
For accurate issue assignment, ensure that the `owner` property defined for tests within your DAGs corresponds directly to valid **GitHub usernames**. The plugin will collect unique test owners from the failed DAG and attempt to assign the GitHub issue to these users.

#### Example task definition
    my_task = BashOperator(
        task_id='process_data',
        bash_command='...',
        owner='github_username_here', # This should be a valid GitHub username
    )

    # Or

    @task(owner='github_username_here') # This should be a valid GitHub username
    def task_a():
        pass

## Configuration and Installation:
1. From GCP console UI, Your Composer Env -> Tab -> Pypi packages -> Edit -> Add 'apache-airflow-providers-github' -> Save

2. From GCP console UI, search for "Secret Manager", and add conn_id 'github_default' into Secret Manager.
   1. key: airflow-connections-<composer_environment_name>-github_default
   2. value:
       {
           "conn_type": "http",
           "host": "https://api.github.com",
           "password": "\<GitHub Personal Access Token\>"
       }

3. Composer -> Airflow configuration overrides -> Edit, reference: https://cloud.google.com/composer/docs/composer-1/configure-secret-manager
   1. secrets | backend | airflow.providers.google.cloud.secrets.secret_manager.CloudSecretManagerBackend
   2. secrets | backends_order | custom,environment_variable,metastore

4. Upload 'on_failure_actions.py' to \<DAG Bucket\>/plugins/