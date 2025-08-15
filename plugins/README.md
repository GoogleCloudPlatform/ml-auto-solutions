The "on_failure_actions" plugin listens to the DAGs run results and takes action for failed DAG runs. Currently, the action is to file a GitHub issue for the failed DAG run.

## Pre-requisites:
To leverage the "on_failure_actions" plugin, ensure the following conditions are met:

### 1.  **DAG Opt-In:**
#### Plugin Activation Instructions

By default, this plugin is **disabled** for all DAGs.

##### Enabling the Plugin for a DAG

If you want your DAG to **trigger this plugin**, add your DAG ID as a new line in **plugins/allow_list.txt**.

##### Disabling the Plugin for a DAG

To prevent your DAG from ever triggering this plugin, add your DAG ID as a new line in **plugins/block_list.txt**. This list is intended to avoid repeated issue postings for DAGs that are not currently resolvable.

> **Note:** Each DAG ID should be placed on its own line, without extra spaces or quotes.

> **Note:** The priority of block list is higher than the one of allow list.

##### Example

```
<DAG_ID 1>
<DAG_ID 2>
...
```

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

### 2.5 **GitHub Owner Mapping for Test Config:**
If you are using test_config.py in your DAG, you should fill test_owner attribute with your GitHub username

#### Example test definition
    maxtext_v4_configs_test = gke_config.get_gke_config(
        num_slices=slice_num,
        cluster=clusters[accelerator],
        time_out_in_min=60,
        test_name=f"maxtext-checkpointing-{mode.value}-{chkpt_mode}",
        run_model_cmds=command,
        docker_image=image.value,
        test_owner='github_username_here',  # This should be a valid GitHub username
    ).run()

## Configuration and Installation:
### Create a GitHub APP
Refer to this [procedure](https://docs.github.com/en/apps/creating-github-apps/registering-a-github-app/registering-a-github-app) to register a GitHub App.

- Give the app a unique name, for example "ml-auto-solutions-app", and record it as "APP_ID". 
- Choose "Issues" for the "Repository permissions".
- Install it to the Target GitHub Repo (ml-auto-solutions). Record the "INSTALLATION_ID" from the web URL. For example, if the URL is https://github.com/apps/ml-auto-solutions-app/installations/78297659, then 78297659 is the INSTALLATION_ID.
- Create a private key and record it as PRIVATE_KEY.

The APP_ID, INSTALLATION_ID, and PRIVATE_KEY will be used in the next steps.

### Set up via GCP console UI
1. From GCP console UI, Your Composer Env -> Tab -> Pypi packages -> Edit -> Add 'apache-airflow-providers-github' -> Save

2. From GCP console UI, search for "Secret Manager". Go to this page and click '+ CREATE SECRET' button.
   1. Name: airflow-connections-<composer_environment_name>-github_app
      Secret value:
      ```  
      {
          "app_id": "<APP_ID>",
          "installation_id": "<INSTALLATION_ID>",
          "private_key": "<PRIVATE_KEY>"
      }
      ```

3. Upload 'on_failure_actions.py' to \<DAG Bucket\>/plugins/

### Set up via Google Cloud CLI
1. Set the environment variables in your terminal:
    ```
    COMPOSER_ENVIRONMENT_NAME=<Composer Environment Name>
    COMPOSER_LOCATION=<Composer Location>
    COMPOSER_DAG_BUCKET=gs://<DAG Bucket>
    APP_ID=<GitHub App Id>
    INSTALLATION_ID=<GitHub Installation Id>
    PRIVATE_KEY=<GitHub App Private Key>
    ```

2. Add `apache-airflow-providers-github` Pypi package to your Composer environment.
    ```
    gcloud composer environments update ${COMPOSER_ENVIRONMENT_NAME} --location=${COMPOSER_LOCATION} --update-pypi-package='apache-airflow-providers-github'
    ```

3. Add a new Secret into Secret Manager for accessing GitHub App.
    ```
    echo "{"app_id": "${APP_ID}", "installation_id": "${INSTALLATION_ID}", "private_key": "${PRIVATE_KEY}"}" | gcloud secrets create airflow-connections-${COMPOSER_ENVIRONMENT_NAME}-github_app --data-file=-
    ```

4. Upload Composer plugins to the folder for dags in your Composer bucket
    ```
    gcloud storage cp ./on_failure_actions.py ${COMPOSER_DAG_BUCKET}/plugins/
    ```


