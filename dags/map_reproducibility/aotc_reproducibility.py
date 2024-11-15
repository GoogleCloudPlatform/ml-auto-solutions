"Bash helper commands for AOTC artifacts"

import os


def set_variables_cmds():
  set_variables = (
      "export PROJECT=supercomputer-testing",
      "export CLUSTER=a3plus-benchmark",
      "export CLUSTER_REGION=australia-southeast1",
      "NOW=$(date +%s)",
      "BUCKET_NAME=regression-testing-xlml",
      "export JOB_NAME=gpt3-xlml-$NOW-175b-nemo",
  )
  return set_variables


def set_project_commands():
  set_project_command = (
      "gcloud config set project $PROJECT",
      "sudo chown -R airflow:airflow /home/airflow/composer_kube_config",
      "gcloud container clusters get-credentials "
      "$CLUSTER --region $CLUSTER_REGION",
  )
  return set_project_command


def install_helm_cmds():
  install_helm_cmd = (
      "curl -fsSL -o get_helm.sh "
      "https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3",
      "chmod 700 get_helm.sh",
      "./get_helm.sh",
  )
  return install_helm_cmd


# By default the composer environment overwrites the
# namespaces to airflow namespaces.
# In order to prevent that it is necessary explicitly
# change the namespace to default.
def namespace_cmds():
  namespace = (
      "kubectl config view | grep namespace",
      "kubectl config set-context --current --namespace=default",
      "kubectl config set-context heml --namespace=default",
      "kubectl get pods --namespace=default",
  )
  return namespace


def wait_for_jobs_cmds():
  wait_for_job = (
      "echo 'will wait for job to start running'",
      "kubectl wait --for=condition=running job/$JOB_NAME"
      " --namespace=default --timeout=10m",
      "echo 'will wait for jobs to finish'",
      "kubectl wait --for=condition=complete "
      "job/$JOB_NAME --namespace=default --timeout=100m",
  )
  return wait_for_job


def copy_bucket_cmds():
  copy_bucket_contents = (
      "COMPLETE_JOB_NAME=$(gcloud storage ls "
      "gs://$BUCKET_NAME/nemo-experiments/ | grep $JOB_NAME)",
      "echo 'copying from' ",
      "echo $COMPLETE_JOB_NAME",
      "cd $REPO_ROOT/src/utils/training_metrics",
      "gcloud storage cp ${COMPLETE_JOB_NAME}"
      "dllogger/rank-0/dllogger.json .",
  )
  return copy_bucket_contents


def get_metrics_cmds():
  # TODO(gunjanj007): get these parameters from the recipe
  get_metrics = (
      "python3 process_training_results.py --file"
      " dllogger.json --batch_size 2048 "
      "--num_accelerators 256 "
      "--precision fp8  "
      "--model_type gpt3-175b "
      "--accelerator_type h100 ",
  )
  return get_metrics


def cleanup_cmds():
  cleanup = (
      "kubectl get pods "
      "--no-headers=true | awk '{print $1}' "
      "| grep $JOB_NAME |  xargs kubectl delete pods",
      "helm uninstall $JOB_NAME",
  )
  return cleanup
