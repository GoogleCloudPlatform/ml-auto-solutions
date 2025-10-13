COMMAND_INSTALL_MAKE = [
    "sudo apt-get update",
    "sudo apt-get install -y make",
]

COMMAND_INSTALL_KUBECTL = [
    "grep -rhE ^deb /etc/apt/sources.list* | grep 'cloud-sdk'",
    "sudo apt-get install -y kubectl",
    "kubectl version --client",
]

COMMAND_INSTALL_DOCKER = [
    "sudo apt-get install ca-certificates curl",
    "sudo install -m 0755 -d /etc/apt/keyrings",
    "sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc",
    "sudo chmod a+r /etc/apt/keyrings/docker.asc",
    'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null',
    "sudo apt-get update",
    "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
    "sudo /usr/bin/dockerd & sleep 0",
    "while ! sudo docker info > /dev/null 2>&1; do echo 'Waiting for Docker to start...'; sleep 2; done",
]


COMMAND_INSTALL_GCLOUD = [
    "curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz",
    "tar -xf google-cloud-cli-linux-x86_64.tar.gz",
    "./google-cloud-sdk/install.sh",
    "./google-cloud-sdk/bin/gcloud init",
]

COMMAND_SWITCH_SERVICE_ACCOUNT = [
    "mkdir -p /root/keys",
    'gcloud secrets versions access "latest" --secret="one-click-key" > /root/keys/sa_key.json',
    "gcloud config set account {service_account}",
    "gcloud auth activate-service-account {service_account} --key-file=/root/keys/sa_key.json",
]

COMMAND_INSTALL_KUBECTL_KJOB = [
    "curl -Lo ./kubectl-kjob https://github.com/kubernetes-sigs/kjob/releases/download/v0.1.0/kubectl-kjob-linux-amd64",
    "chmod +x ./kubectl-kjob",
    "sudo mv ./kubectl-kjob /usr/local/bin/kubectl-kjob",
]

COMMAND_INSTALL_KUBECTL_KUEUE = [
    "curl -Lo ./kubectl-kueue https://github.com/kubernetes-sigs/kueue/releases/download/v0.13.3/kubectl-kueue-linux-amd64",
    "chmod +x ./kubectl-kueue",
    "sudo mv ./kubectl-kueue /usr/local/bin/kubectl-kueue",
]

COMMAND_INSTALL_XPK = [
    "git clone https://github.com/google/xpk.git /root/xpk",
    "cd /root/xpk",
    "python3 -m venv --system-site-packages .venv",
    "source .venv/bin/activate",
    "make install",
    "export PATH=$PATH:$HOME/xpk/bin",
]

COMMAND_BACK_MAXTEXT = [
    "cd /deps",  # back to maxtext folder
    "export USER=root",  # give it a USER to `pip install --upgrade pip`
]

COMMAND_RUN_RECIPE = ["python3 -m benchmarks.recipes.pw_mcjax_benchmark_recipe"]

COMMAND_DELETE_POD = [
    "set -xue",
    "export KUBECONFIG=/tmp/kubeconfig", # Change KUBECONFIG from /home/airflow to /tmp to avoid permission issue.
    "gcloud container clusters get-credentials {cluster_name} --region={region} --project={project}",
    "kubectl delete pod -l airflow-runtime={airflow_runtime} --namespace=default --force --grace-period=0",
]

COMMAND_ENV = (
    COMMAND_INSTALL_MAKE
    + COMMAND_INSTALL_KUBECTL
    + COMMAND_INSTALL_DOCKER
    + COMMAND_INSTALL_GCLOUD
    + COMMAND_SWITCH_SERVICE_ACCOUNT
    + COMMAND_INSTALL_KUBECTL_KJOB
    + COMMAND_INSTALL_KUBECTL_KUEUE
    + COMMAND_INSTALL_XPK
    + COMMAND_BACK_MAXTEXT
)

COMMAND_RECIPE = COMMAND_RUN_RECIPE

COMMAND_ENV = " && ".join(COMMAND_ENV)
COMMAND_RECIPE = " && ".join(COMMAND_RECIPE)
COMMAND_DELETE_POD_STR = " && ".join(COMMAND_DELETE_POD)


if __name__ == "__main__":
    print(f"{COMMAND_ENV = }\n")
    print(f"{COMMAND_RECIPE = }\n")
    TOTAL_COMMAND = " && ".join([COMMAND_ENV, COMMAND_RECIPE])
    print(f"{TOTAL_COMMAND = }\n")
