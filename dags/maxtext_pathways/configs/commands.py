# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

UPDATE_APT = [
    "sudo apt-get update",
]

INSTALL_MAKE = [
    "sudo apt-get install -y make",
]

INSTALL_KUBECTL = [
    "sudo apt-get install -y kubectl",
    "kubectl version --client",
]

INSTALL_DOCKER = [
    "sudo apt-get install ca-certificates curl",
    "sudo install -m 0755 -d /etc/apt/keyrings",
    "sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc",
    "sudo chmod a+r /etc/apt/keyrings/docker.asc",
    'echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null',
    "sudo apt-get update",
    "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin",
    "sudo /usr/bin/dockerd & sleep 0",  # Start the service in background.
    "while ! sudo docker info > /dev/null 2>&1; do echo 'Waiting for Docker to start...'; sleep 5; done",  # Wait for docker to be started.
]

INSTALL_KUBECTL_KJOB = [
    "curl -Lo ./kubectl-kjob https://github.com/kubernetes-sigs/kjob/releases/download/v0.1.0/kubectl-kjob-linux-amd64",
    "chmod +x ./kubectl-kjob",
    "sudo mv ./kubectl-kjob /usr/local/bin/kubectl-kjob",
]

INSTALL_KUBECTL_KUEUE = [
    "curl -Lo ./kubectl-kueue https://github.com/kubernetes-sigs/kueue/releases/download/v0.13.3/kubectl-kueue-linux-amd64",
    "chmod +x ./kubectl-kueue",
    "sudo mv ./kubectl-kueue /usr/local/bin/kubectl-kueue",
]

INSTALL_XPK = [
    "git clone --branch v0.12.0 https://github.com/AI-Hypercomputer/xpk /root/xpk",
    "cd /root/xpk",
    "python3 -m venv --system-site-packages .venv",
    "source .venv/bin/activate",
    "make install",
    "export PATH=$PATH:$HOME/xpk/bin",
]

SWITCH_SERVICE_ACCOUNT = [
    "mkdir -p /root/keys",
    'gcloud secrets versions access "latest" --secret="one-click-key" > /root/keys/sa_key.json',
    "gcloud config set account {service_account}",
    "gcloud auth activate-service-account {service_account} --key-file=/root/keys/sa_key.json",
]

BACK_MAXTEXT = [
    "cd /deps",  # back to maxtext folder
    "export USER=root",  # give it a USER to `pip install --upgrade pip`
]
