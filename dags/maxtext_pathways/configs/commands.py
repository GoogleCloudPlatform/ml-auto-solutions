ENV_COMMAND = (
  "set -xue && "
  "pwd && "
  "ls && "
  "pip list && "
  # install make
  "sudo apt-get update && "
  "sudo apt-get install -y make && "
  # install docker
  "sudo apt-get install ca-certificates curl && "
  "sudo install -m 0755 -d /etc/apt/keyrings && "
  "sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && "
  "sudo chmod a+r /etc/apt/keyrings/docker.asc && "
  "echo \"deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable\" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && "
  "sudo apt-get update && "
  "sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin kubectl && " # install all necessary packages in a single command to avoid any potential locking conflicts
  "sudo /usr/bin/dockerd & sleep 20 && "
  # install gcloud
  "curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && "
  "tar -xf google-cloud-cli-linux-x86_64.tar.gz && "
  "./google-cloud-sdk/install.sh && "
  "./google-cloud-sdk/bin/gcloud init && "
  # install kubectl
  "grep -rhE ^deb /etc/apt/sources.list* | grep \"cloud-sdk\" && "
  "kubectl version --client && "
  # install kubectl-kjob
  "curl -Lo ./kubectl-kjob https://github.com/kubernetes-sigs/kjob/releases/download/v0.1.0/kubectl-kjob-linux-amd64 && "
  "chmod +x ./kubectl-kjob && "
  "sudo mv ./kubectl-kjob /usr/local/bin/kubectl-kjob && "
  # install kubectl-kueue
  "curl -Lo ./kubectl-kueue https://github.com/kubernetes-sigs/kueue/releases/download/v0.13.3/kubectl-kueue-linux-amd64 && "
  "chmod +x ./kubectl-kueue && "
  "sudo mv ./kubectl-kueue /usr/local/bin/kubectl-kueue && "
  # install xpk
  "git clone https://github.com/google/xpk.git ~/xpk && "
  "cd ~/xpk && "
  "python3 -m venv --system-site-packages .venv && "
  "source .venv/bin/activate && "
  "make install && "
  "export PATH=$PATH:$HOME/xpk/bin && "
  # back to maxtext and give it a USER to `pip install --upgrade pip`
  "cd /deps && "
  "export USER=root",
)

RECIPE_COMMAND = (
  "python3 -m benchmarks.recipes.pw_mcjax_benchmark_recipe",
)


if __name__ == '__main__':
  print(f"{ENV_COMMAND = }\n")
  print(f"{RECIPE_COMMAND = }\n")
  COMMAND = ENV_COMMAND[0] + " && " + RECIPE_COMMAND[0]
  print(COMMAND.replace("&& ", "&& \n"))