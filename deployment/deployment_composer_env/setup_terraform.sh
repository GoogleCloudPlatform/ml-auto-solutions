#!/bin/bash

set -e

export TERRAFORM_VERSION_TO_DOWNLOAD=1.8.0

mkdir -p $HOME/.terraform/bin
cd ~/.terraform

curl -Os https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION_TO_DOWNLOAD}/terraform_${TERRAFORM_VERSION_TO_DOWNLOAD}_linux_amd64.zip

unzip terraform_${TERRAFORM_VERSION_TO_DOWNLOAD}_linux_amd64.zip
mv terraform bin/

grep -o '\.terraform/bin' <<< $PATH \
    || grep -H '\.terraform/bin' ~/.bashrc \
    || echo 'export PATH=${PATH}:~/.terraform/bin' >> ~/.bashrc

source ~/.bashrc
terraform -v
