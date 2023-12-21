FROM python:3.10-slim

# Install gcloud cli tools: gcloud and gcsfuse
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl fuse && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update && apt-get install -y google-cloud-cli && \
    curl https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v1.1.0/gcsfuse_1.1.0_amd64.deb -o gcsfuse_1.1.0_amd64.deb -L && \
    dpkg -i gcsfuse_1.1.0_amd64.deb && \
    rm gcsfuse_1.1.0_amd64.deb

# Copy HF model configs from GCS
RUN mkdir /configs && gsutil -m rsync -r gs://optimus-prime-configs /configs

# Baseline dependencies
RUN apt-get update && \
    apt-get install -y git libopenblas-dev libomp5 libgomp1 iproute2 sudo ethtool
