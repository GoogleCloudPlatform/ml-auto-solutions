FROM python:3.10-slim

ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

# Install gcloud cli tools: gcloud and gcsfuse
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl fuse && \
	mkdir -p /usr/local/gcloud && \
    curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz | tar xvz -C /usr/local/gcloud && \
	/usr/local/gcloud/google-cloud-sdk/install.sh && \
	gcloud components install kubectl && \
    curl https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v1.4.0/gcsfuse_1.4.0_amd64.deb -o gcsfuse.deb -L && \
    dpkg -i gcsfuse.deb && \
    rm gcsfuse.deb

# Copy HF model configs from GCS
RUN mkdir /configs && gsutil -m rsync -r gs://optimus-prime-configs /configs

# Baseline dependencies
RUN apt-get update && \
    apt-get install -y git libopenblas-dev libomp5 libgomp1 iproute2 sudo ethtool

# PyTorch setup
RUN pip install --no-cache-dir tensorboardX google-cloud-storage gcsfs fsspec && \
    pip install --no-cache-dir https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch-nightly-cp310-cp310-linux_x86_64.whl && \
	pip install --no-cache-dir https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-nightly-cp310-cp310-linux_x86_64.whl

# Transformers setup
RUN git clone -b llama2-google-next-training https://github.com/pytorch-tpu/transformers.git /tmp/transformers && \
    pip install --no-cache-dir /tmp/transformers accelerate datasets evaluate scikit-learn
