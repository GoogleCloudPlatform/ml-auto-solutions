# Dockerfile for maxtext_end_to_end.py,
# and is saved at gcr.io/cloud-ml-auto-solutions/maxtext_end_to_end with tag nightly or stable.
FROM python:3.10
ARG MODE
RUN echo "Setting up MaxText in $MODE mode..."
RUN pip install --upgrade pip
# Install gcloud
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
# Set up MaxText
RUN git clone https://github.com/google/maxtext.git /tmp/maxtext
RUN cd /tmp/maxtext && bash setup.sh MODE=$MODE
