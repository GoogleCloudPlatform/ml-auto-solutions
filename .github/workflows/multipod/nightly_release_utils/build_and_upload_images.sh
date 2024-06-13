#!/bin/bash

# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script builds and uploads two images - one with only dependencies, the other also has a code snapshot.
# These images are tagged in GCR with both "latest" and date in format YYYY-MM-DD via $(date +%Y-%m-%d)

# Example command:
# bash build_and_upload_images.sh PROJECT=<project> MODE=nightly MERGE_IMAGE_NAME=<MERGE_IMAGE_NAME> OUTPUT_IMAGE_NAME=<OUTPUT_IMAGE_NAME>

set -e

# Set environment variables
for ARGUMENT in "$@"; do
    IFS='=' read -r KEY VALUE <<< "$ARGUMENT"
    export "$KEY"="$VALUE"
    echo "$KEY"="$VALUE"
done

if [[ ! -v OUTPUT_IMAGE_NAME ]] || [[ ! -v PROJECT ]] || [[ ! -v MODE ]] || [[ ! -v MERGE_IMAGE_NAME ]]; then
  echo "You must set OUTPUT_IMAGE_NAME, PROJECT, MODE, and MERGE_IMAGE_NAME"
  exit 1
fi

gcloud auth configure-docker --quiet
image_date=$(date +%Y-%m-%d)

if [[ "$MODE" == "nightly" ]]; then
  merge_image=gcr.io/$PROJECT/$MERGE_IMAGE_NAME:latest
else
  merge_image=gcr.io/$PROJECT/$MERGE_IMAGE_NAME:stable
fi

output_image=gcr.io/$PROJECT/$OUTPUT_IMAGE_NAME:$image_date

docker pull $merge_image
docker build --build-arg MERGE_IMAGE=$merge_image -f .github/workflows/multipod/nightly_release_utils/merge.Dockerfile -t ${OUTPUT_IMAGE_NAME}_runner .
docker tag ${OUTPUT_IMAGE_NAME}_runner $output_image
docker push $output_image
