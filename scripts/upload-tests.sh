# Copyright 2023 Google LLC
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

# Upload tests and utilities to a specified GCS dags folder

set -e

COMPOSER_ENVIRONMENT="test_env"
GCS_DAGS_FOLDER=$1
FOLDERS_TO_UPLOAD=("dags" "xlml")

# TODO(ranran): handle tests from Jsonnet
for folder in "${FOLDERS_TO_UPLOAD[@]}"
do
  gcloud storage rsync --checksums-only --delete-unmatched-destination-objects --recursive "$folder" "$GCS_DAGS_FOLDER"/"$folder"
done

ROOT_PATH=$GCS_DAGS_FOLDER
if [[ "$GCS_DAGS_FOLDER" == */dags ]]; then
    ROOT_PATH="${GCS_DAGS_FOLDER%/dags}"
fi

gcloud storage rsync --checksums-only --delete-unmatched-destination-objects --recursive --exclude '.*\.md$' plugins "$ROOT_PATH"/plugins

echo "Successfully uploaded tests."
