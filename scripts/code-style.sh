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

# Clean up Python codes using Pylint & Pyink
# Googlers: please run `sudo apt install pipx; pipx install pylint --force; pipx install pyink==23.10.0` in advance

set -e

FOLDERS_TO_FORMAT=("dags" "xlml")

# for folder in "${FOLDERS_TO_FORMAT[@]}"
# do
#   pyink "$folder" --pyink-indentation=2 --pyink-use-majority-quotes --line-length=80
# done
#
# for folder in "${FOLDERS_TO_FORMAT[@]}"
# do
#   pylint "./$folder" --fail-under=9.6
# done

HEAD_SHA="$(git rev-parse HEAD)"
BASE_BRANCH="tpu-obs/dev"

if ! git rev-parse --verify "$BASE_BRANCH" >/dev/null 2>&1; then
  git fetch origin "$BASE_BRANCH":"$BASE_BRANCH" || {
    echo "[code-style] base branch '$BASE_BRANCH' not found, skip diff-based check."
    exit 0
  }
fi

CHANGED_PY_FILES="$(
  git diff --name-only --diff-filter=ACM "${BASE_BRANCH}" "${HEAD_SHA}" \
    | grep '\.py$' \
    | while read -r f; do
        for folder in "${FOLDERS_TO_FORMAT[@]}"; do
          if [[ "$f" == "$folder/"* ]]; then
            echo "$f"
            break
          fi
        done
      done \
    | sort -u
)"

if [[ -z "${CHANGED_PY_FILES}" ]]; then
  echo "[pre-push hook] no changed files detected between ${HEAD_SHA} and ${BASE_BRANCH}"
  exit 1
fi

pyink ${CHANGED_PY_FILES} --pyink-indentation=2 --pyink-use-majority-quotes --line-length=80 --check --diff

pylint ${CHANGED_PY_FILES} --fail-under=9.6 --disable=E1123

echo "Successfully clean up all codes."
