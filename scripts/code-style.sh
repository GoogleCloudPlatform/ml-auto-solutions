#!/bin/bash

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may not use this file except in compliance with the License.
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

set -e # Exit immediately if a command exits with a non-zero status.
# set -x # Uncomment for debugging shell script execution

FOLDERS_TO_FORMAT=("dags" "xlml")
PYINK_OPTIONS="--pyink-indentation=2 --pyink-use-majority-quotes --line-length=80"
PYLINT_OPTIONS="--fail-under=9.6" # Pylint options

GIT_DIFF_CMD="git diff --name-only --diff-filter=ACM origin/master...HEAD -- '*.py'"

echo "Identifying changed Python files using: $GIT_DIFF_CMD"
ALL_CHANGED_PY_FILES=()
# Read files into an array, robustly handling spaces or special characters in filenames
while IFS= read -r file; do
  # Ensure file is not empty string which can happen if git diff output is empty
  if [[ -n "$file" ]]; then
    ALL_CHANGED_PY_FILES+=("$file")
  fi
done < <(eval $GIT_DIFF_CMD) # Using eval here as GIT_DIFF_CMD is constructed from trusted script strings

if [ ${#ALL_CHANGED_PY_FILES[@]} -eq 0 ]; then
  echo "No Python files found by the git diff command."
  # Depending on the context (e.g. pre-commit hook),
  # you might want to exit 0 here if no relevant files are found.
  exit 0
fi

echo "Found potentially relevant changed Python files: ${ALL_CHANGED_PY_FILES[*]}"

FILES_TO_PROCESS=()
for file in "${ALL_CHANGED_PY_FILES[@]}"; do
  for folder in "${FOLDERS_TO_FORMAT[@]}"; do
    # Check if the file path starts with the folder path, followed by a slash,
    # or if the file path exactly matches the folder name (less likely for files in folders).
    if [[ "$file" == "$folder/"* ]] || [[ "$file" == "$folder" ]]; then
      # Ensure the file actually exists in the working tree.
      # This is a safeguard, as `git diff --diff-filter=ACM` should list valid files.
      if [ -f "$file" ]; then
        FILES_TO_PROCESS+=("$file")
      else
        echo "Warning: Changed file '$file' not found in working tree. Skipping."
      fi
      break # File found in one of the target folders, no need to check other FOLDERS_TO_FORMAT
    fi
  done
done

if [ ${#FILES_TO_PROCESS[@]} -eq 0 ]; then
  echo "No changed Python files found within the specified folders: ${FOLDERS_TO_FORMAT[*]}"
  exit 0
fi


echo "--- Formatting ${#FILES_TO_PROCESS[@]} changed Python file(s) in specified folders ---"
echo "Files: ${FILES_TO_PROCESS[@]}"
# pyink can accept multiple files as arguments
pyink "${FILES_TO_PROCESS[@]}" $PYINK_OPTIONS

echo "--- Linting ${#FILES_TO_PROCESS[@]} changed Python file(s) in specified folders ---"
echo "Files: ${FILES_TO_PROCESS[@]}"
# pylint can also accept multiple files as arguments
pylint "${FILES_TO_PROCESS[@]}" $PYLINT_OPTIONS

echo "Successfully processed changed Python files in specified folders."
