# Copyright 2024 Google LLC
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


def genereate_startup_script(main_command: str) -> str:
  return f"""
bash -c '{main_command} 2>&1 | tee /tmp/logs &
pid=$!
echo $pid > /tmp/main_process_id.txt
wait $pid
exit_status=$?
echo $exit_status > /tmp/process_exit_status.txt'
"""


def check_if_startup_script_finish() -> str:
  return """
# File paths
pid_file="/tmp/main_process_id.txt"
status_file="/tmp/process_exit_status.txt"
log_file="/tmp/logs"

echo "Waiting for the workload to show up in $pid_file"

# Wait until the PID file exists
while [ ! -f "$pid_file" ]; do
  sleep 1
done

# Extract PID from pid_file
pid=$(cat "$pid_file")

# Tail the log file and terminate when the process with $pid exits
tail -f --pid=$pid --retry $log_file

echo "Process $pid has finished."

# Read and output the exit status
exit_status=$(cat "$status_file")
if [ "$exit_status" -eq 0 ]; then
  echo "The process exited successfully."
else
  echo "The process failed with exit status $exit_status."
  exit 1
fi

exit 0
"""
