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

import shlex


def setup_ops_agent(queued_resource_name) -> str:
  config = generate_ops_agent_config(queued_resource_name)
  return f"""config_file=/etc/google-cloud-ops-agent/config.yaml
  if [[ ! -f $config_file ]];
  then
    curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
    installed=0
    while [ $installed -eq 0 ]
    do
      pid=$(sudo lsof /var/lib/dpkg/lock-frontend | awk "END{{print $2}}")
      if [[ ! -z $pid ]]
      then
        sleep 10
      else
        sudo bash add-google-cloud-ops-agent-repo.sh --also-install
        installed=1
      fi
    done
  fi
  sudo chmod 777 $config_file
  sudo echo "{config}" >> $config_file
  sudo service google-cloud-ops-agent restart
"""


def generate_ops_agent_config(queued_resource_name) -> str:
  return f"""logging:
  receivers:
    {queued_resource_name}:
      type: files
      include_paths:
      - /tmp/{queued_resource_name}
      record_log_file_path: true
  service:
    pipelines:
      default_pipeline:
        receivers: [{queued_resource_name}]
"""


def generate_startup_script(main_command: str, queued_resource_name: str) -> str:
  escaped_command = shlex.quote(main_command)
  setup_ops_agent_cmds = setup_ops_agent(queued_resource_name)
  return f"""
{setup_ops_agent_cmds}
set -o pipefail
bash -c {escaped_command} 2>&1 | tee /tmp/logs &
pid=$!
echo $pid > /tmp/main_process_id.txt
wait $pid
exit_status=$?
echo $exit_status > /tmp/process_exit_status.txt
"""


def monitor_startup_script() -> str:
  return """
# File paths
pid_file="/tmp/main_process_id.txt"
status_file="/tmp/process_exit_status.txt"
log_file="/tmp/logs"

echo "LOGGER: Waiting for the workload to show up in $pid_file"

# Wait until the PID file exists
while [ ! -f "$pid_file" ]; do
  sleep 1
done

# Extract PID from pid_file
pid=$(cat "$pid_file")

echo "LOGGER: Streaming worker 0 logs."
# Tail the log file and terminate when the process with $pid exits
tail -f --pid=$pid --retry $log_file

echo "LOGGER: Process $pid has finished."

# Check if status_file contain any number
if ! grep -q '[0-9]' "$status_file"; then
  echo "LOGGER: The status file contains no exit status."
  exit 1
fi

# Read and output the exit status
exit_status=$(cat "$status_file")

# Check the exit_status
if [ "$exit_status" -eq 0 ]; then
  echo "LOGGER: The process exited successfully."
  exit 0
else
  echo "LOGGER: The process failed with exit status $exit_status."
  exit 1
fi
"""
