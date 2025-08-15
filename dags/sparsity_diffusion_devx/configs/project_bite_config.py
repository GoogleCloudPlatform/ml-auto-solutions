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

"""Utilities to construct configs for solutionsteam_jax_bite DAG."""


import datetime
from typing import Tuple, Optional
from dags.common import test_owner
from xlml.apis import gcp_config, metric_config, task, test_config
from dags import gcs_bucket
from dags.sparsity_diffusion_devx.configs import common
from dags.common.vm_resource import TpuVersion, Project
from airflow.models.taskmixin import DAGNode


GCS_SUBFOLDER_PREFIX = test_owner.Team.SPARSITY_DIFFUSION_DEVX.value


def set_up_axlearn(pinned_version, jax_version) -> Tuple[str]:
  reset_version = ""
  if pinned_version:
    reset_version = f"cd axlearn && git reset --hard {pinned_version} && cd .."

  setup_jax = None
  if jax_version:
    setup_jax = common.set_up_jax_version(jax_version)
  else:
    setup_jax = common.set_up_nightly_jax()

  return (
      common.UPGRADE_PIP,
      common.UPGRADE_SETUPTOOLS,
      common.UPGRADE_PACKAGING,
      "git clone https://github.com/apple/axlearn.git",
      reset_version,
      "python -m pip install ./axlearn[core]",
      *setup_jax,
  )


def get_bite_tpu_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    runtime_version: str,
    model_config: str,
    time_out_in_min: int,
    task_owner: str,
    is_tpu_reserved: bool = False,
    jax_version: Optional[str] = None,
    pinned_version: Optional[str] = None,
    project_name: Optional[Project] = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
    network: str = "default",
    subnetwork: str = "default",
):
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  set_up_cmds = set_up_axlearn(pinned_version, jax_version)
  run_model_cmds = (
      (
          "cd axlearn && python -m axlearn.common.launch_trainer_main"
          f" --module=text.gpt.c4_trainer --config={model_config}"
          f" --trainer_dir={metric_config.SshEnvVars.GCS_OUTPUT.value}"
          f" --data_dir={gcs_bucket.AXLEARN_DIR} --jax_backend=tpu"
      ),
  )

  test_name = f"bite_tpu_training_{'pinned_' if pinned_version else ''}{model_config}_{jax_version.replace('.', '-') if jax_version else 'main'}"
  job_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=set_up_cmds,
      run_model_cmds=run_model_cmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=task_owner,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/jax",
  )

  return task.run_queued_resource_test(
      task_test_config=job_test_config,
      task_gcp_config=job_gcp_config,
  )


def dockerfile_build_cmd(jax_version):
  # Generate pip commands to install certain version of JAX/libTPU e.g.
  # pip install --pre jaxlib==0.5.1  -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
  # pip install jax[tpu]==0.5.1  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  # pip install jax==0.5.1
  if jax_version:
    pip_tpu_jax_install = "\n".join(
        ["RUN " + x for x in common.set_up_jax_version(jax_version)]
    )
  else:
    pip_tpu_jax_install = "\n".join(
        ["RUN " + x for x in common.set_up_nightly_jax()]
    )

  return (
      """cat > Dockerfile_CI <<EOF
FROM python:3.10-slim
WORKDIR /workspace
COPY run_tpu_tests.sh /workspace/
RUN apt update -y
RUN apt install -y git
RUN git clone https://github.com/apple/axlearn.git
WORKDIR /workspace/axlearn
RUN pip install --upgrade pip
RUN pip install -e '.[core,dev,gcp]'
RUN pip install grain
RUN pip install google-cloud-aiplatform
"""
      + pip_tpu_jax_install
      + """
RUN pip freeze
EOF
"""
  )


def get_bite_tpu_unittests_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    runtime_version: str,
    time_out_in_min: int,
    task_owner: str,
    network: str = "default",
    subnetwork: str = "default",
    is_tpu_reserved: bool = False,
    # JAX version defaults to main if not specified
    jax_version: Optional[str] = None,
    project_name: Optional[Project] = Project.CLOUD_ML_AUTO_SOLUTIONS.value,
):
  """Function to create Cloud Composer tasks required to run TPU unittests.

  NB! If any of the pytest tests in Axlearn fail, then the whole task will be
  marked as failed.

  1. Creates list of setup commands which will:
    - Create a dockerfile configured to install Axlearn from github, including
      required prerequists and install the desired version of JAX
    - Create a test script to run inside that docker image (run_tpu_tests.sh)
      which runs pytest in the axlearn directory
    - Build a docker image - dockerfile_build_cmd() from dockerfile
  2. Creates list of test run commands which will:
    - On the TPU VM, run the docker image and execute the bash test script
    - Save the STDOUT to a log file, Save the docker logs to a file and copy these
      logs and the Axlearn XML test result output to the GCS output bucket
    - Get the exit code of the pytest command (in the container) and pass it back
      to the TPU VM code which then returns failure for the Airflow task.
  3. Uses test_config.TpuVmTest() class to generate a TPU VM/SSH keys, etc
    and run the setup commands (set_up_cmds) and then the test commands
    (run_model_cmds)

  Returns:
      A task group generated from the TpuVmTest() class.
  """
  unittest_setupcmds = (
      "mkdir test-results",
      # create configuration files needed
      dockerfile_build_cmd(jax_version),
      # create script to run the tests inside of the container
      # incluedes a basic sanity check python script which prints out TPU env info for reference
      # Save the tests exit code to a file which will be mapped to the local directory
      # Optional extra args to pass to axlearn for debugging etc purposes
      "export EXTRA_UNITTEST_ARGS='--maxfail=200'",
      """cat > run_tpu_tests.sh <<EOF
#!/bin/bash
set -x
echo '#### Starting TPU JAX Tests'
pip freeze
cd /workspace/axlearn
mkdir test-results
JAX_PLATFORMS='tpu' python -c 'import jax; jax.print_environment_info() ; print(f"Global device count: {jax.device_count()}")'
JAX_ENABLE_X64=True pytest --no-header -v -m "not high_cpu or fp64" --dist worksteal ${EXTRA_UNITTEST_ARGS} \
  --ignore axlearn/common/inference_test.py \
  --ignore axlearn/common/ssm_kernels/mamba_kernels_test.py \
  --ignore axlearn/common/ssm_test.py
TESTS_EXIT_CODE=\$?
echo '#### TPU JAX Tests finished.'
echo "Test exit code is \${TESTS_EXIT_CODE}"
echo "\${TESTS_EXIT_CODE}" > /workspace/axlearn/test-results/tests_exit_code.txt
cp -Rv /workspace/axlearn/test-results /tmp_docker/
EOF
""",
      "chmod +x run_tpu_tests.sh",
      "cat Dockerfile_CI",
      "cat run_tpu_tests.sh",
      "sudo docker build -f Dockerfile_CI -t ml-auto-solutions/tpu_unittests .",
      # Configure the TPU VM environment so it can run the post tests python utils
      "pip install google-api-core",
      "pip install google-cloud-bigquery",
  )
  unittest_runcmds = (
      "echo '#### Start docker image - tpu_unittests'",
      "cd ${HOME}",
      # To ensure filenames are consistent, define them all here and then pass the env vars to the
      # python scripts, gcloud commands etc
      "export TESTS_RUN_DATETIME=$(echo {{ ts_nodash_with_tz }} | tr '+-,@.' '-')",
      "export TESTS_RUN_ID='{{ run_id }}'",
      "export TEST_TASK_ID='{{ task.task_id }}'",
      'export TESTS_RUN_TYPE="${TESTS_RUN_ID%__*}"',
      "export TESTS_STDOUT_FILENAME=bite_unittests_std_out_err_${TESTS_RUN_TYPE}_${TESTS_RUN_DATETIME}.txt",
      "export TESTS_DOCKERLOG_FILENAME=bite_unittests_docker_log_${TESTS_RUN_TYPE}_${TESTS_RUN_DATETIME}.txt",
      "export TESTS_HTML_FILENAME=bite_unittests_${TESTS_RUN_TYPE}_${TESTS_RUN_DATETIME}.html",
      "echo Test Task ID:                ${TEST_TASK_ID}",
      "echo Test Run Timestamp:          ${TESTS_RUN_DATETIME}",
      "echo Test Run stdout filename:    ${TESTS_STDOUT_FILENAME}",
      "echo Test Run dockerlog filename: ${TESTS_DOCKERLOG_FILENAME}",
      "echo Test Run html filename:      ${TESTS_HTML_FILENAME}",
      # Run the unittest as non-root user, ulimit param req to mmap TPUs inside docker (default limit is 8192)
      "sudo docker run --network=host --privileged --ulimit memlock=-1:-1 -v ${PWD}:/tmp_docker ml-auto-solutions/tpu_unittests  \
        /bin/bash -c '/workspace/run_tpu_tests.sh' 2>&1 | tee test-results/${TESTS_STDOUT_FILENAME}",
      "sudo docker logs $( sudo docker ps --latest --quiet ) > ${HOME}/test-results/${TESTS_DOCKERLOG_FILENAME} 2>&1",
      # Post unit test processing
      "echo '#### Post Processing - tpu_unittests'",
      "echo GCS_OUTPUT env variable - ${GCS_OUTPUT}",
      "export TESTS_EXIT_CODE=$(cat test-results/tests_exit_code.txt)",
      "gcloud storage cp test-results/${TESTS_STDOUT_FILENAME}    ${GCS_OUTPUT}axlearn-test-results/",
      "gcloud storage cp test-results/${TESTS_DOCKERLOG_FILENAME} ${GCS_OUTPUT}axlearn-test-results/",
      # If the pytest crashes, testing.xml might not exist
      "if [[ -r test-results/testing.xml ]]; then gcloud storage cp test-results/testing.xml ${GCS_OUTPUT}axlearn-test-results ; else echo '*** Error - Tests failed to output XML results file'; fi",
      # Get the GCS bucket where the DAGs are located so that we can copy python utils to export the XML to BQ and HTML
      'echo gcloud composer environments describe ${COMPOSER_ENVIRONMENT} --project ${GCP_PROJECT} --location=${COMPOSER_LOCATION} --format="value(config.dagGcsPrefix)"',
      'export DAGS=`gcloud composer environments describe ${COMPOSER_ENVIRONMENT} --project ${GCP_PROJECT} --location=${COMPOSER_LOCATION} --format="value(config.dagGcsPrefix)"`',
      "echo DAGS is ${DAGS}",
      # Copy python utils from the GCS DAG folder to local test TPU VM
      "gcloud storage cp ${DAGS}/dags/sparsity_diffusion_devx/configs/xml_to_html.py xml_to_html.py",
      "gcloud storage cp ${DAGS}/dags/sparsity_diffusion_devx/configs/xml_to_bq.py xml_to_bq.py",
      "ls -al test-results/",
      # Export XML results to HTML and BQ - ignore errors from parsing output e.g. if the XML file
      # does not exist, as we don't want to stop the rest of the post processing
      "set +e ; python xml_to_html.py test-results/testing.xml ${TESTS_HTML_FILENAME}",
      "set +e ; python xml_to_bq.py ${TESTS_EXIT_CODE} test-results/testing.xml ${TESTS_RUN_TYPE} '{{ ts_nodash_with_tz }}' ${COMPOSER_ENVIRONMENT} \"${TEST_TASK_ID}\" \
        ${GCS_OUTPUT} ${TESTS_STDOUT_FILENAME} ${TESTS_DOCKERLOG_FILENAME} ${TESTS_HTML_FILENAME}",
      "set +e ; gcloud storage cp ${TESTS_HTML_FILENAME}                   ${GCS_OUTPUT}axlearn-test-results/",
      # Pytest will return up to an exit code of 6 - greater then this is probably segfault (139) or something else
      'if [[ `cat test-results/tests_exit_code.txt` -gt 6 ]]; then echo "*** Error - Tests failed to complete run *** Exit code ${TESTS_EXIT_CODE}" ; fi',
      # Output useful debug info
      "echo Test Results / Logs location: ${GCS_OUTPUT}axlearn-test-results/",
      "echo List of test results / logs:  $(gcloud storage ls ${GCS_OUTPUT}axlearn-test-results/ )",
      # Pass the results of the tests back to Cloud Composer
      "echo 'Tests exit code: '$(cat test-results/tests_exit_code.txt)",
      "if [[ `cat test-results/tests_exit_code.txt` -ne 0 ]]; then exit ${TESTS_EXIT_CODE} ; fi",
  )
  job_gcp_config = gcp_config.GCPConfig(
      project_name=project_name,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  test_name = f"bite_tpu_unittest_{jax_version.replace('.','-') if jax_version else 'main'}"

  tpu_unittests_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
          network=network,
          subnetwork=subnetwork,
      ),
      test_name=test_name,
      set_up_cmds=unittest_setupcmds,
      run_model_cmds=unittest_runcmds,
      timeout=datetime.timedelta(minutes=time_out_in_min),
      task_owner=task_owner,
      gcs_subfolder=f"{GCS_SUBFOLDER_PREFIX}/jax",
  )
  return task.run_queued_resource_test(
      task_test_config=tpu_unittests_test_config,
      task_gcp_config=job_gcp_config,
  )
