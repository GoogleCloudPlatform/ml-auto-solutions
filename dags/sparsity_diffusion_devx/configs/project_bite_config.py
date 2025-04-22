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

  test_name = f"bite_training_{'pinned_' if pinned_version else ''}{model_config}_{jax_version.replace('.', '-') if jax_version else 'main'}"
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


def get_bite_tpu_unittests_config(
    tpu_version: TpuVersion,
    tpu_cores: int,
    tpu_zone: str,
    runtime_version: str,
    time_out_in_min: int,
    task_owner: str,
    is_tpu_reserved: bool = False,
    pinned_version: Optional[str] = None,
):
  unittest_setupcmds = (
      # create configuration files needed
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
RUN pip install -U --pre libtpu-nightly -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --pre -U jaxlib -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
RUN pip install git+https://github.com/google/jax
RUN pip freeze
EOF
""",
      # create script to run the tests inside of the container
      # incluedes a basic sanity check python script which prints out TPU env info for reference
      """cat > run_tpu_tests.sh <<EOF
set -x
echo '#### Starting TPU JAX Tests'
pip freeze
JAX_PLATFORMS='tpu' python -c 'import jax; jax.print_environment_info() ; print(f"Global device count: {jax.device_count()}")'
cd /workspace/axlearn
pytest --no-header -v axlearn/common/flash_attention/
EOF
""",
      "chmod +x run_tpu_tests.sh",
      "sudo docker build -f Dockerfile_CI -t ml-auto-solutions/tpu_unittests .",
  )
  # Run the unittest as non-root user, ulimit param req to mmap TPUs inside docker (default limit is 8192)
  unittest_runcmds = (
      "echo '#### Start docker image - tpu_unittests'",
      "sudo docker run --network=host --privileged --ulimit memlock=-1:-1  ml-auto-solutions/tpu_unittests  /bin/bash -c '/workspace/run_tpu_tests.sh'",
  )
  job_gcp_config = gcp_config.GCPConfig(
      project_name=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      zone=tpu_zone,
      dataset_name=metric_config.DatasetOption.XLML_DATASET,
  )

  tpu_unittests_test_config = test_config.TpuVmTest(
      test_config.Tpu(
          version=tpu_version,
          cores=tpu_cores,
          runtime_version=runtime_version,
          reserved=is_tpu_reserved,
      ),
      test_name="bite_unittests",
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
