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

"""The file for common projects, zone, and runtime versions."""

import datetime
import enum
from xlml.apis.xpk_cluster_config import XpkClusterConfig


V5_NETWORKS_PREFIX = "projects/tpu-prod-env-automated"
V5_NETWORKS = f"{V5_NETWORKS_PREFIX}/global/networks/mas-test"
V5E_SUBNETWORKS = f"{V5_NETWORKS_PREFIX}/regions/us-east1/subnetworks/mas-test"
V5P_SUBNETWORKS = f"{V5_NETWORKS_PREFIX}/regions/us-east5/subnetworks/mas-test"
V6E_SUBNETWORKS = (
    f"{V5_NETWORKS_PREFIX}/regions/us-central2/subnetworks/mas-test"
)
# TODO: Figure V6E_GCE_NETWORK and V6E_GCE_SUBNETWORK
V6E_GCE_NETWORK = "default"
V6E_GCE_SUBNETWORK = "default"

BM_NETWORKS_PREFIX_BENCHMARKING = "projects/cloud-ml-benchmarking"
BM_NETWORKS = f"{BM_NETWORKS_PREFIX_BENCHMARKING}/global/networks/mas-test"
A100_BM_SUBNETWORKS = "regions/us-west4/subnetworks/mas-test"
V4_BM_SUBNETWORKS = f"{BM_NETWORKS}/regions/us-central2/subnetworks/mas-test"
V5E_BM_SUBNETWORKS = f"{BM_NETWORKS}/regions/us-west1/subnetworks/mas-test"
V5P_BM_SUBNETWORKS = f"{BM_NETWORKS}/regions/us-east5/subnetworks/mas-test"

INFERENCE_NETWORK_PREFIX = "projects/cloud-tpu-inference-test"
INFERENCE_NETWORKS = f"{INFERENCE_NETWORK_PREFIX}/global/networks/mas-test"
US_EAST5_INFERENCE_SUBNETWORKS = (
    "regions/us-central1/subnetworks/mas-test-us-east5"
)
H100_INFERENCE_SUBNETWORKS = (
    "regions/us-central1/subnetworks/mas-test-us-central1"
)
A100_INFERENCE_SUBNETWORKS = (
    "regions/us-central1/subnetworks/mas-test-us-central1"
)
L4_INFERENCE_SUBNETWORKS = (
    "regions/us-central1/subnetworks/mas-test-us-central1"
)
H200_INFERENCE_SUBNETWORKS = (
    "regions/europe-west1/subnetworks/mas-test-europe-west1	"
)


class Project(enum.Enum):
  """Common GCP projects."""

  CLOUD_ML_AUTO_SOLUTIONS = "cloud-ml-auto-solutions"
  CLOUD_ML_BENCHMARKING = "cloud-ml-benchmarking"
  TPU_PROD_ENV_MULTIPOD = "tpu-prod-env-multipod"
  TPU_PROD_ENV_AUTOMATED = "tpu-prod-env-automated"
  CLOUD_TPU_MULTIPOD_DEV = "cloud-tpu-multipod-dev"
  SUPERCOMPUTER_TESTING = "supercomputer-testing"
  CLOUD_TPU_INFERENCE_TEST = "cloud-tpu-inference-test"
  TPU_PROD_ENV_LARGE_ADHOC = "tpu-prod-env-large-adhoc"
  TPU_PROD_ENV_ONE_VM = "tpu-prod-env-one-vm"
  TPU_PROD_ENV_LARGE_CONT = "tpu-prod-env-large-cont"


class ImageProject(enum.Enum):
  """Common image projects for GPU."""

  DEEP_LEARNING_PLATFORM_RELEASE = "deeplearning-platform-release"
  ML_IMAGES = "ml-images"


class ImageFamily(enum.Enum):
  """Common image families for GPU."""

  COMMON_CU124_DEBIAN_11 = "common-cu124-debian-11"


class Region(enum.Enum):
  """Common GCP regions."""

  # used for GKE
  US_CENTRAL1 = "us-central1"


class Zone(enum.Enum):
  """Common GCP zones."""

  # reserved/on-demand v2-32 in cloud-ml-auto-solutions
  US_CENTRAL1_A = "us-central1-a"
  # on-demand v3-8 in cloud-ml-auto-solutions
  US_CENTRAL1_B = "us-central1-b"
  # reserved v4-8 & v4-32 in cloud-ml-auto-solutions
  US_CENTRAL2_B = "us-central2-b"
  # reserved/on-demand v2-8 in cloud-ml-auto-solutions
  # & reserved h100 in supercomputer-testing
  US_CENTRAL1_C = "us-central1-c"
  # committed resource for A100
  US_CENTRAL1_F = "us-central1-f"
  # reserved v5e in tpu-prod-env-automated
  US_EAST1_C = "us-east1-c"
  # reserved v3-8 & reserved/on-demand v3-32 in cloud-ml-auto-solutions
  US_EAST1_D = "us-east1-d"
  # reserved h100-mega in supercomputer-testing
  US_EAST4_A = "us-east4-a"
  # reserved v5p in tpu-prod-env-automated
  US_EAST5_A = "us-east5-a"
  # reserved v6e in tpu-prod-env-one-vm
  US_EAST5_B = "us-east5-b"
  # reserved v6e in tpu-prod-env-automated
  US_EAST5_C = "us-east5-c"
  # reserved v5e in tpu-prod-env-multipod
  US_WEST4_B = "us-west4-b"
  # reserved v5e in cloud-tpu-inference-test
  US_WEST1_C = "us-west1-c"
  # reserved a3+ cluster in supercomputer-testing
  AUSTRALIA_SOUTHEAST1_C = "australia-southeast1-c"
  # reserved H200 capacity in cloud-tpu-inference-test
  EUROPE_WEST1_B = "europe-west1-b"
  # reserved TRILLIUM capacity
  EUROPE_WEST4_A = "europe-west4-a"
  SOUTHAMERICA_WEST1_A = "southamerica-west1-a"
  # reserved v5e capacity in tpu-prod-env-multipod
  EUROPE_WEST4_B = "europe-west4-b"
  # reserved l4 in cloud-tpu-inference-test
  ASIA_EAST1_A = "asia-east1-a"
  ASIA_EAST1_C = "asia-east1-c"


class MachineVersion(enum.Enum):
  """Common machine types."""

  N1_STANDARD_8 = "n1-standard-8"
  N1_STANDARD_16 = "n1-standard-16"  # 60GB memory
  N1_STANDARD_32 = "n1-standard-32"
  A2_HIGHGPU_1G = "a2-highgpu-1g"
  A2_HIGHGPU_4G = "a2-highgpu-4g"
  A2_ULTRAGPU_1G = "a2-ultragpu-1g"
  A2_ULTRAGPU_2G = "a2-ultragpu-2g"
  A2_ULTRAGPU_4G = "a2-ultragpu-4g"
  A2_ULTRAGPU_8G = "a2-ultragpu-8g"
  A3_HIGHGPU_8G = "a3-highgpu-8g"
  A3_MEGAGPU_8G = "a3-megagpu-8g"
  A3_ULTRAGPU_8G = "a3-ultragpu-8g"
  G2_STAND_4 = "g2-standard-4"
  G2_STAND_16 = "g2-standard-16"  # 64GB memory
  G2_STAND_32 = "g2-standard-32"  # 128GB memroy
  G2_STAND_48 = "g2-standard-48"  # 4 GPUs, 192GB memory
  G2_STAND_96 = "g2-standard-96"  # 8 GPUs, 384GB memory


class AcceleratorType(enum.Enum):
  CPU = "CPU"
  GPU = "GPU"
  TPU = "TPU"


class TpuVersion(enum.Enum):
  """Common TPU versions."""

  V2 = "2"
  V3 = "3"
  V4 = "4"
  V5E = "5litepod"
  V5P = "5p"
  TRILLIUM = "6e"


class GpuVersion(enum.Enum):
  """Common GPU versions."""

  L4 = "nvidia-l4"
  A100 = "nvidia-tesla-a100"
  A100_80G = "nvidia-a100-80gb"
  H100 = "nvidia-h100-80gb"
  H200 = "nvidia-h200-80gb"
  XPK_H100 = "h100-80gb-8"
  XPK_H100_MEGA = "h100-mega-80gb-8"
  V100 = "nvidia-tesla-v100"


class CpuVersion(enum.Enum):
  """Common CPU versions."""

  M1_MEGAMEM = "m1-megamem-96"
  N2_STANDARD = "n2-standard-64"


class RuntimeVersion(enum.Enum):
  """Common runtime versions."""

  TPU_VM_TF_NIGHTLY = "tpu-vm-tf-nightly"
  TPU_VM_TF_NIGHTLY_POD = "tpu-vm-tf-nightly-pod"
  TPU_VM_TF_STABLE_SE = "tpu-vm-tf-2.16.0-se"
  TPU_VM_TF_STABLE_POD_SE = "tpu-vm-tf-2.16.0-pod-se"
  TPU_VM_TF_STABLE_PJRT = "tpu-vm-tf-2.16.0-pjrt"
  TPU_VM_TF_STABLE_POD_PJRT = "tpu-vm-tf-2.16.0-pod-pjrt"
  TPU_VM_TF_V5P_ALPHA = "tpu-vm-tf-v5p-alpha-sc"
  TPU_UBUNTU2204_BASE = "tpu-ubuntu2204-base"
  TPU_VM_V4_BASE = "tpu-vm-v4-base"
  V2_ALPHA_TPUV5_LITE = "v2-alpha-tpuv5-lite"
  V2_ALPHA_TPUV5 = "v2-alpha-tpuv5"
  V2_ALPHA_TPUV6 = "v2-alpha-tpuv6e"


class XpkClusters:
  """Common XPK cluster configs."""

  TPU_V4_8_MAS_CLUSTER = XpkClusterConfig(
      name="mas-v4-8",
      device_version=TpuVersion.V4,
      core_count=8,
      project=Project.CLOUD_ML_AUTO_SOLUTIONS.value,
      zone=Zone.US_CENTRAL2_B.value,
  )
  TPU_V4_8_MAXTEXT_CLUSTER = XpkClusterConfig(
      name="v4-8-maxtext",
      device_version=TpuVersion.V4,
      core_count=8,
      project=Project.TPU_PROD_ENV_MULTIPOD.value,
      zone=Zone.US_CENTRAL2_B.value,
  )
  TPU_V4_16_CLUSTER = XpkClusterConfig(
      name="v4-16-maxtext",
      device_version=TpuVersion.V4,
      core_count=16,
      project=Project.TPU_PROD_ENV_MULTIPOD.value,
      zone=Zone.US_CENTRAL2_B.value,
  )
  TPU_V4_128_CLUSTER = XpkClusterConfig(
      name="v4-128-bodaborg-us-central2-b",
      device_version=TpuVersion.V4,
      core_count=128,
      project=Project.CLOUD_TPU_MULTIPOD_DEV.value,
      zone=Zone.US_CENTRAL2_B.value,
  )
  TPU_V5P_8_CLUSTER = XpkClusterConfig(
      name="v5p-8-bodaborg-europe-west4-b",
      device_version=TpuVersion.V5P,
      core_count=8,
      project=Project.CLOUD_TPU_MULTIPOD_DEV.value,
      zone=Zone.EUROPE_WEST4_B.value,
  )
  TPU_V5E_256_CLUSTER = XpkClusterConfig(
      name="v5e-256-bodaborg-europe-west4",
      device_version=TpuVersion.V5E,
      core_count=256,
      project=Project.TPU_PROD_ENV_MULTIPOD.value,
      zone=Zone.EUROPE_WEST4_B.value,
  )
  TPU_V6E_256_CLUSTER = XpkClusterConfig(
      name="bodaborg-v6e-256",
      device_version=TpuVersion.TRILLIUM,
      core_count=256,
      project=Project.TPU_PROD_ENV_LARGE_ADHOC.value,
      zone=Zone.US_CENTRAL2_B.value,
  )
  TPU_V6E_256_MLPERF_CLUSTER = XpkClusterConfig(
      name="bodaborg-v6e-256-lcscld-c",
      device_version=TpuVersion.TRILLIUM,
      core_count=256,
      project=Project.TPU_PROD_ENV_ONE_VM.value,
      zone=Zone.SOUTHAMERICA_WEST1_A.value,
  )
  TPU_V6E_16_IN_MEM_CLUSTER = XpkClusterConfig(
      name="in-mem-airflow-v6e-16",
      device_version=TpuVersion.TRILLIUM,
      core_count=16,
      project=Project.TPU_PROD_ENV_ONE_VM.value,
      zone=Zone.US_EAST5_C.value,
  )
  GPU_A3_CLUSTER = XpkClusterConfig(
      name="ninacai-maxtext-a3",
      device_version=GpuVersion.XPK_H100,
      core_count=8,
      project=Project.SUPERCOMPUTER_TESTING.value,
      zone=Zone.US_EAST5_A.value,
  )
  GPU_A3PLUS_CLUSTER = XpkClusterConfig(
      name="a3plus-benchmark",
      device_version=GpuVersion.XPK_H100_MEGA,
      core_count=8,
      project=Project.SUPERCOMPUTER_TESTING.value,
      zone=Zone.AUSTRALIA_SOUTHEAST1_C.value,
  )
  CPU_M1_MEGAMEM_96_CLUSTER = XpkClusterConfig(
      name="m1-megamem-96-shared",
      device_version=CpuVersion.M1_MEGAMEM,
      core_count=96,
      project=Project.TPU_PROD_ENV_MULTIPOD.value,
      zone=Zone.US_CENTRAL1_B.value,
  )
  CPU_N2_STANDARD_64_CLUSTER = XpkClusterConfig(
      name="shared-n2-standard-64",
      device_version=CpuVersion.N2_STANDARD,
      core_count=64,
      project=Project.TPU_PROD_ENV_MULTIPOD.value,
      zone=Zone.US_CENTRAL1_B.value,
  )


class DockerImage(enum.Enum):
  """Common docker images."""

  XPK_JAX_TEST = "gcr.io/cloud-ml-auto-solutions/xpk_jax_test:latest"
  PYTORCH_NIGHTLY = (
      "us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/"
      f"xla:nightly_3.10_tpuvm_{datetime.datetime.today().strftime('%Y%m%d')}"
  )
  AXLEARN_TPU_JAX_STABLE_STACK = (
      "us-docker.pkg.dev/tpu-prod-env-multipod/bite/tpu/axlearn:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  AXLEARN_GPU_JAX_NIGHTLY = (
      "us-docker.pkg.dev/tpu-prod-env-multipod/bite/gpu/jax_nightly:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_TPU_JAX_STABLE_STACK = (
      "gcr.io/tpu-prod-env-multipod/maxtext_jax_stable_stack:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_TPU_STABLE_STACK_NIGHTLY_JAX = (
      "gcr.io/tpu-prod-env-multipod/maxtext_stable_stack_nightly_jax:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_TPU_JAX_STABLE_STACK_CANDIDATE = (
      "gcr.io/tpu-prod-env-multipod/maxtext_stable_stack_candidate:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXDIFFUSION_TPU_JAX_STABLE_STACK = (
      "gcr.io/tpu-prod-env-multipod/maxdiffusion_jax_stable_stack:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXDIFFUSION_TPU_STABLE_STACK_NIGHTLY_JAX = (
      "gcr.io/tpu-prod-env-multipod/maxdiffusion_jax_nightly:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXDIFFUSION_TPU_JAX_STABLE_STACK_CANDIDATE = (
      "gcr.io/tpu-prod-env-multipod/maxdiffusion_stable_stack_candidate:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_TPU_JAX_NIGHTLY = (
      "gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_GPU_JAX_PINNED = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_pinned:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_GPU_JAX_STABLE_STACK = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_stable_stack:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_GPU_JAX_STABLE = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_stable:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_GPU_STABLE_STACK_NIGHTLY_JAX = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_stable_stack_nightly_jax:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  CLOUD_HYBRIDSIM_NIGHTLY = (
      "us-docker.pkg.dev/cloud-tpu-v2-images-dev/hybridsim/cloud_hybridsim_gcloud_python:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
