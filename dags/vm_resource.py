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


V5_NETWORKS_PREFIX = "projects/tpu-prod-env-automated"
V5_NETWORKS = f"{V5_NETWORKS_PREFIX}/global/networks/mas-test"
V5E_SUBNETWORKS = f"{V5_NETWORKS_PREFIX}/regions/us-east1/subnetworks/mas-test"
V5P_SUBNETWORKS = f"{V5_NETWORKS_PREFIX}/regions/us-east5/subnetworks/mas-test"

BM_NETWORKS_PREFIX_BENCHMARKING = "projects/cloud-ml-benchmarking"
BM_NETWORKS = f"{BM_NETWORKS_PREFIX_BENCHMARKING}/global/networks/mas-test"
V4_BM_SUBNETWORKS = f"{BM_NETWORKS}/regions/us-central2/subnetworks/mas-test"
V5E_BM_SUBNETWORKS = f"{BM_NETWORKS}/regions/us-west1/subnetworks/mas-test"
V5P_BM_SUBNETWORKS = f"{BM_NETWORKS}/regions/us-east5/subnetworks/mas-test"

INFERENCE_NETWORK_PREFIX = "projects/cloud-tpu-inference-test"
INFERENCE_NETWORKS = f"{INFERENCE_NETWORK_PREFIX}/global/networks/mas-test"
H100_INFERENCE_SUBNETWORKS = (
    "regions/us-central1/subnetworks/mas-test-us-central1"
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


class ImageProject(enum.Enum):
  """Common image projects for GPU."""

  DEEP_LEARNING_PLATFORM_RELEASE = "deeplearning-platform-release"


class ImageFamily(enum.Enum):
  """Common image families for GPU."""

  COMMON_CU121_DEBIAN_11 = "common-cu121-debian-11"


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
  # reserved v5e in tpu-prod-env-multipod
  US_WEST4_B = "us-west4-b"
  # reserved v5e in cloud-tpu-inference-test
  US_WEST1_C = "us-west1-c"
  # reserved a3+ cluster in supercomputer-testing
  AUSTRALIA_SOUTHEAST1_C = "australia-southeast1-c"
  # reserved trillium in tpu-prod-env-large-adhoc
  EUROPE_WEST4_A = "europe-west4-a"


class MachineVersion(enum.Enum):
  """Common machine types."""

  N1_STANDARD_8 = "n1-standard-8"
  N1_STANDARD_16 = "n1-standard-16"  # 60GB memory
  N1_STANDARD_32 = "n1-standard-32"
  A2_HIGHGPU_1G = "a2-highgpu-1g"
  A2_HIGHGPU_4G = "a2-highgpu-4g"
  A3_HIGHGPU_8G = "a3-highgpu-8g"
  G2_STAND_4 = "g2-standard-4"
  G2_STAND_16 = "g2-standard-16"  # 64GB memory
  G2_STAND_32 = "g2-standard-32"  # 128GB memroy


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
  H100 = "nvidia-h100-80gb"
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


class ClusterName(enum.Enum):
  """Common XPK cluster names."""

  V4_8_CLUSTER = "mas-v4-8"
  V4_32_CLUSTER = "mas-v4-32"
  V5E_4_CLUSTER = "mas-v5e-4"
  V5E_16_CLUSTER = "mas-v5e-16"
  V4_8_MULTISLICE_CLUSTER = "v4-8-maxtext"
  V4_16_MULTISLICE_CLUSTER = "v4-16-maxtext"
  V4_128_MULTISLICE_CLUSTER = "v4-128-bodaborg-us-central2-b"
  V5P_8_MULTISLICE_CLUSTER = "v5p-8-bodaborg-us-east5-a"
  V5E_16_MULTISLICE_CLUSTER = "v5e-16-bodaborg"
  V5E_256_MULTISLICE_CLUSTER = "v5e-256-bodaborg"
  V5E_256_US_WEST_4_MULTISLICE_CLUSTER = "v5e-256-bodaborg-us-west4"
  BODABORG_V6E_256_EUROPE_WEST4_A = "bodaborg-v6e-256-europe-west4-a"
  A3_CLUSTER = "maxtext-a3-20n"
  A3PLUS_CLUSTER = "a3plus-benchmark"
  CPU_M1_MEGAMEM_96 = "m1-megamem-96-shared"
  CPU_N2_STANDARD_64 = "shared-n2-standard-64"


class DockerImage(enum.Enum):
  """Common docker images."""

  XPK_JAX_TEST = "gcr.io/cloud-ml-auto-solutions/xpk_jax_test:latest"
  PYTORCH_NIGHTLY = (
      "us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/"
      f"xla:nightly_3.10_tpuvm_{datetime.datetime.today().strftime('%Y%m%d')}"
  )
  MAXTEXT_TPU_JAX_STABLE = (
      "us-docker.pkg.dev/tpu-prod-env-multipod/maxtext-jax-stable-stack/tpu:"
      f"jax0.4.30-rev1-{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXDIFFUSION_TPU_STABLE = (
      "us-docker.pkg.dev/tpu-prod-env-multipod/maxdiffusion-jax-stable-stack/tpu:"
      f"jax0.4.30-rev1-{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_TPU_JAX_NIGHTLY = (
      "gcr.io/tpu-prod-env-multipod/maxtext_jax_nightly:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_GPU_JAX_PINNED = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_pinned:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_GPU_JAX_STABLE = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_stable:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  MAXTEXT_GPU_JAX_NIGHTLY = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_nightly:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
  CLOUD_HYBRIDSIM_NIGHTLY = (
      "us-docker.pkg.dev/cloud-tpu-v2-images-dev/hybridsim/cloud_hybridsim_gcloud_python:"
      f"{datetime.datetime.today().strftime('%Y-%m-%d')}"
  )
