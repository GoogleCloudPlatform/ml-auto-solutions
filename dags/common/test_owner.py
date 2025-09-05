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

"""
This file is used to look up the GitHub username of a test owner.

The GitHub username is then used to assign the GitHub Issue created by the alert plugin.

Please append your GitHub username if you need to add your test owner item.
"""

import enum


class Team(enum.Enum):
  SOLUTIONS_TEAM = "solutions_team"
  PYTORCH_XLA = "pytorch_xla"
  MULTIPOD = "multipod"
  MLCOMPASS = "mlcompass"
  INFERENCE = "inference"
  FRAMEWORK = "framework3p"
  LLM_DEVX = "llm_devx"
  SPARSITY_DIFFUSION_DEVX = "sparsity_diffusion_devx"
  PERFORMANCE = "performance"
  PRODUCTIVITY = "productivity"


# Default test owner
AIRFLOW = "airflow"
# XLML - TensorFlow
ERIC_L = "Eric L."
CHANDRA_D = "chandrasekhard2"
GAGIK_A = "gagika"

# PYTORCH
PEI_Z = "Pei Z."
MANFEI_B = "manfeiBai"

# MaxText
TONY_C = "tonyjohnchen"
JON_B = "Jon B."
RAYMOND_Z = "Raymond Z."
MATT_D = "gobbleturk"
PRIYANKA_G = "Priyanka G."
SURBHI_J = "SurbhiJainUSC"
ZHIYU_L = "Zhiyu L."
MOHIT_K = "khatwanimohit"
ANISHA_M = "A9isha"
YUWEI_Y = "Yuwei Y."
RISHABH_B = "notabee"
NUOJIN_C = "NuojCheng"
BRANDEN_V = "bvandermoon"

# Multi-tier Checkpointing
ABHINAV_S = "abhinavclemson"
XUEFENG_G = "xuefgu"
CAMILO_Q = "camiloCienet"
DEPP_L = "ooops678"

# MLCompass
ORTI_B = "ortibazar"

# Sparsity & Diffusion DevX
RAN_R = "RissyRan"
PARAM_B = "parambole"
KUNJAN_P = "coolkp"
MICHELLE_Y = "michelle-yooh"
SHUNING_J = "shuningjin"
ROHAN_B = "Rohan-Bierneni"

# Inference
ANDY_Y = "Andy Y."
XIANG_S = "sixiang-google"
MORGAN_D = "Morgan D."
YIJIA_J = "jyj0w0"
PATE_M = "patemotter"

# 3P Ecosystems
RICHARD_L = "richardsliu"
WENXIN_D = "wenxindongwork"

# FRAMEWORK
QINY_Y = "qinyiyan"

# JAX
AKANKSHA_G = "guptaaka"

# MAP_REPRODUCIBILITY
GUNJAN_J = "gunjanj007"

# Bite
Maggie_Z = "jiya-zhang"
Andrew_S = "asall"
