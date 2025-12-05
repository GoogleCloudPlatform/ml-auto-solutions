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
  JAX_MODELS_AND_PERFORMANCE = "jax_models_and_performance"
  PERFORMANCE = "performance"
  PRODUCTIVITY = "productivity"


# Default test owner
AIRFLOW = "airflow"
# XLML - TensorFlow
CHANDRA_D = "chandrasekhard2"
GAGIK_A = "gagika"

# PYTORCH
MANFEI_B = "manfeiBai"
BHAVYA_B = "bhavya01"

# MaxText
TONY_C = "tonyjohnchen"
MATT_D = "gobbleturk"
SURBHI_J = "SurbhiJainUSC"
MOHIT_K = "khatwanimohit"
ANISHA_M = "A9isha"
RISHABH_B = "notabee"
NUOJIN_C = "NuojCheng"
BRANDEN_V = "bvandermoon"
HENGTAO_G = "hengtaoguo"

# Multi-tier Checkpointing
ABHINAV_S = "abhinavclemson"
XUEFENG_G = "xuefgu"
CAMILO_Q = "camiloCienet"
DEPP_L = "ooops678"
JACKY_F = "RexBearIU"
SHARON_Y = "Shuang-cnt"

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
XIANG_S = "sixiang-google"
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
BRYAN_W = "bwuu"

# Bite
Maggie_Z = "jiya-zhang"
Andrew_S = "asall"

# Dashboard
SEVERUS_H = "severus-ho"

# maxtext_pathways
JULIE_K = "JulieKuo"

# TPU Observability
YUNA_T = "yuna-tzeng"
QUINN_M = "QuinnMMcGarry"
