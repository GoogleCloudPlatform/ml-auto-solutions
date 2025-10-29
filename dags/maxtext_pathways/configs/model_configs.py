# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common model perf configs"""

import enum


class MaxTextV5eModelConfigs(enum.Enum):
  # Refers to model configs in https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/maxtext_v5e_model_configs.py
  DEFAULT_16B_V5E_256 = "default_16b_v5e_256"
  DEFAULT_32B_V5E_256 = "default_32b_v5e_256"
  DEFAULT_64B_V5E_256 = "default_64b_v5e_256"
  DEFAULT_128B_V5E_256 = "default_128b_v5e_256"
  GPT_3_175B_V5E_256 = "gpt_3_175b_v5e_256"
  LLAMA2_7B_V5E_256 = "llama2_7b_v5e_256"
  LLAMA2_13B_V5E_256 = "llama2_13b_v5e_256"
  LLAMA2_70B_V5E_256 = "llama2_70b_v5e_256"
  LLAMA3_1_8B_8192_V5E_256 = "llama3_1_8b_8192_v5e_256"


class MaxTextV5pModelConfigs(enum.Enum):
  # Refers to model configs in https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/maxtext_v5p_model_configs.py
  DEEPSEEK_V3_EP_256_V5P_512 = "deepseek_v3_ep_256_v5p_512"
  LLAMA4_SCOUT_DROPLESS_V5P_256 = "llama4_scout_dropless_v5p_256"
  LLAMA4_MAVERICK_DROPLESS_V5P_256 = "llama4_maverick_dropless_v5p_256"
  LLAMA2_70B_V5P_128 = "llama2_70b_v5p_128"
  LLAMA2_7B_V5P_128 = "llama2_7b_v5p_128"
  GPT_3_175B_V5P_128 = "gpt_3_175b_v5p_128"
  GPT_3_175B_V5P_128_SC = "gpt_3_175b_v5p_128_sc"


class MaxTextTrilliumModelConfigs(enum.Enum):
  # Refers to model configs in https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/maxtext_trillium_model_configs.py
  DEFAULT_BASIC_1 = "default_basic_1"
  DEFAULT_32 = "default_32"
  DEFAULT_64 = "default_64"
  DEFAULT_128 = "default_128"
  DEFAULT_256 = "default_256"
  DEFAULT_512 = "default_512"
  GPT_3_175B = "gpt_3_175b"
  GPT_3_175B_BF16 = "gpt_3_175b_bf16"
  LLAMA2_7B_4096 = "llama2_7b_4096"
  LLAMA2_70B_4096 = "llama2_70b_4096"
  LLAMA2_70B_4096_SYNTHETIC = "llama2_70b_4096_synthetic"
  LLAMA2_70B_4096_SC = "llama2_70b_4096_sc"
  LLAMA2_70B_4096_SC_REAL_DATA_TFDS = "llama2_70b_4096_sc_real_data_tfds"
  LLAMA2_70B_4096_SC_REAL_DATA_GRAIN = "llama2_70b_4096_sc_real_data_grain"
  LLAMA2_70B_4096_SC_REAL_DATA_GRAIN_CHECKPOINT = (
      "llama2_70b_4096_sc_real_data_grain_checkpoint"
  )
  LLAMA2_70B_4096_RD_LR = "llama2_70b_4096_rd_lr"
  LLAMA3_8B_8192 = "llama3_8b_8192"
  LLAMA3_70B_8192 = "llama3_70b_8192"
  LLAMA3_1_405B_8192_FSDP_DCN = "llama3_1_405b_8192_fsdp_dcn"
  LLAMA3_1_405B_8192_PURE_FSDP_ICI = "llama3_1_405b_8192_pure_fsdp_ici"
  LLAMA3_1_8B_8192 = "llama3_1_8b_8192"
  LLAMA3_1_8B_8192_BS5 = "llama3_1_8b_8192_bs5"
  LLAMA3_1_8B_8192_NO_COLLECTIVE_MATMUL = (
      "llama3_1_8b_8192_no_collective_matmul"
  )
  LLAMA3_1_70B_8192 = "llama3_1_70b_8192"
  LLAMA3_1_70B_8192_BS2 = "llama3_1_70b_8192_bs2"
  LLAMA3_1_70B_8192_BS2_BFLOAT16_NO_COLLECTIVE_MATMUL = (
      "llama3_1_70b_8192_bs2_bfloat16_no_collective_matmul"
  )
  LLAMA3_1_70B_8192_BS4 = "llama3_1_70b_8192_bs4"
  LLAMA3_1_70B_8192_SYNTHETIC = "llama3_1_70b_8192_synthetic"
  LLAMA3_1_70B_8192_RD_GRAIN = "llama3_1_70b_8192_rd_grain"
  LLAMA3_1_70B_8192_SYNTHETIC_CKPT = "llama3_1_70b_8192_synthetic_ckpt"
  LLAMA3_1_70B_8192_RD_CKPT_GRAIN = "llama3_1_70b_8192_rd_ckpt_grain"
  LLAMA3_1_70B_8192_PW_LR_RD = "llama3_1_70b_8192_pw_lr_rd"
  LLAMA3_1_70B_8192_ITER_REAL_DATA_AND_CHECKPOINTING_TFDS = (
      "llama3_1_70b_8192_iter_real_data_and_checkpointing_tfds"
  )
  LLAMA3_1_70B_8192_SYNTH = "llama3_1_70b_8192_synth"
  LLAMA3_1_70B_129024 = "llama3_1_70b_129024"
  MISTRAL_7B = "mistral_7b"
  MIXTRAL_8X7B_DROPLESS = "mixtral_8x7b_dropless"
  MIXTRAL_8X7B_DROPPED = "mixtral_8x7b_dropped"
  MIXTRAL_8X7B_DROPPED_INT8 = "mixtral_8x7b_dropped_int8"
  MIXTRAL_8X22B_DROPPED = "mixtral_8x22b_dropped"
  DEEPSEEK_V3_EP16 = "deepseek_v3_ep16"
  GEMMA2_9B_8192 = "gemma2_9b_8192"
  GEMMA2_27B_8192 = "gemma2_27b_8192"
  LLAMA3_1_70B_131072 = "llama3_1_70b_131072"
  CUSTOM_MOE_700B = "custom_moe_700b"
