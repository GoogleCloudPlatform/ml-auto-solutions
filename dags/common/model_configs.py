# Copyright 2025 Google LLC
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

"""Common model perf configs"""

import enum


class MaxTextV5eModelConfigs(enum.Enum):
  # Refers to model configs in https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/maxtext_v5e_model_configs.py
  DEFAULT_16B = "default_16b_v5e_256"
  DEFAULT_32B = "default_32b_v5e_256"
  DEFAULT_64B = "default_64b_v5e_256"
  DEFAULT_128B = "default_128b_v5e_256"
  GPT3_175B = "gpt_3_175b_v5e_256"
  LLAMA2_7B = "llama2_7b_v5e_256"
  LLAMA2_13B = "llama2_13b_v5e_256"
  LLAMA2_70B = "llama2_70b_v5e_256"


class MaxTextTrilliumModelConfigs(enum.Enum):
  # Refers to model configs in https://github.com/AI-Hypercomputer/maxtext/blob/main/benchmarks/maxtext_trillium_model_configs.py
  GPT3_175B = "gpt_3_175b"
  LLAMA2_70B_4096 = "llama2_70b_4096_synthetic"
  LLAMA3_1_8B_8192 = "llama3_1_8b_8192"
  LLAMA3_1_70B_8192 = "llama3_1_70b_8192"
  LLAMA3_1_70B_129024 = "llama3_1_70b_129024"
  LLAMA3_1_405B_8192 = "llama3_1_405b_8192_fsdp_dcn"
  MIXTRAL_8X7B_DROPLESS = "mixtral_8x7b_dropless"
  MIXTRAL_8X7B_DROPPED = "mixtral_8x7b_dropped"
  MIXTRAL_8X7B_DROPPED_INT8 = "mixtral_8x7b_dropped_int8"
  DEEPSEEK_V3_EP16 = "deepseek_v3_ep16"
