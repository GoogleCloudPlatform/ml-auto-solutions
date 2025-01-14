// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local utils = import 'templates/utils.libsonnet';

{
  local stable_diffusion_2_train = self.stable_diffusion_2_train,
  stable_diffusion_2_train:: common.PyTorchTest + common.Functional + common.PyTorchTpuVmMixin {
    modelName: 'stable-diffusion-2-train',
    command: [
      'python',
      'diffusers/examples/text_to_image/train_text_to_image_xla.py',
      '--pretrained_model_name_or_path=stabilityai/stable-diffusion-2-base',
      '--dataset_name=lambdalabs/naruto-blip-captions',
      '--resolution=512',
      '--center_crop',
      '--random_flip',
      '--train_batch_size=8',
      '--max_train_steps=20',
      '--learning_rate=1e-06',
      '--mixed_precision=bf16',
      '--output_dir=/tmp/output',
      '--dataloader_num_workers=8',
      '--loader_prefetch_size=4',
      '--device_prefetch_size=4',
      '--loader_prefetch_factor=4',
    ],
    tpuSettings+: {
      tpuVmExports+: |||
        export PJRT_DEVICE=TPU
        export XLA_USE_SPMD=1
      |||,
      tpuVmExtraSetup: |||
        cat > ~/hf-constraints.txt << 'HF_CONSTRAINTS_EOF'
        %s
        HF_CONSTRAINTS_EOF

        git clone https://github.com/pytorch-tpu/diffusers.git

        cd diffusers
        sudo pip3 install -e . -c ~/hf-constraints.txt
        pip3 install -r examples/text_to_image/requirements.txt -c ~/hf-constraints.txt
        pip3 install 'torch_xla[pallas]' -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
      ||| % common.HuggingfacePipVersionConstraints,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  local v5p_8 = self.v5p_8,
  v5p_8:: {
    tpuSettings+: {
      softwareVersion: 'v2-alpha-tpuv5',
    },
    accelerator: tpus.v5p_8,
  },

  local trillium_4 = self.trillium_4,
  trillium_4:: {
    tpuSettings+: {
      softwareVersion: 'v2-alpha-tpuv6e',
    },
    accelerator: tpus.trillium_4,
  },

  configs: [
    stable_diffusion_2_train + v4_8 + timeouts.Hours(3),
    stable_diffusion_2_train + v5p_8 + timeouts.Hours(3),
    stable_diffusion_2_train + trillium_4 + timeouts.Hours(3),
  ],
}
