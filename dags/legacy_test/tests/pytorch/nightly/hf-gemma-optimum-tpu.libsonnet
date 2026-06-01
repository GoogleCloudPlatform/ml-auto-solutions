// Copyright 2020 Google LLC
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
  local gemma2b = self.gemma2b,
  gemma2b:: common.PyTorchTest {
    modelName: 'gemma2b',
  },

  local infer = self.infer,
  infer:: common.Functional + common.PyTorchTpuVmMixin {
    modelName+: '-infer',
    command: [
      'bash',
      '-c',
      |||
        sudo docker exec -it testhf bash -c '
        python -m pytest -sv text-generation-inference/tests
        '
      |||,
    ],

    tpuSettings+: {
      tpuVmExtraSetup: |||
        # create docker container
        export TPUVM_IMAGE_URL=us-central1-docker.pkg.dev/tpu-pytorch-releases/docker/xla
        export TPUVM_IMAGE_VERSION=8f1dcd5b03f993e4da5c20d17c77aff6a5f22d5455f8eb042d2e4b16ac460526

        sudo docker pull ${TPUVM_IMAGE_URL}@sha256:${TPUVM_IMAGE_VERSION}

        sudo docker run --privileged  --shm-size 16G --name testhf -it -d ${TPUVM_IMAGE_URL}@sha256:${TPUVM_IMAGE_VERSION}

        # sudo docker run -ti --rm --privileged --network=host --name testhf ${TPUVM_IMAGE_URL}@sha256:${TPUVM_IMAGE_VERSION} bash -c "
        sudo docker exec -it testhf bash -c '
          #TODO: add HF token available as public token
          export HF_TOKEN=xxx
          apt install sudo
          sudo pip install -U "huggingface_hub[cli]"
          sudo huggingface-cli login --token $HF_TOKEN
          git clone https://github.com/huggingface/optimum-tpu.git
          cd optimum-tpu/

          ### sudo make tgi_test
          # test_installs
          python -m pip install .[tests] -f https://storage.googleapis.com/libtpu-releases/index.html
        
          # tgi_server
	        python -m pip install -r text-generation-inference/server/build-requirements.txt
	        make -C text-generation-inference/server clean
	        VERSION=${VERSION} TGI_VERSION=${TGI_VERSION} make -C text-generation-inference/server gen-server

          # tgi_test: test_installs tgi_server
	        find text-generation-inference -name "text_generation_server-$(VERSION)-py3-none-any.whl" \
	                                 -exec python -m pip install --force-reinstall {} \;
	        # python -m pytest -sv text-generation-inference/tests
        ‘
      |||,
    },
  },

  local v4_8 = self.v4_8,
  v4_8:: {
    accelerator: tpus.v4_8,
  },

  configs: [
    gemma + v4_8 + infer + timeouts.Hours(3),
  ],
}
