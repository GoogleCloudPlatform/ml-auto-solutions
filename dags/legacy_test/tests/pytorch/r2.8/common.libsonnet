// Copyright 2024 Google LLC
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

local common = import '../common.libsonnet';
local experimental = import '../experimental.libsonnet';
local mixins = import 'templates/mixins.libsonnet';
local utils = import 'templates/utils.libsonnet';
local volumes = import 'templates/volumes.libsonnet';

local rcVersion = 'rc1';

// make sure the vision commit aligns with upstream. E.g., for 2.8 release:
// https://github.com/pytorch/pytorch/blob/release/2.8/.github/ci_commit_pins/vision.txt.
local vision_commit = 'd23a6e1664d20707c11781299611436e1f0c104f';


{
  local r2_8 = {
    frameworkPrefix: 'pt-2-8',
    tpuSettings+: {
      softwareVersion: 'tpu-ubuntu2204-base',
    },
    imageTag: 'r2.8.0-%(rc)s_3.12' % {rc: rcVersion},
  },
  PyTorchTest:: common.PyTorchTest + r2_8 {
    local config = self,

    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': {
            image: config.podTemplate.spec.containerMap.train.image,
            env+: [
              {
                name: 'TPU_NAME',
                valueFrom: {
                  fieldRef: {
                    fieldPath: "metadata.annotations['name.cloud-tpus.google.com/train']",
                  },
                },
              },
            ],
            command: [
              'python3',
              '-c',
              |||
                import importlib_metadata
                import os
                import re

                import cloud_tpu_client

                requirements = importlib_metadata.requires('torch_xla')
                libtpu_pattern = r'libtpu-nightly ?@ https:\/\/storage.googleapis.com\/cloud-tpu-tpuvm-artifacts\/wheels\/libtpu-nightly\/libtpu_nightly-\d.\d.dev(\d{8})-\w+-\w+-\w+.whl'
                libtpu_matches = [
                  re.findall(libtpu_pattern, req)[0]
                  for req in requirements
                  if re.match(libtpu_pattern, req)
                ]
                assert len(libtpu_matches) == 1, f'{len(libtpu_matches)} matches in {requirements} (pattern: `{libtpu_pattern}`)'
                libtpu_date = libtpu_matches[0]
                print('libtpu date:', libtpu_date)

                ctc = cloud_tpu_client.Client(tpu=os.path.basename('$(TPU_NAME)'), zone=os.path.dirname('$(TPU_NAME)'))
                ctc.wait_for_healthy()
                ctc.configure_tpu_version(f'pytorch-2.8-dev{libtpu_date}', restart_type='always')
                ctc.wait_for_healthy()
              |||,
            ],
          },
        },
      },
    },
  },
  Functional:: mixins.Functional {
    schedule: '0 6 * * *',
    tpuSettings+: {
      preemptible: false,
    },
  },
  Convergence:: mixins.Convergence,
  PyTorchTpuVmMixin:: experimental.PyTorchTpuVmMixin + experimental.PjRt {
    local config = self,

    tpuSettings+: {
      softwareVersion: 'tpu-ubuntu2204-base',
      tpuVmPytorchSetup: |||
        pip3 install -U 'setuptools>=70.0.0,<71.0.0'
        # `unattended-upgr` blocks us from installing apt dependencies
        sudo systemctl stop unattended-upgrades || true
        sudo systemctl disable unattended-upgrades || true
        sudo killall --signal SIGKILL unattended-upgrades || true
        sudo dpkg --configure -a || true
        sudo rm /var/lib/dpkg/lock-frontend || true
        sudo apt purge unattended-upgrades -y || true
        echo "unattended-upgrades stopped."

        sudo apt-get -y update
        sudo apt install -y libopenblas-base
        # for huggingface tests
        sudo apt install -y libsndfile-dev
        # Install torchvision by pinned commit in PyTorch 2.8 release branch.
        pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/test/cpu
        # torchvision commit reference: https://github.com/pytorch/pytorch/blob/release/2.8/.github/ci_commit_pins/vision.txt
        pip install --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@%(vision_commit)s"
        pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/tpuvm/torch_xla-2.8.0%(rc)s-cp310-cp310-linux_x86_64.whl
        pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html
        pip install pillow
        git clone --depth=1 https://github.com/pytorch/pytorch.git
        cd pytorch
        git clone -b v2.8.0-%(rc)s https://github.com/pytorch/xla.git
      ||| % {rc: rcVersion, vision_commit: vision_commit},
    },
    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': null,
        },
      },
    },
  },

  datasetsVolume: volumes.PersistentVolumeSpec {
    name: 'pytorch-datasets-claim',
    mountPath: '/datasets',
  },
  GpuMixin:: {
    local config = self,
    imageTag+: '_cuda_12.6',

    // TODO(wcromar): Merge TPU VM setup script with GPU entrypoint
    tpuSettings+: {
      tpuVmExports+: |||
        export PJRT_DEVICE=CUDA
      |||,
    },

    entrypoint: [
      'bash',
      '-cxue',
      |||
        export PATH=/usr/local/nvidia/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

        nvidia-smi
        pip uninstall -y torch torchvision
        pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/test/cpu
        pip install --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@%(vision_commit)s"
        pip install https://storage.googleapis.com/pytorch-xla-releases/wheels/cuda/12.6/torch_xla-2.8.0%(rc)s-cp310-cp310-linux_x86_64.whl

        mkdir -p pytorch/xla
        git clone -b v2.8.0-%(rc)s https://github.com/pytorch/xla.git pytorch/xla

        %(cmd)s

        # Run whatever is in `command` here
        "${@:0}"
      ||| % {cmd: config.tpuSettings.tpuVmExports, rc: rcVersion, vision_commit: vision_commit},
    ],
    command: [
      'torchrun',
      '--nnodes=%d' % config.accelerator.num_hosts,
      '--node_rank=$(JOB_COMPLETION_INDEX)',
      '--nproc_per_node=%d' % config.accelerator.processes,
      '--rdzv_endpoint=$(JOB_NAME)-0.headless-svc:12355',
    ] + super.command[1:],

    podTemplate+:: {
      spec+: {
        initContainerMap+:: {
          'tpu-version': null,
        },
        containerMap+:: {
          train+: {
            envMap+: {
              GPU_NUM_DEVICES: '%d' % config.accelerator.count,
            },
          },
        },
      },
    },
  },


  Accelerate:: {
    local config = self,
    tpuSettings+: {
      tpuVmExports+: |||
        export PATH=~/.local/bin:$PATH
      |||,
      tpuVmExtraSetup: |||
        if [ -d "$HOME/.local/bin" ] ; then
          export PATH="$HOME/.local/bin:$PATH"
        fi

        cat > ~/hf-constraints.txt << 'HF_CONSTRAINTS_EOF'
        %s
        HF_CONSTRAINTS_EOF

        pip install pytest -c ~/hf-constraints.txt

        pip install git+https://github.com/zpcore/accelerate.git@v1.2.1-patch

        mkdir -p ~/.cache/huggingface/accelerate/
        cat > ~/.cache/huggingface/accelerate/default_config.yaml << 'HF_CONFIG_EOF'
        compute_environment: LOCAL_MACHINE
        distributed_type: XLA
        downcast_bf16: 'no'
        machine_rank: 0
        main_training_function: main
        mixed_precision: 'no'
        num_machines: 1
        num_processes: null
        rdzv_backend: static
        same_network: true
        tpu_env: []
        tpu_use_cluster: false
        tpu_use_sudo: false
        use_cpu: false
        HF_CONFIG_EOF

        accelerate env
      ||| % common.HuggingfacePipVersionConstraints,
    },
  },

  HuggingfacePipVersionConstraints:: common.HuggingfacePipVersionConstraints,

  // DEPRECATED: Use PyTorchTpuVmMixin instead
  tpu_vm_r2_8_install: self.PyTorchTpuVmMixin.tpuSettings.tpuVmPytorchSetup,
}
