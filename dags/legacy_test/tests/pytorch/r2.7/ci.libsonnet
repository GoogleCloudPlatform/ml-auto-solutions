local experimental = import '../experimental.libsonnet';
local common = import 'common.libsonnet';
local tpus = import 'templates/tpus.libsonnet';
local timeouts = import 'templates/timeouts.libsonnet';

// Runs the same script we use in our TPU CI, nightly.
// TODO: Remove this and run all tests in CI.
{
  local ci = self.ci,
  ci:: common.PyTorchTest + common.Functional {
    modelName: 'ci',
    command: [
      'bash',
      '-c',
      |||
        cd pytorch/xla

        test/tpu/run_tests.sh
      |||,
    ],

  } + timeouts.Hours(3),
  local pjrt = self.pjrt,
  pjrt:: common.PyTorchTpuVmMixin {
    tpuSettings+: {
      tpuVmExtraSetup: |||
        pip install expecttest==0.1.6 rich
        pip install torch_xla[pallas] -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_releases.html
      |||,
    },
  },

  local v5litepod_4 = self.v5litepod_4,
  v5litepod_4:: {
    accelerator: tpus.v5litepod_4,
  },

  local trillium_4 = self.trillium_4,
  trillium_4:: {
    tpuSettings+: {
      softwareVersion: 'v2-alpha-tpuv6e',
    },
    accelerator: tpus.trillium_4,
  },

  configs: [
    ci + v5litepod_4 + pjrt,
    ci + pjrt + trillium_4,
  ],
}
