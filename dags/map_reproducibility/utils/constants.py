KUEUE_NAME = "a3-ultra"
OPTIMIZER = "adam"
NUM_STEPS = 15


class Optimizer:
  ADAM = "adam"
  ADAMW = "adamw"


class Schedule:
  """Class containing schedule cron expressions (PST converted to UTC)."""

  # WEEKDAY at 6:00 PM PST except Thursday => 2:00 AM UTC next day
  WEEKDAY_PST_6PM_EXCEPT_THURSDAY = "0 2 * * 2,3,4,6"
  WEEKDAY_PST_6_30PM_EXCEPT_THURSDAY = "30 2 * * 2,3,4,6"
  WEEKDAY_PST_7PM_EXCEPT_THURSDAY = "0 3 * * 2,3,4,6"
  WEEKDAY_PST_7_30PM_EXCEPT_THURSDAY = "30 3 * * 2,3,4,6"

  # WEEKDAY at 6:00 PM PDT except Thursday => 1:00 AM UTC next day
  WEEKDAY_PDT_6PM_EXCEPT_THURSDAY = "0 1 * * 2,3,4,6"
  WEEKDAY_PDT_6_30PM_EXCEPT_THURSDAY = "30 1 * * 2,3,4,6"
  WEEKDAY_PDT_7PM_EXCEPT_THURSDAY = "0 2 * * 2,3,4,6"
  WEEKDAY_PDT_7_30PM_EXCEPT_THURSDAY = "30 2 * * 2,3,4,6"
  WEEKDAY_PDT_8PM_EXCEPT_THURSDAY = "0 3 * * 2,3,4,6"
  WEEKDAY_PDT_8_30PM_EXCEPT_THURSDAY = "30 3 * * 2,3,4,6"
  WEEKDAY_PDT_9PM_EXCEPT_THURSDAY = "0 4 * * 2,3,4,6"
  WEEKDAY_PDT_9_30PM_EXCEPT_THURSDAY = "30 4 * * 2,3,4,6"

  # WEEKDAY at 12:00 AM PDT except Thursday => 1:00 AM UTC next day
  WEEKDAY_PDT_12AM_EXCEPT_THURSDAY = "0 7 * * 2,3,4,6"
  WEEKDAY_PDT_12_30AM_EXCEPT_THURSDAY = "30 7 * * 2,3,4,6"
  WEEKDAY_PDT_1AM_EXCEPT_THURSDAY = "0 8 * * 2,3,4,6"
  WEEKDAY_PDT_1_30AM_EXCEPT_THURSDAY = "30 8 * * 2,3,4,6"
  WEEKDAY_PDT_2AM_EXCEPT_THURSDAY = "0 9 * * 2,3,4,6"
  WEEKDAY_PDT_2_30AM_EXCEPT_THURSDAY = "30 9 * * 2,3,4,6"
  WEEKDAY_PDT_3AM_EXCEPT_THURSDAY = "0 10 * * 2,3,4,6"
  WEEKDAY_PDT_3_30AM_EXCEPT_THURSDAY = "30 10 * * 2,3,4,6"
  WEEKDAY_PDT_6AM_7AM_EXCEPT_THURSDAY = "0 13,14 * * 2,3,4,6"

  SATURDAY_PDT_12AM = "0 7 * * 6"


class Image:
  MAXTEXT_JAX_STABLE_NIGHTLY = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_stable_stack_nightly_jax"
  )
  MAXTEXT_JAX_STABLE_RELEASE = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_stable_stack"
  )
  MAXTEXT_JAX_STABLE_NIGHTLY_OLD = "gcr.io/supercomputer-testing/jax3p_nightly"
  MAXTEXT_JAX_STABLE_RELEASE_OLD = "gcr.io/supercomputer-testing/jax3p_stable"
  NEMO_STABLE_RELEASE_A3U = "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo24.07-gib1.0.3-A3U"
  NEMO_STABLE_RELEASE_A4 = "us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.02-gib1.0.5-A4"
