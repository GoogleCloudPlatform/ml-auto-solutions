KUEUE_NAME = "a3-ultra"
OPTIMIZER = "adam"
NUM_STEPS = 20


class Optimizer:
  ADAM = "adam"
  ADAMW = "adamw"


class Schedule:
  """Class containing schedule cron expressions (PST converted to UTC)."""

  # Daily at 6:00 PM PST except Thursday => 2:00 AM UTC next day
  DAILY_PST_6PM_EXCEPT_THURSDAY = "0 2 * * 1,2,3,4,6,0"
  DAILY_PST_6_30PM_EXCEPT_THURSDAY = "30 2 * * 1,2,3,4,6,0"
  DAILY_PST_7PM_EXCEPT_THURSDAY = "0 3 * * 1,2,3,4,6,0"
  DAILY_PST_7_30PM_EXCEPT_THURSDAY = "30 3 * * 1,2,3,4,6,0"

  # Daily at 6:00 PM PDT except Thursday => 1:00 AM UTC next day
  DAILY_PDT_6PM_EXCEPT_THURSDAY = "0 1 * * 1,2,3,4,6,0"
  DAILY_PDT_6_30PM_EXCEPT_THURSDAY = "30 1 * * 1,2,3,4,6,0"
  DAILY_PDT_7PM_EXCEPT_THURSDAY = "0 2 * * 1,2,3,4,6,0"
  DAILY_PDT_7_30PM_EXCEPT_THURSDAY = "30 2 * * 1,2,3,4,6,0"
  DAILY_PDT_8PM_EXCEPT_THURSDAY = "0 3 * * 1,2,3,4,6,0"
  DAILY_PDT_8_30PM_EXCEPT_THURSDAY = "30 3 * * 1,2,3,4,6,0"
  DAILY_PDT_9PM_EXCEPT_THURSDAY = "0 4 * * 1,2,3,4,6,0"


class Image:
  MAXTEXT_JAX_STABLE_NIGHTLY = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_stable_stack_nightly_jax"
  )
  MAXTEXT_JAX_STABLE_RELEASE = (
      "gcr.io/tpu-prod-env-multipod/maxtext_gpu_jax_stable_stack"
  )
  MAXTEXT_JAX_STABLE_NIGHTLY_OLD = "gcr.io/supercomputer-testing/jax3p_nightly"
  MAXTEXT_JAX_STABLE_RELEASE_OLD = "gcr.io/supercomputer-testing/jax3p_stable"
