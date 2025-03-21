KUEUE_NAME = "a3-ultra"
OPTIMIZER = "adam"
NUM_STEPS = 20


class Optimizer:
  ADAM = "adam"
  ADAMW = "adamw"


class Schedule:
  """Class containing schedule cron expressions."""

  DAILY_6PM_EXCEPT_THURSDAY = "0 18 * * 0,1,2,3,5,6"
  DAILY_6_30PM_EXCEPT_THURSDAY = "30 18 * * 0,1,2,3,5,6"
  DAILY_7PM_EXCEPT_THURSDAY = "0 19 * * 0,1,2,3,5,6"
  DAILY_7_30PM_EXCEPT_THURSDAY = "30 19 * * 0,1,2,3,5,6"
