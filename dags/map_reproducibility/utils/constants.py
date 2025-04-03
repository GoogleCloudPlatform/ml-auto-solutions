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
  DAILY__PST_6_30PM_EXCEPT_THURSDAY = "30 2 * * 1,2,3,4,6,0"
  DAILY_PST_7PM_EXCEPT_THURSDAY = "0 3 * * 1,2,3,4,6,0"
  DAILY_PST_7_30PM_EXCEPT_THURSDAY = "30 3 * * 1,2,3,4,6,0"
