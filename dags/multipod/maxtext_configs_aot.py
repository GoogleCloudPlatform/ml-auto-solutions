# Copyright 2024 Google LLC
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

"""
A DAG to run AOT compilation tests for MaxText model configs.
"""
import datetime
from airflow import models
from dags import composer_env
from dags.vm_resource import TpuVersion, Zone
from dags.multipod.configs import maxtext_gce_config
from dags.multipod.configs.common import SetupMode

# Run once a day at 6 am UTC (10 pm PST)
SCHEDULED_TIME = "0 5 * * *" if composer_env.is_prod_env() else None

with models.DAG(
    dag_id="maxtext_configs_aot",
    schedule=None,
    tags=["multipod_team", "maxtext", "stable", "nightly"],
    start_date=datetime.datetime(2024, 2, 19),
    catchup=False,
    concurrency=2,
) as dag:
  # Testing configurations
  model_configs = {
      # accelerator: [(model_size, num_cores), ...],
      "v4": [("22b", 128), ("52b", 384)],
      "v5e": [("16b", 256), ("32b", 256), ("64b", 256), ("128b", 256)],
      "v5p": [
          ("32b", 128),
          ("64b", 128),
          ("128b", 256),
          ("128b", 512),
          ("256b", 1024),
          ("512b", 1024),
          ("1024b", 2048),
          ("1024b", 4096),
      ],
  }
  num_slices = [1, 2]
  test_modes = [SetupMode.STABLE, SetupMode.NIGHTLY]

  run_model_cmds_dict = {}
  for tpu, models in model_configs.items():
    run_model_cmds = ["cd /tmp/maxtext"]
    for model_size, num_cores in models:
      for n in num_slices:
        cmd = f"bash MaxText/configs/{tpu}/{model_size}.sh EXECUTABLE=train_compile.py M_COMPILE_TOPOLOGY={tpu}-{num_cores} M_COMPILE_TOPOLOGY_NUM_SLICES={n}"
        run_model_cmds.append(cmd)
    run_model_cmds_dict[tpu] = run_model_cmds

  for mode in test_modes:
    maxtext_v4_configs_test = maxtext_gce_config.get_maxtext_configs_aot_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        is_tpu_reserved=False,
        test_name=f"maxtext-aot-v4-configs-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v4"],
        test_mode=mode,
    ).run()

    maxtext_v5e_configs_test = maxtext_gce_config.get_maxtext_configs_aot_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        is_tpu_reserved=False,
        test_name=f"maxtext-aot-v5e-configs-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v5e"],
        test_mode=mode,
    ).run()

    maxtext_v5p_configs_test = maxtext_gce_config.get_maxtext_configs_aot_config(
        tpu_version=TpuVersion.V4,
        tpu_cores=8,
        tpu_zone=Zone.US_CENTRAL2_B.value,
        time_out_in_min=60,
        is_tpu_reserved=False,
        test_name=f"maxtext-aot-v5p-configs-{mode.value}",
        run_model_cmds=run_model_cmds_dict["v5p"],
        test_mode=mode,
    ).run()

    maxtext_v4_configs_test >> maxtext_v5e_configs_test >> maxtext_v5p_configs_test
