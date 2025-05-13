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

from dags.map_reproducibility.utils.constants import Schedule

DAG_CONFIGS_MEGA = {
    "recipes/a3mega/a3mega_llama3.1-8b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_llama3.1-8b_8gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_llama3.1-8b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_llama3.1-8b_16gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_mixtral-8x7b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_mixtral-8x7b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_llama3.1-70b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 25,
        "backfill_group_nightly": 2,
        "backfill_group_release": 2,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12_30AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12_30AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_llama3.1-70b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 20,
        "backfill_group_nightly": 3,
        "backfill_group_release": 3,
        "nightly_schedule": Schedule.WEEKDAY_PDT_1AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_1AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_llama3.1-405b_512gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 40,
        "backfill_group_nightly": 4,
        "backfill_group_release": 5,
        "nightly_schedule": Schedule.WEEKDAY_PDT_1_30AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_2AM_EXCEPT_THURSDAY,
    },
    "recipes/a3mega/a3mega_llama3.1-405b_512gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 50,
        "backfill_group_nightly": 6,
        "backfill_group_release": 7,
        "nightly_schedule": Schedule.WEEKDAY_PDT_2_30AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_3AM_EXCEPT_THURSDAY,
    },
}

DAG_CONFIGS_ULTRA = {
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_mixtral-8x7b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_mixtral-8x7b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 20,
        "backfill_group_nightly": 2,
        "backfill_group_release": 2,
        "schedule": Schedule.WEEKDAY_PDT_12_30AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 3,
        "backfill_group_release": 3,
        "schedule": Schedule.WEEKDAY_PDT_1AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 30,
        "backfill_group_nightly": 4,
        "backfill_group_release": 4,
        "schedule": Schedule.WEEKDAY_PDT_1_30AM_EXCEPT_THURSDAY,
    },
    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 40,
        "backfill_group_nightly": 5,
        "backfill_group_release": 5,
        "schedule": Schedule.WEEKDAY_PDT_2AM_EXCEPT_THURSDAY,
    },
}


DAG_CONFIGS_A4 = {
    "recipes/a4/a4_llama3.1-8b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_llama3.1-8b_8gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_llama3.1-8b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_llama3.1-8b_16gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_mixtral-8x7b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_mixtral-8x7b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_llama3.1-70b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 20,
        "backfill_group_nightly": 2,
        "backfill_group_release": 2,
        "schedule": Schedule.WEEKDAY_PDT_12_30AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_llama3.1-70b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 3,
        "backfill_group_release": 3,
        "schedule": Schedule.WEEKDAY_PDT_1AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_llama3.1-405b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 30,
        "backfill_group_nightly": 4,
        "backfill_group_release": 4,
        "schedule": Schedule.WEEKDAY_PDT_1_30AM_EXCEPT_THURSDAY,
    },
    "recipes/a4/a4_llama3.1-405b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 40,
        "backfill_group_nightly": 5,
        "backfill_group_release": 5,
        "schedule": Schedule.WEEKDAY_PDT_2AM_EXCEPT_THURSDAY,
    },
}

DAG_CONFIGS_A4_NEMO = {
    "recipes/a4/nemo/a4_llama3.1-70b_256gpus_fp8_nemo.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.SATURDAY_PDT_12AM,
        "release_schedule": Schedule.SATURDAY_PDT_12AM,
    }
}
