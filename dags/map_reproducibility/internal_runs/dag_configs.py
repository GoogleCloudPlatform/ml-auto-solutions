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
        "nightly_schedule": "45 10 * * 2,3,4,6",
        "release_schedule": "45 7 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_llama3.1-8b_8gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "45 20 * * 2,3,4,6",
        "release_schedule": "45 9 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_llama3.1-8b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "15 12 * * 2,3,4,6",
        "release_schedule": "15 8 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_llama3.1-8b_16gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "0 13 * * 2,3,4,6",
        "release_schedule": "15 9 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_mixtral-8x7b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "30 11 * * 2,3,4,6",
        "release_schedule": "45 8 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_mixtral-8x7b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "30 21 * * 2,3,4,6",
        "release_schedule": "15 10 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_llama3.1-70b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 25,
        "backfill_group_nightly": 2,
        "backfill_group_release": 2,
        "nightly_schedule": "15 14 * * 2,3,4,6",
        "release_schedule": "45 13 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_llama3.1-70b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 20,
        "backfill_group_nightly": 3,
        "backfill_group_release": 3,
        "nightly_schedule": "30 15 * * 2,3,4,6",
        "release_schedule": "0 15 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_llama3.1-405b_512gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 40,
        "backfill_group_nightly": 4,
        "backfill_group_release": 5,
        "nightly_schedule": "0 16 * * 2,3,4,6",
        "release_schedule": "0 17 * * 2,3,4,6",
    },
    "recipes/a3mega/a3mega_llama3.1-405b_512gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 50,
        "backfill_group_nightly": 6,
        "backfill_group_release": 7,
        "nightly_schedule": "45 17 * * 2,3,4,6",
        "release_schedule": "45 18 * * 2,3,4,6",
    },
}

DAG_CONFIGS_ULTRA = {
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "45 5 * * 2,3,4,6",
        "release_schedule": "45 2 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_8gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "0 8 * * 2,3,4,6",
        "release_schedule": "30 6 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "15 10 * * 2,3,4,6",
        "release_schedule": "0 2 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_llama3.1-8b_16gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "45 8 * * 2,3,4,6",
        "release_schedule": "0 5 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_mixtral-8x7b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "15 7 * * 2,3,4,6",
        "release_schedule": "30 9 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_mixtral-8x7b_16gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": "15 4 * * 2,3,4,6",
        "release_schedule": "30 3 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 20,
        "backfill_group_nightly": 2,
        "backfill_group_release": 2,
        "nightly_schedule": "0 11 * * 2,3,4,6",
        "release_schedule": "30 11 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_llama3.1-70b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 15,
        "backfill_group_nightly": 3,
        "backfill_group_release": 3,
        "nightly_schedule": "45 12 * * 2,3,4,6",
        "release_schedule": "0 12 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_fp8_maxtext.yaml": {
        "timeout_minutes": 30,
        "backfill_group_nightly": 4,
        "backfill_group_release": 4,
        "nightly_schedule": "30 13 * * 2,3,4,6",
        "release_schedule": "15 14 * * 2,3,4,6",
    },
    "recipes/a3ultra/a3ultra_llama3.1-405b_256gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 40,
        "backfill_group_nightly": 5,
        "backfill_group_release": 5,
        "nightly_schedule": "45 9 * * 2,3,4,6",
        "release_schedule": "0 15 * * 2,3,4,6",
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

DAG_CONFIGS_ULTRA_NEMO = {
    "recipes/a3ultra/nemo/a3ultra_llama3.1-8b_8gpus_fp8_nemo.yaml": {
        "timeout_minutes": 20,
        "release_schedule": "30 2 * * 6",
    },
    "recipes/a3ultra/nemo/a3ultra_llama3.1-8b_8gpus_bf16_nemo.yaml": {
        "timeout_minutes": 20,
        "release_schedule": "0 2 * * 6",
    },
    "recipes/a3ultra/nemo/a3ultra_llama3.1-70b_256gpus_fp8_nemo.yaml": {
        "timeout_minutes": 25,
        "release_schedule": "45 10 * * 6",
    },
}
