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


DAG_CONFIGS_INFERENCE_MEGA = {
    "recipes/inference/a3mega/a3mega_llama2-70b_8gpus_bf16_maxtext.yaml": {
        "timeout_minutes": 40,
        "backfill_group_nightly": 1,
        "backfill_group_release": 1,
        "nightly_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
        "release_schedule": Schedule.WEEKDAY_PDT_12AM_EXCEPT_THURSDAY,
    },
}
