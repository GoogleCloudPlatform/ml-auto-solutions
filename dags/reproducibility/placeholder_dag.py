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

import datetime
from airflow import models
from airflow.operators.bash import BashOperator
from dags.reproducibility.configs.simple_config import get_simple_config

with models.DAG(
    dag_id="reproducibility_simple_dag",
    schedule=None,
    tags=["simple", "aotc", "nightly", "reproducibility", "experimental", "xlml"],
    start_date=datetime.datetime(2024, 10, 22),
    catchup=False,
) as dag:
  t1 = BashOperator(
      task_id="print_env",
  )

  simple = get_simple_config().run()
