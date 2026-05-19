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

"""TaskGroupWithTimeout: a TaskGroup that enforces a shared deadline timeout.

Usage:
  from dags.common.task_group_with_timeout import TaskGroupWithTimeout
"""

from dags.common.task_group_with_timeout.task_group_with_timeout import (
    TaskGroupWithTimeout,
)
