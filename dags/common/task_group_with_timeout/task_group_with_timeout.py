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

"""TaskGroupWithTimeout: timeout enforcement for Airflow TaskGroups."""

import logging
from datetime import datetime, timedelta, timezone

from airflow.exceptions import AirflowFailException
from airflow.models import BaseOperator
from airflow.models.mappedoperator import MappedOperator
from airflow.models.taskmixin import DAGNode
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.context import Context
from airflow.utils.task_group import TaskGroup
from airflow.utils.timeout import timeout as AirflowTimeout
from airflow.utils.trigger_rule import TriggerRule


class TaskGroupWithTimeout(TaskGroup):
  """A TaskGroup that enforces a per-task timeout.

  Each task in the group shares a single deadline: the first task to run
  sets the deadline to `now + timeout`, and each subsequent task receives
  only the time remaining until that deadline.

  This customized object is implemented by intercepting `TaskGroup`'s `.add()`
  at parsing phase, and wrapping `Task`'s `.execute()` to allow setting up a
  dynamic timeout value so that it can affect `.execute()` at runtime phase.

  Limitations:
    1. Dynamic Task Mapping: Tasks generated via `.expand()` return a
       `MappedOperator` which will unmap as a `BaseOperator` only at runtime;
       therefore, there's no `.execute()` method to wrap at parsing phase.
    2. Nested TaskGroups: The `.add()` interception only applies to direct
       children. Tasks placed inside a nested `TaskGroup` will bypass this
       parent group's customized wrapper and evade the shared timeout budget.

  Args:
    group_id: Unique identifier for this TaskGroup.
    timeout: Timeout as a timedelta (e.g. `timedelta(minutes=30)`).
    is_teardown: When `True`, the group runs even if an upstream group
      has failed — suitable for cleanup/teardown groups (e.g. a `post_test`
      group following a `testing` group). Defaults to `False`.
    **kwargs: Additional arguments passed to TaskGroup.
  """

  ROOT_TASK_ID = "provision_taskgroup_session"

  def __init__(
      self,
      group_id,
      timeout: timedelta,
      is_teardown: bool = False,
      **kwargs,
  ):
    super().__init__(group_id=group_id, **kwargs)
    self.group_name = f"{self.__class__.__name__}: '{group_id}'"
    self.timeout = timeout
    self.trigger_rule = (
        TriggerRule.ALL_DONE if is_teardown else TriggerRule.ALL_SUCCESS
    )
    self._root_node = None

  def __enter__(self):
    """Inject `_root_node` when entering the group context.

    Overridden because this group's timeout mechanism needs a single anchor
    task to record the start time; see class docstring for the full flow.
    """
    tg = super().__enter__()
    self._root_node = PythonOperator(
        task_id=self.ROOT_TASK_ID,
        python_callable=lambda: datetime.now(timezone.utc).isoformat(),
        trigger_rule=self.trigger_rule,
    )
    return tg

  def __exit__(self, *args):
    """Wire `_root_node` as upstream of in-group root children on context exit.

    Overridden to guarantee `_root_node` runs first. Only children with no
    in-group sibling upstream get a direct edge; others inherit transitively.
    """
    children_ids = set(self.children.keys())
    for child in self.children.values():
      if child is self._root_node:
        continue
      # If a sibling already chains into this child, the dependency on
      # _root_node is satisfied transitively — no need to add a direct edge.
      if child.upstream_task_ids & children_ids:
        continue
      child.set_upstream(self._root_node)
    return super().__exit__(*args)

  def add(self, node: DAGNode):
    node = super().add(node)

    match node:
      case TaskGroup():
        # Tasks inside a nested TaskGroup will skip this parent's logic.
        # This means they will escape the shared timeout limit.
        # To prevent this, we intentionally block nested TaskGroups here.
        #
        # TODO: support nested TaskGroupWithTimeout
        raise AirflowFailException(
            f"{self.__class__.__name__} does not support nested TaskGroups"
        )

      case MappedOperator():
        # Mapped tasks don't have an `.execute()` method at this stage.
        # This means they will escape the shared timeout limit.
        # To prevent this, we intentionally block MappedOperators here.
        raise AirflowFailException(
            f"{self.__class__.__name__} does not support Dynamic Task Mapping"
        )

      case BaseOperator() if node.task_id.endswith(f".{self.ROOT_TASK_ID}"):
        # Skip the root node, which only initiates the session of this task
        # group and requires no interception.
        return node

      case BaseOperator():
        # Use the unbound method so `self` binds at execution time, after
        # Airflow resolves XComArg placeholders. Binding via `node.execute` at
        # the parsing phase leaks unresolved placeholders into XCom and breaks
        # DAG serialization.
        original_execute = type(node).execute

        group_name = self.group_name
        timeout = self.timeout
        root_node_id = self._root_node.task_id

        def wrapped_execute(context: Context):
          task_instance = context.get("task_instance")

          start_time_str = task_instance.xcom_pull(task_ids=root_node_id)
          if not start_time_str:
            raise AirflowFailException(
                "Failed to overwrite timeout for task: "
                f"{group_name} session wasn't initiated."
            )

          start_time = datetime.fromisoformat(start_time_str)
          deadline = start_time + timeout
          remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
          if remaining <= 0:
            raise AirflowFailException(f"{group_name} timeout exceeded")

          task = task_instance.task

          # Take the minimum value as the effective timeout to ensure all tasks
          # are strictly bounded under this task group's shared deadline.
          effective_timeout_sec = min(remaining, _determine_task_timeout(task))
          logging.info(
              f"{group_name}; "
              f"task: '{task_instance.task_id}'; "
              f"effective timeout: {effective_timeout_sec}s"
          )

          # Group-budget exhaustion is enforced by the `remaining <= 0` check
          # above on the next retry; let AirflowTaskTimeout propagate normally.
          with AirflowTimeout(seconds=int(effective_timeout_sec)):
            return original_execute(task, context)

        node.execute = wrapped_execute
        return node


def _determine_task_timeout(task: BaseOperator) -> float:
  """
  Determines the effective timeout for a task by identifying which limit
  triggers first.

  This method centralizes the logic for various operator types.
  - For sensors, it resolves the potential overlap between sensor-specific
    timeouts and general execution timeouts.
  - For standard operators, it takes "inf" as the value when no limit is
    set, which aligns with the API's behavior of allowing unlimited
    execution.
  """
  # Since Airflow treats an unset `execution_timeout` as unlimited,
  # we take "inf" as its value to align with this behavior
  is_set = task.execution_timeout is not None
  inf = float("inf")
  timeout_1 = task.execution_timeout.total_seconds() if is_set else inf

  if isinstance(task, BaseSensorOperator):
    # This attribute has a default value stored in the configuration file;
    # therefore, `timeout` will always be set.
    timeout_2 = task.timeout
    return min(timeout_1, timeout_2)

  return timeout_1
