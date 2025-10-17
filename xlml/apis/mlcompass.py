"""MLCompass integration library."""

import dataclasses
import os
from typing import TYPE_CHECKING, Any, Optional, cast, Dict, List
from pendulum.datetime import DateTime
from airflow.models import taskmixin, skipmixin, baseoperator, abstractoperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.utils.context import Context
from airflow.utils.task_group import TaskGroup
import dataclasses_json

_STATE_BUCKET_NAME = 'mlcompass-jax-artifacts'
_STATE_OBJECT_PATH_TEMPLATE = 'xlml/{uuid}/mlcompass_state.json'


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class CommitInfo:
  repo: str
  commit: str
  branch: str


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class BenchmarkData:
  """Subset of MLCompass benchmark data."""

  test_name: str
  xlml_node_id: str
  commit_map: Optional[Dict[str, CommitInfo]] = None


@dataclasses_json.dataclass_json
@dataclasses.dataclass
class MLCompassState:
  """State for running XLML benchmarks."""

  state_uuid: Optional[str] = None
  mlcompass_tracking_id: Optional[str] = None
  execution_mode: Optional[str] = None
  environment_name: Optional[str] = None
  dag_id: Optional[str] = None
  dag_run_id: Optional[str] = None
  airflow_gcp_project: Optional[str] = None
  airflow_gcp_location: Optional[str] = None
  workdir_bucket: Optional[str] = None
  workdir_path: Optional[str] = None
  benchmarks: List[BenchmarkData] = dataclasses.field(default_factory=list)


def get_bucket_name(context: Optional[dict[str, Any]]) -> str:
  if context and (params := context.get('params')):
    return params.get('mlcompass_workdir_bucket', _STATE_BUCKET_NAME)


def get_state_uuid(context: Optional[dict[str, Any]]) -> Optional[str]:
  if not context:
    return None
  return context.get('params', {}).get('mlcompass_state_uuid')


def load_state(context: Optional[dict[str, Any]]) -> Optional[MLCompassState]:
  state_uuid = get_state_uuid(context)
  if state_uuid:
    gcs_hook = GCSHook()
    content = gcs_hook.download(
        bucket_name=get_bucket_name(context),
        object_name=_STATE_OBJECT_PATH_TEMPLATE.format(uuid=state_uuid),
    )
    state = MLCompassState.from_json(content)
    print(f'Loaded MLCompass state: {content}')
    state.dag_run_id = context['run_id'] if context else None
    state.airflow_gcp_project = state.airflow_gcp_project or os.getenv(
        'GCP_PROJECT'
    )
    state.airflow_gcp_location = state.airflow_gcp_location or os.getenv(
        'COMPOSER_LOCATION'
    )
    return state
  return None


def get_all_tasks(node: taskmixin.DAGNode) -> list[taskmixin.DAGNode]:
  """Get all descendant nodes of a given DAGNode."""
  if isinstance(node, abstractoperator.AbstractOperator):
    return [node]
  if not isinstance(node, TaskGroup):
    raise ValueError(
        f'Input node is not a TaskGroup or an AbstractOperator: {type(node)}'
    )
  result = []
  groups_to_visit = [node]
  while groups_to_visit:
    visiting = groups_to_visit.pop(0)
    for child in visiting.children.values():
      if isinstance(child, abstractoperator.AbstractOperator):
        result.append(child)
      elif isinstance(child, TaskGroup):
        groups_to_visit.append(child)
      else:
        raise ValueError(
            f'Encountered a DAGNode that is not a TaskGroup or an AbstractOperator: {type(child)}'
        )
  return result


class ScheduleOperator(baseoperator.BaseOperator, skipmixin.SkipMixin):
  """An operator to schedule MLCompass benchmarks."""

  def __init__(
      self, *, node_map: dict[str, list[taskmixin.DAGNode]], **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self.node_map = node_map

  def execute(self, context: Context) -> None:
    state = load_state(context)
    if not state:
      self.log.info('No MLCompass state found; scheduling all benchmarks.')
      return
    self.log.info(f'Loaded MLCompass state: {state.to_json(indent=2)}')
    dag_run = context['dag_run']
    skip_tasks = []
    requested_node_ids = {
        benchmark.xlml_node_id for benchmark in state.benchmarks
    }
    for key, nodes in self.node_map.items():
      if key in requested_node_ids:
        self.log.info(f'Scheduling benchmark: {key}')
      else:
        self.log.info(f'Skipping benchmark: {key}')
        skip_tasks.extend(nodes)
    self.skip(
        dag_run=dag_run,
        execution_date=cast(DateTime, dag_run.execution_date),
        tasks=skip_tasks,
        map_index=context['ti'].map_index,
    )


class Scheduler:
  """A scheduler to register MLCompass benchmarks in a DAG."""

  def __init__(self) -> None:
    self.node_map = {}
    self.schedule = ScheduleOperator(
        task_id='mlcompass_schedule', node_map=self.node_map
    )

  def register(self, *node: taskmixin.DAGNode) -> None:
    for n in node:
      self.node_map[n.node_id] = get_all_tasks(n)
      self.schedule >> n
