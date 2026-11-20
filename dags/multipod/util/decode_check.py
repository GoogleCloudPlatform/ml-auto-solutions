# Copyright 2026 Google LLC
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

"""Checks the text that MaxText decode steps print in E2E workloads.

The MaxText E2E test scripts decode a fixed prompt before and after training.
A decode step exits with 0 even when the model prints garbage, so the E2E DAGs
run `check_decode_output` after each training workload. It reads the decode
outputs back from the workload's Cloud Logging entries and checks that each
one is text: not empty, without replacement characters, not the same words
over and over and, for the fixed prompts in `_EXPECTED_TOPICS`, on topic. The
checks are loose on purpose, so hardware or sharding changes that alter the
exact wording don't trip them.

It parses the formats that MaxText prints:

- `maxtext.inference.decode`: Input `<prompt>` -> `<output>`
- `maxtext.inference.vllm_decode`: Prompt: <prompt>, Generated text: <output>
- `maxtext.inference.vllm_decode --use_tunix`: Prompt: <prompt>, then
  Output: <output> in a second log record

Cloud Logging stores each line of a message as its own entry, so the lines
that one print or log call wrote are joined first.
"""

import collections
import dataclasses
import datetime
import re
import time
from typing import Any, Callable, NoReturn, Sequence

from absl import logging
from airflow.decorators import task
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.models.xcom_arg import XComArg
from airflow.operators.python import get_current_context
from airflow.utils.task_group import TaskGroup
from google.cloud import logging_v2

from xlml.utils import gcp_util
from xlml.utils import gke
from xlml.utils.time_util import TimeUtil

# Entries that hold the start of a decode output. The logging agent stores
# lines with an absl header, such as the first line of a vllm_decode message,
# as jsonPayload and every other line as textPayload.
_MARKER_FILTER = (
    '(textPayload:"Input `" OR textPayload:"Generated text:"'
    ' OR jsonPayload.message:"Generated text:"'
    ' OR jsonPayload.message=~"^Output: ")'
)

# The lines of one message reach Cloud Logging a few microseconds apart, while
# the next message comes at least milliseconds later.
_MAX_GAP_IN_MESSAGE = datetime.timedelta(milliseconds=50)

# How much log to read around a marker entry to rebuild its whole message.
_TIME_BEFORE_MARKER = datetime.timedelta(seconds=2)
_TIME_AFTER_MARKER = datetime.timedelta(seconds=10)

_MAX_MARKERS = 1000
_MAX_LINES_AROUND_MARKER = 5000

# Each E2E test script decodes once before and once after training.
_EXPECTED_DECODES = 2

# Log ingestion can lag behind the end of the workload.
_READ_ATTEMPTS = 3
_SECONDS_BETWEEN_READS = 60

# Margin before the DAG run's logical date when reading the workload's logs.
_TIME_BEFORE_DAG_RUN = datetime.timedelta(minutes=10)
_DEFAULT_LOOKBACK = datetime.timedelta(days=1)

_DECODE_RE = re.compile(
    r"^Input `(?P<prompt>.*?)` -> `(?P<output>.*?)`$", re.DOTALL | re.MULTILINE
)
_VLLM_DECODE_RE = re.compile(
    r"(?:Prompt: (?P<prompt>.*?), )?Generated text: (?P<output>.*)", re.DOTALL
)
# With --use_tunix, the prompt is in the record before.
_TUNIX_DECODE_RE = re.compile(r"\AOutput: (?P<output>.*)", re.DOTALL)
_PROMPT_PREFIX = "Prompt: "

# Lines that other loggers write and that end a decode output, e.g.
# "(MainProcess pid=147) WARNING 11-19 03:49:47 [dp_scheduler.py:1684] ...".
_LOG_RECORD_RE = re.compile(
    r"^(?:\(\w+ pid=\d+\) )?"
    r"(?:[IWEF]\d{4} \d\d:\d\d:\d\d"
    r"|(?:DEBUG|INFO|WARNING|ERROR|CRITICAL)\b"
    r"|\d{4}-\d\d-\d\d[ T]\d\d:\d\d:\d\d)"
)

_WORD_RE = re.compile(r"\w+(?:['’]\w+)*")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# An output is degenerate if one word appears this many times in a row ...
_MIN_REPEATS_OF_WORD = 4
# ... or a phrase of up to this many words appears this many times in a row.
_MAX_WORDS_IN_REPEATED_PHRASE = 50
_MIN_REPEATS_OF_PHRASE = 3
_MAX_WORDS_TO_QUOTE = 10
# Long outputs with fewer distinct words than this share are degenerate too.
_MIN_WORDS_FOR_DIVERSITY = 20
_MIN_DISTINCT_WORD_SHARE = 0.2

_PROMPT_EXCERPT_CHARS = 120
_OUTPUT_EXCERPT_CHARS = 300


@dataclasses.dataclass(frozen=True)
class _ExpectedTopic:
  """Words that the output for a fixed prompt should contain at least one of."""

  prompt: re.Pattern[str]
  description: str
  keywords: re.Pattern[str]


_EXPECTED_TOPICS = (
    # Thinking models can use up all the output tokens before they name a
    # landmark, so mentioning London is enough.
    _ExpectedTopic(
        prompt=re.compile(r"famous landmarks in London", re.IGNORECASE),
        description="London or a London landmark",
        keywords=re.compile(
            r"london|big ben|buckingham|westminster|tower bridge|thames"
            r"|british museum|st\.? paul|trafalgar|parliament|hyde park",
            re.IGNORECASE,
        ),
    ),
    _ExpectedTopic(
        prompt=re.compile(r"describe (?:the |this )?image", re.IGNORECASE),
        description="what the image shows",
        keywords=re.compile(
            r"image|photo|picture|skyline|city|building|scene", re.IGNORECASE
        ),
    ),
)


@dataclasses.dataclass(frozen=True)
class LogLine:
  """One Cloud Logging entry, which holds one line of a container's output.

  Attributes:
    timestamp: When the line was written.
    text: The line, without the trailing newline.
    pod: The pod that wrote the line.
    container: The container that wrote the line.
    stream: The log name, e.g. `projects/<project>/logs/stdout`.
    starts_record: Whether the logging agent parsed a log header on this line,
      which makes it the first line of a new log record.
  """

  timestamp: datetime.datetime
  text: str
  pod: str = ""
  container: str = ""
  stream: str = ""
  starts_record: bool = False


@dataclasses.dataclass(frozen=True)
class DecodeOutput:
  """The prompt and the output of one decode step."""

  prompt: str
  output: str
  pod: str
  timestamp: datetime.datetime


def to_log_line(entry: Any) -> LogLine:
  """Converts a `google.cloud.logging` entry to a `LogLine`."""
  payload = entry.payload
  starts_record = isinstance(payload, dict)
  if starts_record:
    payload = payload.get("message", "")
  labels = (entry.resource.labels if entry.resource else None) or {}
  return LogLine(
      timestamp=entry.timestamp,
      text=str(payload or "").rstrip("\n"),
      pod=labels.get("pod_name", ""),
      container=labels.get("container_name", ""),
      stream=entry.log_name or "",
      starts_record=starts_record,
  )


def message_bounds(lines: Sequence[LogLine], index: int) -> tuple[int, int]:
  """Finds the message that `lines[index]` belongs to.

  Args:
    lines: The lines of one container stream, in time order.
    index: The index of a line in the message.

  Returns:
    The indices of the first and the last line that the same print or log
    call wrote.
  """

  def same_message(earlier: LogLine, later: LogLine) -> bool:
    return (
        not later.starts_record
        and later.timestamp - earlier.timestamp <= _MAX_GAP_IN_MESSAGE
    )

  start = index
  while start > 0 and same_message(lines[start - 1], lines[start]):
    start -= 1
  end = index
  while (
      end + 1 < len(lines)
      and same_message(lines[end], lines[end + 1])
      and not _LOG_RECORD_RE.match(lines[end + 1].text)
  ):
    end += 1
  return start, end


def parse_decode_output(message: str) -> tuple[str, str] | None:
  """Returns the prompt and the output in a decode message, if it is one.

  The prompt is empty for the `--use_tunix` format, which logs it in the
  record before; see `prompt_before`.
  """
  for pattern in (_DECODE_RE, _VLLM_DECODE_RE, _TUNIX_DECODE_RE):
    match = pattern.search(message)
    if match:
      prompt = match.groupdict().get("prompt") or ""
      return prompt.strip(), match.group("output")
  return None


def prompt_before(lines: Sequence[LogLine], start: int) -> str:
  """Returns the prompt if the record before `lines[start]` logs one."""
  for i in range(start - 1, -1, -1):
    if lines[i].starts_record:
      if not lines[i].text.startswith(_PROMPT_PREFIX):
        return ""
      first, last = message_bounds(lines, i)
      prompt = "\n".join(line.text for line in lines[first : last + 1])
      return prompt[len(_PROMPT_PREFIX) :].strip()
  return ""


def workload_log_filter(
    cluster_name: str, namespace: str, workload_id: str
) -> str:
  """Returns a Cloud Logging filter for the containers of a workload."""
  return (
      'resource.type="k8s_container"'
      f' AND resource.labels.cluster_name="{cluster_name}"'
      f' AND resource.labels.namespace_name="{namespace}"'
      ' AND labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"='
      f'"{workload_id}"'
  )


def read_decode_outputs(
    project_id: str,
    cluster_name: str,
    namespace: str,
    workload_id: str,
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    query: Callable[..., Sequence[Any]] = gcp_util.query_log_entries,
) -> list[DecodeOutput]:
  """Reads the decode outputs that a workload printed from Cloud Logging.

  Every host of a multi-host workload prints the same outputs, so only the
  pod with the most of them is read.

  Args:
    project_id: The project of the workload's GKE cluster.
    cluster_name: The name of the workload's GKE cluster.
    namespace: The Kubernetes namespace of the workload.
    workload_id: The workload ID, which is also the JobSet name.
    start_time: The start of the time range to read.
    end_time: The end of the time range to read.
    query: The function that reads log entries, for tests.

  Returns:
    The decode outputs in the order they were printed.
  """
  workload_filter = workload_log_filter(cluster_name, namespace, workload_id)
  markers = [
      to_log_line(entry)
      for entry in query(
          project_id=project_id,
          filter_str=f"{workload_filter} AND {_MARKER_FILTER}",
          start_time=TimeUtil.from_datetime(start_time),
          end_time=TimeUtil.from_datetime(end_time),
          order_by=logging_v2.ASCENDING,
          max_results=_MAX_MARKERS,
      )
  ]
  if not markers:
    return []

  markers_per_pod = collections.Counter(marker.pod for marker in markers)
  pod = min(markers_per_pod, key=lambda p: (-markers_per_pod[p], p))
  outputs = []
  message_starts = set()
  for marker in markers:
    if marker.pod != pod:
      continue
    lines = [
        to_log_line(entry)
        for entry in query(
            project_id=project_id,
            filter_str=(
                f'{workload_filter} AND resource.labels.pod_name="{pod}"'
                " AND resource.labels.container_name="
                f'"{marker.container}" AND logName="{marker.stream}"'
            ),
            start_time=TimeUtil.from_datetime(
                marker.timestamp - _TIME_BEFORE_MARKER
            ),
            end_time=TimeUtil.from_datetime(
                marker.timestamp + _TIME_AFTER_MARKER
            ),
            order_by=logging_v2.ASCENDING,
            max_results=_MAX_LINES_AROUND_MARKER,
        )
    ]
    index = next(
        (
            i
            for i, line in enumerate(lines)
            if line.timestamp == marker.timestamp and line.text == marker.text
        ),
        None,
    )
    if index is None:
      logging.warning(
          "Could not read the lines around %s in %s.", marker.timestamp, pod
      )
      continue
    start, end = message_bounds(lines, index)
    # Skips a second marker in a message that was already read.
    if (marker.stream, lines[start].timestamp) in message_starts:
      continue
    message_starts.add((marker.stream, lines[start].timestamp))
    parsed = parse_decode_output(
        "\n".join(line.text for line in lines[start : end + 1])
    )
    if parsed is None:
      continue
    prompt, output = parsed
    outputs.append(
        DecodeOutput(
            prompt=prompt or prompt_before(lines, start),
            output=output,
            pod=pod,
            timestamp=marker.timestamp,
        )
    )
  return outputs


def find_repeated_words(text: str) -> str | None:
  """Returns a word or phrase that `text` repeats in a row, if any."""
  words = [word.lower() for word in _WORD_RE.findall(text)]
  for size in range(1, _MAX_WORDS_IN_REPEATED_PHRASE + 1):
    repeats = _MIN_REPEATS_OF_WORD if size == 1 else _MIN_REPEATS_OF_PHRASE
    span = size * repeats
    for start in range(len(words) - span + 1):
      if all(
          words[i] == words[i + size] for i in range(start, start + span - size)
      ):
        phrase = words[start : start + size]
        if size > _MAX_WORDS_TO_QUOTE:
          return " ".join(phrase[:_MAX_WORDS_TO_QUOTE]) + " ..."
        return " ".join(phrase)
  return None


def find_problems(prompt: str, output: str) -> list[str]:
  """Returns what is wrong with a decode output, or nothing if it looks fine."""
  text = output.strip()
  if not text:
    return ["The output is empty."]
  problems = []
  if "\ufffd" in text:
    problems.append("The output has U+FFFD replacement characters.")
  if _CONTROL_CHAR_RE.search(text):
    problems.append("The output has control characters.")
  repeated = find_repeated_words(text)
  if repeated:
    problems.append(f'The output repeats "{repeated}".')
  words = [word.lower() for word in _WORD_RE.findall(text)]
  if len(words) >= _MIN_WORDS_FOR_DIVERSITY and len(
      set(words)
  ) < _MIN_DISTINCT_WORD_SHARE * len(words):
    problems.append(
        f"The output has only {len(set(words))} distinct words out of"
        f" {len(words)}."
    )
  for topic in _EXPECTED_TOPICS:
    if topic.prompt.search(prompt) and not topic.keywords.search(text):
      problems.append(f"The output doesn't mention {topic.description}.")
  return problems


def _excerpt(text: str, max_chars: int) -> str:
  text = " ".join(text.split())
  if len(text) > max_chars:
    text = text[: max_chars - 3] + "..."
  return repr(text)


def format_report(
    test_name: str,
    workload_id: str,
    results: Sequence[tuple[DecodeOutput, list[str]]],
    expected_decodes: int = _EXPECTED_DECODES,
) -> str:
  """Describes the decode outputs of a workload and their problems."""
  bad = sum(1 for _, problems in results if problems)
  lines = [
      f"{test_name}, workload {workload_id}: found {len(results)} decode"
      f" outputs (expected {expected_decodes}), {bad} with problems."
  ]
  if len(results) < expected_decodes:
    lines.append(
        f"- Missing decode outputs: found {len(results)}, expected"
        f" {expected_decodes} (before and after training)."
    )
  for number, (decode, problems) in enumerate(results, start=1):
    status = "PROBLEM" if problems else "OK"
    lines += [
        "",
        f"Decode {number} of {len(results)} ({status}), printed at"
        f" {decode.timestamp:%Y-%m-%d %H:%M:%S} UTC by {decode.pod}",
        f"  Prompt: {_excerpt(decode.prompt, _PROMPT_EXCERPT_CHARS)}",
        f"  Output: {_excerpt(decode.output, _OUTPUT_EXCERPT_CHARS)}",
    ]
    lines += [f"  - {problem}" for problem in problems]
  return "\n".join(lines)


def workload_id_of(gke_task_group: TaskGroup) -> XComArg:
  """Returns the workload ID that a `BaseRunnerTask.run()` group generates.

  `XpkTask` and `GclusterTask` generate the ID in
  `<group>.pre_process.generate_workload_id`.

  Args:
    gke_task_group: The task group that `BaseRunnerTask.run()` returned.

  Returns:
    The XCom of the task that generates the workload ID.
  """
  pre_process = gke_task_group.get_child_by_label("pre_process")
  return XComArg(pre_process.get_child_by_label("generate_workload_id"))


def _report_problem(message: str, blocking: bool) -> NoReturn:
  if blocking:
    raise AirflowFailException(message)
  logging.warning(
      "Skipping the task instead of failing it (warning-only mode):\n%s",
      message,
  )
  raise AirflowSkipException(message)


@task
def check_decode_output(
    workload_id: str,
    project_id: str,
    cluster_name: str,
    namespace: str,
    test_name: str,
    expected_decodes: int = _EXPECTED_DECODES,
    blocking: bool = False,
) -> None:
  """Checks the decode outputs that a MaxText E2E workload printed.

  In warning-only mode, the default, a problem marks the task as skipped
  rather than failed: it stands out in the grid, but it doesn't fail the DAG
  run or block the `latest` tag.

  Args:
    workload_id: The workload ID, from `workload_id_of`.
    project_id: The project of the workload's GKE cluster.
    cluster_name: The name of the workload's GKE cluster.
    namespace: The Kubernetes namespace of the workload.
    test_name: The test name to show in messages, e.g. "gemma3-4b sft".
    expected_decodes: How many decode outputs the test prints. Fewer is a
      problem.
    blocking: Whether a problem fails the task.
  """
  logical_date = get_current_context().get("logical_date")
  end_time = datetime.datetime.now(datetime.timezone.utc)
  start_time = (
      logical_date - _TIME_BEFORE_DAG_RUN
      if logical_date
      else end_time - _DEFAULT_LOOKBACK
  )
  logs = "Logs: " + gke.LOGGING_URL_FORMAT.format(
      project=project_id,
      cluster=cluster_name,
      namespace=namespace,
      workload_id=workload_id,
  )

  decodes = []
  error = None
  for attempt in range(1, _READ_ATTEMPTS + 1):
    try:
      decodes = read_decode_outputs(
          project_id=project_id,
          cluster_name=cluster_name,
          namespace=namespace,
          workload_id=workload_id,
          start_time=start_time,
          end_time=end_time,
      )
      error = None
    except Exception as e:  # pylint: disable=broad-exception-caught
      error = e
      logging.warning("Could not read the logs of %s: %s", workload_id, e)
    if attempt == _READ_ATTEMPTS or (
        error is None and len(decodes) >= expected_decodes
    ):
      break
    logging.info(
        "Found %d of %d decode outputs so far. Reading again in %d seconds.",
        len(decodes),
        expected_decodes,
        _SECONDS_BETWEEN_READS,
    )
    time.sleep(_SECONDS_BETWEEN_READS)
    end_time = datetime.datetime.now(datetime.timezone.utc)

  if error is not None and not decodes:
    if blocking:
      raise error
    _report_problem(
        f"{test_name}: could not read the logs of workload {workload_id}:"
        f" {error}\n{logs}",
        blocking,
    )
  if not decodes:
    _report_problem(
        f"{test_name}: found no decode output in the logs of workload"
        f" {workload_id}.\n{logs}",
        blocking,
    )

  results = [
      (decode, find_problems(decode.prompt, decode.output))
      for decode in decodes
  ]
  report = format_report(test_name, workload_id, results, expected_decodes)
  report = f"{report}\n\n{logs}"
  if len(decodes) < expected_decodes or any(
      problems for _, problems in results
  ):
    _report_problem(report, blocking)
  logging.info(report)
