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

"""Tests for dags.multipod.util.decode_check."""

import datetime
import re
import types
import unittest
from unittest import mock

from airflow import models
from airflow.exceptions import AirflowFailException, AirflowSkipException

from dags.common import test_owner
from dags.common.vm_resource import GkeClusters
from dags.multipod import maxtext_e2e_tpu_post_training
from dags.multipod import maxtext_e2e_tpu_pre_training
from dags.multipod.configs import gke_config
from dags.multipod.util import decode_check

_T0 = datetime.datetime(2026, 11, 19, 3, 46, 54, tzinfo=datetime.timezone.utc)
_LONDON_PROMPT = "Suggest some famous landmarks in London."
_IMAGE_PROMPT = (
    "<start_of_turn>user\nDescribe image <start_of_image><end_of_turn>\n"
    "<start_of_turn>model\n"
)


def _lines(texts, start=_T0, starts_record=()):
  """Returns lines 10 microseconds apart, as one print call writes them."""
  return [
      decode_check.LogLine(
          timestamp=start + datetime.timedelta(microseconds=10 * i),
          text=text,
          starts_record=i in starts_record,
      )
      for i, text in enumerate(texts)
  ]


def _message(lines, index):
  start, end = decode_check.message_bounds(lines, index)
  return "\n".join(line.text for line in lines[start : end + 1])


class ParseTest(unittest.TestCase):

  def test_parses_one_line_decode(self):
    self.assertEqual(
        decode_check.parse_decode_output(
            "Input `I love to` -> ` read, but I have to be careful with`"
        ),
        ("I love to", " read, but I have to be careful with"),
    )

  def test_parses_decode_over_many_lines(self):
    # gemma3-4b multimodal SFT on Nov 19, 2026.
    lines = _lines([
        "Input `<start_of_turn>user",
        "Describe image <start_of_image><end_of_turn>",
        "<start_of_turn>model",
        "` -> `Here's a description of the image:",
        "",
        "**Overall Impression:**",
        "",
        "The image is a bright, expansive cityscape view of Seattle,`",
    ])

    prompt, output = decode_check.parse_decode_output(_message(lines, 0))

    self.assertIn("Describe image", prompt)
    self.assertTrue(output.startswith("Here's a description of the image:"))
    self.assertTrue(output.endswith("cityscape view of Seattle,"))

  def test_parses_vllm_decode_with_chat_template(self):
    # qwen3-4b SFT on Nov 19, 2026. The logging agent stores the line with
    # the absl header as jsonPayload, which starts a new record.
    lines = _lines(
        [
            "\rProcessed prompts: 100%|##########| 1/1 [00:37<00:00]",
            "Prompt: <|im_start|>user",
            f"{_LONDON_PROMPT}<|im_end|>",
            "<|im_start|>assistant",
            ", Generated text: <think>",
            "Okay, the user is asking for some famous landmarks in London.",
            "</think>",
            "",
            "1. **Tower of London** - A historic fortress.",
        ],
        starts_record={1},
    )

    prompt, output = decode_check.parse_decode_output(_message(lines, 4))

    self.assertIn(_LONDON_PROMPT, prompt)
    self.assertNotIn("Processed prompts", prompt)
    self.assertTrue(output.startswith("<think>\nOkay"))
    self.assertTrue(output.endswith("A historic fortress."))

  def test_parses_one_line_vllm_decode(self):
    self.assertEqual(
        decode_check.parse_decode_output(
            f"Prompt: {_LONDON_PROMPT}, Generated text: Big Ben."
        ),
        (_LONDON_PROMPT, "Big Ben."),
    )

  def test_parses_tunix_decode_with_prompt_in_record_before(self):
    # gemma3-4b LoRA on Nov 19, 2026.
    lines = _lines(
        [
            f"Prompt: {_LONDON_PROMPT}",
            "Output: Okay, let's dive into some famous landmarks in London!",
            "* **Buckingham Palace**",
        ],
        starts_record={0, 1},
    )

    self.assertEqual(decode_check.message_bounds(lines, 1), (1, 2))
    self.assertEqual(
        decode_check.parse_decode_output(_message(lines, 1)),
        (
            "",
            "Okay, let's dive into some famous landmarks in London!\n"
            "* **Buckingham Palace**",
        ),
    )
    self.assertEqual(decode_check.prompt_before(lines, 1), _LONDON_PROMPT)

  def test_no_prompt_if_record_before_is_something_else(self):
    lines = _lines(
        ["Generating output...", "Output: Big Ben"], starts_record={0, 1}
    )

    self.assertEqual(decode_check.prompt_before(lines, 1), "")

  def test_ignores_other_lines(self):
    self.assertIsNone(decode_check.parse_decode_output("completed step: 1"))

  def test_message_ends_at_a_pause(self):
    lines = _lines(["Input `I love to` -> ` cook`"]) + _lines(
        ["Input `I love to` -> ` read`"],
        start=_T0 + datetime.timedelta(seconds=3),
    )

    self.assertEqual(decode_check.message_bounds(lines, 0), (0, 0))
    self.assertEqual(decode_check.message_bounds(lines, 1), (1, 1))

  def test_message_ends_at_the_next_log_record(self):
    lines = _lines(
        [
            "Prompt: hi, Generated text: Big Ben is a clock tower.",
            "(MainProcess pid=147) WARNING 11-19 03:49:47"
            " [dp_scheduler.py:1684] Rank 0: Worker process already terminated",
            "Generating output...",
        ],
        starts_record={0, 2},
    )

    self.assertEqual(decode_check.message_bounds(lines, 0), (0, 0))
    self.assertEqual(decode_check.message_bounds(lines, 2), (2, 2))


class FindProblemsTest(unittest.TestCase):

  # Outputs from the daily E2E runs on Nov 19 and Nov 20, 2026.
  _GOOD_OUTPUTS = (
      ("I love to", " read, but I have to be careful with"),
      ("I love to", " make my own bread. I love the smell"),
      ("I love to", " read. I love to read a lot."),
      ("I love to", " see the world in a different light. I"),
      (
          _LONDON_PROMPT,
          "<think>\nOkay, the user is asking for some famous landmarks in"
          " London. I need to come up with a list of the most well-known"
          " ones. Let me start by recalling the major ones. The Eiffel Tower"
          " is a big one, but wait, that's in Paris. Oh right, I need to make"
          " sure they're in London. So maybe the Tower of London.",
      ),
      (
          _LONDON_PROMPT,
          'analysisThe user asks: "Suggest some famous landmarks in London."'
          " They want a list of famous landmarks.\n| Landmark | Why it's"
          " famous |\n|----------|-----------------|\n| **Big Ben & the"
          " Houses of Parliament** | The world-renowned clock tower. |",
      ),
      (
          _IMAGE_PROMPT,
          "Here's a description of the image:\n\n**Overall Impression:**\n\n"
          "The image is a bright, expansive cityscape view of Seattle,",
      ),
  )

  def test_good_outputs_pass(self):
    for prompt, output in self._GOOD_OUTPUTS:
      with self.subTest(output=output):
        self.assertEqual(decode_check.find_problems(prompt, output), [])

  def test_finds_repeated_word(self):
    # gemma3-4b pre-training on Nov 19, 2026.
    self.assertEqual(
        decode_check.find_problems("I love to", " to to to to to to to to to"),
        ['The output repeats "to".'],
    )

  def test_finds_repeated_word_after_other_words(self):
    # gemma4-26b pre-training on Nov 19, 2026.
    self.assertEqual(
        decode_check.find_problems(
            "I love to", " be true true true true true true true true"
        ),
        ['The output repeats "true".'],
    )

  def test_finds_repeated_phrase(self):
    self.assertEqual(
        decode_check.find_problems(
            "I love to", " read a book, read a book, read a book"
        ),
        ['The output repeats "read a book".'],
    )

  def test_finds_long_repeated_list(self):
    # llama3.1-70b SFT on Nov 19, 2026 printed this list 53 times.
    places = (
        "Jerusalem, Bethlehem, Nazareth, Galilee, Masada, Dead Sea, Eilat,"
        " Tel Aviv, Haifa, Akko, Caesarea, "
    )

    self.assertEqual(
        decode_check.find_problems(
            _LONDON_PROMPT, "Big Ben, Tower Bridge, " + places * 5
        ),
        [
            'The output repeats "jerusalem bethlehem nazareth galilee masada'
            ' dead sea eilat tel aviv ...".'
        ],
    )

  def test_finds_few_distinct_words(self):
    # A de Bruijn sequence repeats no phrase in a row.
    words = {"0": "the", "1": "city", "2": "of"}
    output = " ".join(words[c] for c in "000100201101202102211121222")

    self.assertEqual(
        decode_check.find_problems("I love to", output),
        ["The output has only 3 distinct words out of 27."],
    )

  def test_finds_empty_output(self):
    self.assertEqual(
        decode_check.find_problems("I love to", " \n "),
        ["The output is empty."],
    )

  def test_finds_replacement_characters(self):
    self.assertEqual(
        decode_check.find_problems(_LONDON_PROMPT, "Big Ben \ufffd\ufffd"),
        ["The output has U+FFFD replacement characters."],
    )

  def test_finds_output_off_topic(self):
    self.assertEqual(
        decode_check.find_problems(_LONDON_PROMPT, "I like turtles a lot."),
        ["The output doesn't mention London or a London landmark."],
    )
    self.assertEqual(
        decode_check.find_problems(_IMAGE_PROMPT, "I love to cook pasta."),
        ["The output doesn't mention what the image shows."],
    )

  def test_open_prompt_has_no_topic(self):
    self.assertEqual(decode_check.find_problems("I love to", " cook"), [])


def _entry(text, seconds, pod, stream="stdout", starts_record=False):
  """Returns a fake `google.cloud.logging` entry."""
  return types.SimpleNamespace(
      payload={"message": text} if starts_record else text,
      timestamp=_T0 + datetime.timedelta(seconds=seconds),
      resource=types.SimpleNamespace(
          labels={"pod_name": pod, "container_name": "workload-container"}
      ),
      log_name=f"projects/p/logs/{stream}",
  )


def _is_marker(entry):
  if isinstance(entry.payload, dict):
    text = entry.payload["message"]
    return "Generated text:" in text or text.startswith("Output: ")
  return "Input `" in entry.payload or "Generated text:" in entry.payload


class ReadDecodeOutputsTest(unittest.TestCase):

  def _read(self, log):
    filters = []

    def query(project_id, filter_str, start_time, end_time, **_):
      self.assertEqual(project_id, "p")
      filters.append(filter_str)
      pod = re.search(r'pod_name="([^"]+)"', filter_str)

      def matches(entry):
        if not start_time.time <= entry.timestamp.timestamp() <= end_time.time:
          return False
        if pod:
          return entry.resource.labels["pod_name"] == pod[1]
        return _is_marker(entry)

      return [entry for entry in log if matches(entry)]

    outputs = decode_check.read_decode_outputs(
        project_id="p",
        cluster_name="c",
        namespace="ns",
        workload_id="w",
        start_time=_T0 - datetime.timedelta(hours=1),
        end_time=_T0 + datetime.timedelta(hours=1),
        query=query,
    )
    for filter_str in filters:
      self.assertIn(
          'labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"="w"', filter_str
      )
    return outputs

  def test_reads_each_decode_of_one_host(self):
    # Both hosts print the same output before and after training.
    pods = ("w-main-job-0-0-aaaaa", "w-main-job-0-1-bbbbb")
    decode = "Input `I love to` -> ` read, but I have to be careful with`"
    log = [_entry(decode, 0, pod) for pod in pods]
    log += [_entry(decode, 600, pod) for pod in pods]

    outputs = self._read(log)

    self.assertEqual(
        [(output.output, output.pod) for output in outputs],
        [(" read, but I have to be careful with", pods[0])] * 2,
    )

  def test_reads_vllm_decode_over_many_entries(self):
    pod = "w-pathways-head-0-0-ccccc"
    log = [
        _entry("Prompt: <|im_start|>user", 0, pod, "stderr", True),
        _entry(f"{_LONDON_PROMPT}<|im_end|>", 0.00001, pod, "stderr"),
        _entry(", Generated text: Big Ben", 0.00002, pod, "stderr"),
        _entry("and the London Eye.", 0.00003, pod, "stderr"),
        _entry("Generating output...", 5, pod, "stderr", True),
    ]

    (output,) = self._read(log)

    self.assertIn(_LONDON_PROMPT, output.prompt)
    self.assertEqual(output.output, "Big Ben\nand the London Eye.")

  def test_reads_tunix_decode_and_its_prompt(self):
    pod = "w-pathways-head-0-0-ddddd"
    log = [
        _entry(f"Prompt: {_LONDON_PROMPT}", 0, pod, "stderr", True),
        _entry("Output: Okay, let's dive in!", 0.00005, pod, "stderr", True),
        _entry("* **Buckingham Palace**", 0.00006, pod, "stderr"),
    ]

    (output,) = self._read(log)

    self.assertEqual(output.prompt, _LONDON_PROMPT)
    self.assertEqual(
        output.output, "Okay, let's dive in!\n* **Buckingham Palace**"
    )

  def test_reads_nothing_without_decode(self):
    self.assertEqual(self._read([]), [])


@mock.patch.object(decode_check.time, "sleep")
@mock.patch.object(
    decode_check, "get_current_context", return_value={"logical_date": _T0}
)
@mock.patch.object(decode_check, "read_decode_outputs")
class CheckDecodeOutputTest(unittest.TestCase):

  def _check(self, blocking=False):
    decode_check.check_decode_output.function(
        workload_id="sft-v6e-4-12345",
        project_id="p",
        cluster_name="c",
        namespace="ns",
        test_name="qwen3-4b sft",
        blocking=blocking,
    )

  def _decode(self, output):
    return decode_check.DecodeOutput(
        prompt=_LONDON_PROMPT, output=output, pod="pod", timestamp=_T0
    )

  def test_passes_good_output(self, read, *_):
    read.return_value = [self._decode("Big Ben"), self._decode("Big Ben")]

    self._check()

    read.assert_called_once()
    self.assertEqual(
        read.call_args.kwargs["start_time"],
        _T0 - datetime.timedelta(minutes=10),
    )

  def test_skips_on_bad_output(self, read, *_):
    read.return_value = [self._decode("Big Ben"), self._decode("ok ok ok ok")]

    with self.assertRaisesRegex(AirflowSkipException, "qwen3-4b sft") as e:
      self._check()

    self.assertIn('The output repeats "ok".', str(e.exception))
    self.assertIn("Decode 2 of 2 (PROBLEM)", str(e.exception))
    self.assertIn(
        "Logs: https://console.cloud.google.com/logs/viewer?project=p&",
        str(e.exception),
    )
    self.assertIn("sft-v6e-4-12345", str(e.exception).split("Logs: ")[1])

  def test_fails_on_bad_output_when_blocking(self, read, *_):
    read.return_value = [self._decode("Big Ben"), self._decode("")]

    with self.assertRaisesRegex(AirflowFailException, "output is empty"):
      self._check(blocking=True)

  def test_skips_when_a_decode_is_missing(self, read, _, sleep):
    read.return_value = [self._decode("Big Ben")]

    with self.assertRaisesRegex(
        AirflowSkipException, "Missing decode outputs: found 1, expected 2"
    ):
      self._check()

    self.assertEqual(read.call_count, 3)
    self.assertEqual(sleep.call_count, 2)

  def test_reads_again_until_every_decode_is_in(self, read, _, sleep):
    read.side_effect = [
        [self._decode("Big Ben")],
        [self._decode("Big Ben"), self._decode("Big Ben")],
    ]

    self._check()

    self.assertEqual(read.call_count, 2)
    self.assertEqual(sleep.call_count, 1)

  def test_accepts_one_decode_when_one_is_expected(self, read, *_):
    read.return_value = [self._decode("Big Ben")]

    decode_check.check_decode_output.function(
        workload_id="w",
        project_id="p",
        cluster_name="c",
        namespace="ns",
        test_name="t",
        expected_decodes=1,
    )

    read.assert_called_once()

  def test_reads_again_before_skipping_without_decode(self, read, _, sleep):
    read.return_value = []

    with self.assertRaisesRegex(AirflowSkipException, "found no decode") as e:
      self._check()

    self.assertEqual(read.call_count, 3)
    self.assertEqual(sleep.call_count, 2)
    self.assertIn("Logs: https://", str(e.exception))

  def test_skips_when_logs_cannot_be_read(self, read, _, sleep):
    read.side_effect = RuntimeError("403 Permission denied")

    with self.assertRaisesRegex(AirflowSkipException, "Permission denied"):
      self._check()

    self.assertEqual(read.call_count, 3)
    self.assertEqual(sleep.call_count, 2)

  def test_raises_when_logs_cannot_be_read_and_blocking(self, read, *_):
    read.side_effect = RuntimeError("403 Permission denied")

    with self.assertRaisesRegex(RuntimeError, "Permission denied"):
      self._check(blocking=True)

    self.assertEqual(read.call_count, 3)

  def test_reads_again_after_an_error(self, read, _, sleep):
    read.side_effect = [
        RuntimeError("503 Service unavailable"),
        [self._decode("Big Ben"), self._decode("Big Ben")],
    ]

    self._check(blocking=True)

    self.assertEqual(read.call_count, 2)
    self.assertEqual(sleep.call_count, 1)


class DagTest(unittest.TestCase):

  def test_workload_id_of_returns_the_generated_id(self):
    with models.DAG(
        dag_id="decode_check_test",
        schedule=None,
        start_date=datetime.datetime(2026, 1, 1),
    ):
      group = gke_config.get_gke_config(
          time_out_in_min=60,
          test_name="pre",
          run_model_cmds=("true",),
          docker_image="image",
          cluster=GkeClusters.TPU_V6E_BODABORG_NAP_CLUSTER,
          test_owner=test_owner.SURBHI_J,
          use_gcluster=True,
      ).run(skip_post_process=True)

      workload_id = decode_check.workload_id_of(group)

    self.assertEqual(
        workload_id.operator.task_id,
        f"{group.group_id}.pre_process.generate_workload_id",
    )

  def test_checks_every_training_workload(self):
    for dag in (
        maxtext_e2e_tpu_pre_training.dag,
        maxtext_e2e_tpu_post_training.dag,
    ):
      checks = [
          t for t in dag.tasks if t.task_id.endswith(".check_decode_output")
      ]
      goodput_checks = [
          t for t in dag.tasks if t.task_id.endswith(".check_workload_goodput")
      ]
      with self.subTest(dag=dag.dag_id):
        self.assertTrue(checks)
        self.assertEqual(len(checks), len(goodput_checks))
      for check in checks:
        with self.subTest(task=check.task_id):
          group_id = check.task_id.rsplit(".", 1)[0]
          goodput_check = dag.get_task(f"{group_id}.check_workload_goodput")
          # Runs after the training workload, like the Goodput check ...
          self.assertLessEqual(
              goodput_check.upstream_task_ids, check.upstream_task_ids
          )
          # ... and reads the ID of that workload.
          (workload_id_task,) = [
              t
              for t in check.upstream_task_ids
              if t.endswith(".pre_process.generate_workload_id")
          ]
          self.assertTrue(workload_id_task.startswith(f"{group_id}."))
          self.assertNotIn("to-hf", workload_id_task)


if __name__ == "__main__":
  unittest.main()
