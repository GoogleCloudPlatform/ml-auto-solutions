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

"""Tests for dags.multipod.util.pathways_images."""

import datetime
import json
import types
import unittest
from unittest import mock

from airflow import models
from airflow.exceptions import AirflowSkipException
from airflow.utils.trigger_rule import TriggerRule

from dags.common import test_owner
from dags.common.vm_resource import GkeClusters
from dags.multipod import maxtext_e2e_tests
from dags.multipod import maxtext_e2e_tpu_post_training
from dags.multipod.configs import gke_config
from dags.multipod.util import pathways_images
from xlml.utils import gke

_REGISTRY = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways"
_SERVER = "sha256:" + "3462904394" * 6 + "38ab"
_SERVER_2 = "sha256:" + "5be0f1d2c3" * 6 + "a4b5"
_PROXY = "sha256:" + "50008420aa" * 6 + "80a7"


def _status(name, image, digest):
  return types.SimpleNamespace(
      name=name,
      image=f"{_REGISTRY}/{image}:latest",
      image_id=f"{_REGISTRY}/{image}@{digest}" if digest else "",
  )


def _head_pod(server=_SERVER, proxy=_PROXY):
  """A head pod as gcluster creates it: the Pathways containers are sidecars."""
  return types.SimpleNamespace(
      status=types.SimpleNamespace(
          init_container_statuses=[
              _status("pathways-proxy", "proxy_server", proxy),
              _status("pathways-rm", "server", server),
          ],
          container_statuses=[
              types.SimpleNamespace(
                  name="workload-container",
                  image="maxtext:latest",
                  image_id="maxtext@sha256:" + "0" * 64,
              )
          ],
      )
  )


def _worker_pod(server=_SERVER):
  return types.SimpleNamespace(
      status=types.SimpleNamespace(
          init_container_statuses=None,
          container_statuses=[_status("pathways-worker", "server", server)],
      )
  )


def _record(*server_digests):
  return {
      "workload_id": "sft-v6e-4-12345",
      "images": {
          "pathways-rm": {
              "image": f"{_REGISTRY}/server:latest",
              "digests": list(server_digests[:1]),
          },
          "pathways-worker": {
              "image": f"{_REGISTRY}/server:latest",
              "digests": list(server_digests),
          },
      },
  }


class ImagesOfPodsTest(unittest.TestCase):

  def test_reads_the_head_sidecars_and_the_workers(self):
    images = pathways_images.images_of_pods([_head_pod(), _worker_pod()])

    self.assertEqual(
        images,
        {
            "pathways-proxy": {
                "image": f"{_REGISTRY}/proxy_server:latest",
                "digests": [_PROXY],
            },
            "pathways-rm": {
                "image": f"{_REGISTRY}/server:latest",
                "digests": [_SERVER],
            },
            "pathways-worker": {
                "image": f"{_REGISTRY}/server:latest",
                "digests": [_SERVER],
            },
        },
    )

  def test_keeps_every_digest_of_a_container(self):
    images = pathways_images.images_of_pods(
        [_worker_pod(_SERVER_2), _worker_pod(_SERVER), _worker_pod(_SERVER)]
    )

    self.assertEqual(
        images["pathways-worker"]["digests"], sorted([_SERVER, _SERVER_2])
    )

  def test_skips_containers_that_have_not_started(self):
    pending = types.SimpleNamespace(status=None)

    images = pathways_images.images_of_pods(
        [pending, _head_pod(server=""), _worker_pod(server="")]
    )

    self.assertEqual(list(images), ["pathways-proxy"])


class ServerDigestsTest(unittest.TestCase):

  def test_collects_the_resource_manager_and_worker_digests(self):
    records = {"a sft": _record(_SERVER), "b rl": _record(_SERVER, _SERVER_2)}

    self.assertEqual(
        pathways_images.server_digests(records), sorted([_SERVER, _SERVER_2])
    )


@mock.patch.object(pathways_images.time, "sleep")
@mock.patch.object(pathways_images.gke, "list_workload_pods")
@mock.patch.object(pathways_images.gke, "get_core_api_client")
class RecordPathwaysImagesTest(unittest.TestCase):

  def _record(self):
    return pathways_images.record_pathways_images.function(
        workload_id="sft-v6e-4-12345",
        project_id="p",
        region="r",
        cluster_name="c",
        namespace="ns",
    )

  def test_records_every_pathways_container(self, _, list_pods, sleep):
    list_pods.return_value = types.SimpleNamespace(
        items=[_head_pod(), _worker_pod()]
    )

    record = self._record()

    self.assertEqual(record["workload_id"], "sft-v6e-4-12345")
    self.assertEqual(
        sorted(record["images"]), sorted(pathways_images.PATHWAYS_CONTAINERS)
    )
    # Lists the pods of the whole JobSet, so the worker pods too.
    self.assertEqual(list_pods.call_args.kwargs["use_pathways"], False)
    self.assertEqual(
        list_pods.call_args.kwargs["workload_kind"], gke.WorkloadKind.JOBSET
    )
    sleep.assert_not_called()

  def test_reads_again_until_the_workers_start(self, _, list_pods, sleep):
    list_pods.side_effect = [
        types.SimpleNamespace(items=[_head_pod()]),
        types.SimpleNamespace(items=[_head_pod(), _worker_pod()]),
    ]

    record = self._record()

    self.assertIn("pathways-worker", record["images"])
    self.assertEqual(sleep.call_count, 1)

  def test_returns_what_it_found_after_the_last_read(self, _, list_pods, sleep):
    list_pods.return_value = types.SimpleNamespace(items=[_head_pod()])

    record = self._record()

    self.assertEqual(
        sorted(record["images"]), ["pathways-proxy", "pathways-rm"]
    )
    # Read on every attempt, and waited between them.
    self.assertGreater(sleep.call_count, 1)
    self.assertEqual(list_pods.call_count, sleep.call_count + 1)

  def test_skips_when_no_container_reports_its_image(
      self, _, list_pods, unused_sleep
  ):
    list_pods.return_value = types.SimpleNamespace(items=[])

    with self.assertRaises(AirflowSkipException):
      self._record()

  def test_skips_when_the_pods_cannot_be_read(
      self, client, list_pods, unused_sleep
  ):
    client.side_effect = RuntimeError("no credentials")

    with self.assertRaises(AirflowSkipException):
      self._record()
    list_pods.assert_not_called()


@mock.patch.object(pathways_images, "GCSHook")
class WritePathwaysImagesTest(unittest.TestCase):

  def test_writes_one_file_for_the_run(self, hook):
    digests = pathways_images.write_pathways_images.function(
        records={"a sft": _record(_SERVER), "b rl": None},
        run_name="e2e-123",
        dag_run_id="run-1__post_training",
    )

    self.assertEqual(digests, _SERVER)
    upload = hook.return_value.upload
    upload.assert_called_once()
    self.assertEqual(
        upload.call_args.kwargs["bucket_name"], "runner-maxtext-logs"
    )
    self.assertEqual(
        upload.call_args.kwargs["object_name"],
        "e2e_runs/e2e-123/pathways_images.json",
    )
    self.assertEqual(
        json.loads(upload.call_args.kwargs["data"]),
        {
            "run_name": "e2e-123",
            "dag_run_id": "run-1__post_training",
            "server_digests": [_SERVER],
            "workloads": {"a sft": _record(_SERVER)},
        },
    )

  def test_skips_when_no_workload_recorded_its_images(self, hook):
    with self.assertRaises(AirflowSkipException):
      pathways_images.write_pathways_images.function(
          records={"a sft": None},
          run_name="e2e-123",
          dag_run_id="run-1__post_training",
      )
    hook.assert_not_called()

  def test_skips_when_the_file_cannot_be_written(self, hook):
    hook.return_value.upload.side_effect = RuntimeError("denied")

    with self.assertRaises(AirflowSkipException):
      pathways_images.write_pathways_images.function(
          records={"a sft": _record(_SERVER)},
          run_name="e2e-123",
          dag_run_id="run-1__post_training",
      )


@mock.patch.object(pathways_images, "GCSHook")
class ReadPathwaysServerDigestsTest(unittest.TestCase):

  def _read(self, post_training_run_id="run-1__post_training"):
    return pathways_images.read_pathways_server_digests.function(
        run_name="e2e-123", post_training_run_id=post_training_run_id
    )

  def test_returns_every_server_digest_of_the_run(self, hook):
    # Two digests: `server:latest` moved while the workloads started.
    hook.return_value.download.return_value = json.dumps({
        "dag_run_id": "run-1__post_training",
        "server_digests": [_SERVER, _SERVER_2],
    }).encode()

    self.assertEqual(self._read(), f"{_SERVER},{_SERVER_2}")
    hook.return_value.download.assert_called_once_with(
        bucket_name="runner-maxtext-logs",
        object_name="e2e_runs/e2e-123/pathways_images.json",
    )

  def test_returns_nothing_for_a_file_from_another_run(self, hook):
    hook.return_value.download.return_value = json.dumps({
        "dag_run_id": "run-0__post_training",
        "server_digests": [_SERVER],
    }).encode()

    self.assertEqual(self._read(), "")

  def test_returns_nothing_without_a_file(self, hook):
    hook.return_value.download.side_effect = RuntimeError("404")

    self.assertEqual(self._read(), "")


class ReRunTest(unittest.TestCase):

  def test_rerun_does_not_send_the_digests_of_an_earlier_attempt(self):
    """A GitHub re-run keeps the run name, so both attempts use one file."""
    files = {}

    class FakeGCSHook:

      def upload(self, bucket_name, object_name, data, **_):
        files[(bucket_name, object_name)] = data

      def download(self, bucket_name, object_name):
        return files[(bucket_name, object_name)].encode()

    with mock.patch.object(pathways_images, "GCSHook", FakeGCSHook):
      pathways_images.write_pathways_images.function(
          records={"a sft": _record(_SERVER)},
          run_name="e2e-123",
          dag_run_id="attempt-1__post_training",
      )
      # No workload of the second attempt records its images.
      with self.assertRaises(AirflowSkipException):
        pathways_images.write_pathways_images.function(
            records={"a sft": None},
            run_name="e2e-123",
            dag_run_id="attempt-2__post_training",
        )

      digests = pathways_images.read_pathways_server_digests.function(
          run_name="e2e-123", post_training_run_id="attempt-2__post_training"
      )

    self.assertEqual(digests, "")


class DagTest(unittest.TestCase):

  def test_workload_start_of_returns_the_start_sensor(self):
    with models.DAG(
        dag_id="pathways_images_test",
        schedule=None,
        start_date=datetime.datetime(2026, 1, 1),
    ):
      group = gke_config.get_gke_config(
          time_out_in_min=60,
          test_name="sft",
          run_model_cmds=("true",),
          docker_image="image",
          cluster=GkeClusters.TPU_V6E_BODABORG_NAP_CLUSTER,
          test_owner=test_owner.SURBHI_J,
          use_pathways=True,
          use_gcluster=True,
      ).run(skip_post_process=True)

      start = pathways_images.workload_start_of(group)

    self.assertEqual(
        start.task_id,
        f"{group.group_id}.run_model.launch_workload.wait_for_workload_start",
    )

  def test_records_every_post_training_workload(self):
    dag = maxtext_e2e_tpu_post_training.dag
    records = [
        t for t in dag.tasks if t.task_id.endswith(".record_pathways_images")
    ]
    checks = [
        t for t in dag.tasks if t.task_id.endswith(".check_decode_output")
    ]
    self.assertTrue(records)
    self.assertEqual(len(records), len(checks))
    for record in records:
      with self.subTest(task=record.task_id):
        group_id = record.task_id.rsplit(".", 1)[0]
        # Runs as soon as the workload of its own group starts.
        (start,) = [
            t
            for t in record.upstream_task_ids
            if t.endswith(".wait_for_workload_start")
        ]
        (workload_id,) = [
            t
            for t in record.upstream_task_ids
            if t.endswith(".generate_workload_id")
        ]
        self.assertTrue(start.startswith(f"{group_id}."))
        self.assertTrue(workload_id.startswith(f"{group_id}."))
        self.assertNotIn("to-hf", workload_id)

    write = dag.get_task("write_pathways_images")
    self.assertEqual(write.trigger_rule, TriggerRule.ALL_DONE)
    self.assertEqual(write.upstream_task_ids, {t.task_id for t in records})
    self.assertEqual(write.op_kwargs["dag_run_id"], "{{ run_id }}")

  def test_sends_the_server_digests_with_the_post_training_result(self):
    dag = maxtext_e2e_tests.dag
    read = dag.get_task("read_pathways_server_digests")
    callback = dag.get_task("fire_github_callback_post_training")

    self.assertEqual(read.trigger_rule, TriggerRule.ALL_DONE)
    self.assertEqual(read.upstream_task_ids, {"trigger_tpu_post_training"})
    # Reads only the file of the post-training run that this run triggered.
    self.assertEqual(
        read.op_kwargs["post_training_run_id"],
        dag.get_task("trigger_tpu_post_training").trigger_run_id,
    )
    self.assertEqual(callback.upstream_task_ids, {read.task_id})
    payload = callback.op_kwargs["client_payload"]
    self.assertIn(
        "read_pathways_server_digests", payload["pathways_server_digests"]
    )
    # GitHub accepts at most 10 top-level properties in client_payload.
    self.assertLessEqual(len(payload), 10)


if __name__ == "__main__":
  unittest.main()
