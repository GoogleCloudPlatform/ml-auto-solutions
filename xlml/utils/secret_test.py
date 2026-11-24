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

"""Unit tests for secret.py."""

import unittest
from unittest import mock

from airflow.exceptions import AirflowFailException
from xlml.utils import secret


class ResolveEnvTest(unittest.TestCase):

  def test_empty(self):
    self.assertEqual(secret.resolve_env(None), {})
    self.assertEqual(secret.resolve_env({}), {})

  @mock.patch("xlml.utils.secret.mask_secret")
  @mock.patch("xlml.utils.secret.safe_get_from_variable")
  def test_reads_variable_and_masks_value(self, mock_get, mock_mask):
    mock_get.return_value = "hf_canary"

    resolved = secret.resolve_env({"HF_TOKEN": "HF_TOKEN_MOE"})

    self.assertEqual(resolved, {"HF_TOKEN": "hf_canary"})
    mock_get.assert_called_once_with("HF_TOKEN_MOE", None)
    mock_mask.assert_called_once_with("hf_canary")

  @mock.patch("xlml.utils.secret.mask_secret")
  @mock.patch("xlml.utils.secret.safe_get_from_variable")
  def test_removes_quotes_stored_around_the_value(self, mock_get, _):
    for stored in ("'hf_canary'", '"hf_canary"', " 'hf_canary'\n", "hf_canary"):
      mock_get.return_value = stored
      self.assertEqual(
          secret.resolve_env({"HF_TOKEN": "HF_TOKEN"}),
          {"HF_TOKEN": "hf_canary"},
      )

  @mock.patch("xlml.utils.secret.mask_secret")
  @mock.patch("xlml.utils.secret.safe_get_from_variable")
  def test_keeps_quotes_that_do_not_match(self, mock_get, _):
    mock_get.return_value = "'hf_canary"
    self.assertEqual(
        secret.resolve_env({"HF_TOKEN": "HF_TOKEN"}), {"HF_TOKEN": "'hf_canary"}
    )

  @mock.patch("xlml.utils.secret.safe_get_from_variable", return_value="''")
  def test_empty_quoted_variable_fails_the_task(self, _):
    with self.assertRaises(AirflowFailException):
      secret.resolve_env({"HF_TOKEN": "HF_TOKEN"})

  @mock.patch("xlml.utils.secret.safe_get_from_variable", return_value=None)
  def test_missing_variable_fails_the_task(self, _):
    with self.assertRaises(AirflowFailException):
      secret.resolve_env({"HF_TOKEN": "HF_TOKEN"})


if __name__ == "__main__":
  unittest.main()
