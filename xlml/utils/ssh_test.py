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

"""Tests for ssh.py."""

import io
import json
import unittest
from unittest import mock

from airflow.serialization import serde
from google.cloud import secretmanager
import paramiko

from xlml.utils import ssh


class FakeSecretManagerClient:
  """Keeps secrets in memory. All instances share one store."""

  secrets = {}
  payloads = {}

  def create_secret(self, request):
    name = f"{request.parent}/secrets/{request.secret_id}"
    self.secrets[name] = request.secret
    return secretmanager.Secret(name=name)

  def add_secret_version(self, request):
    self.payloads[request.parent] = request.payload.data

  def access_secret_version(self, request):
    secret_name = request.name.removesuffix("/versions/latest")
    return secretmanager.AccessSecretVersionResponse(
        payload=secretmanager.SecretPayload(data=self.payloads[secret_name])
    )


class SshTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    FakeSecretManagerClient.secrets = {}
    FakeSecretManagerClient.payloads = {}
    patcher = mock.patch.object(
        ssh.secretmanager,
        "SecretManagerServiceClient",
        FakeSecretManagerClient,
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def test_generated_keys_have_no_private_key_in_xcom(self):
    keys = ssh.generate_ssh_keys.function()

    xcom_value = json.dumps(serde.serialize(keys))

    self.assertNotIn("PRIVATE KEY", xcom_value)
    self.assertIn(keys.public, xcom_value)

  def test_generated_private_key_is_stored_in_an_expiring_secret(self):
    keys = ssh.generate_ssh_keys.function()

    self.assertRegex(
        keys.private_key_secret,
        r"^projects/cloud-ml-auto-solutions/secrets/xlml-one-time-ssh-key-",
    )
    secret = FakeSecretManagerClient.secrets[keys.private_key_secret]
    self.assertGreater(secret.ttl.total_seconds(), 0)

  def test_load_private_key_matches_generated_public_key(self):
    keys = ssh.generate_ssh_keys.function()

    private_key = ssh.load_private_key(keys)

    pkey = paramiko.RSAKey.from_private_key(io.StringIO(private_key))
    self.assertEqual(
        f"{pkey.get_name()} {pkey.get_base64()}", keys.public.strip()
    )

  @mock.patch("xlml.utils.ssh.safe_get_from_variable")
  def test_persistent_keys_have_no_private_key_in_xcom(self, mock_get):
    variables = {
        "os-login-ssh-user": "user",
        "os-login-ssh-private-key": "PRIVATE KEY",
        "os-login-ssh-public-key": "ssh-rsa AAAA",
    }
    mock_get.side_effect = lambda key, default: variables[key]

    keys = ssh.obtain_persist_ssh_keys.function()

    self.assertNotIn("PRIVATE KEY", json.dumps(serde.serialize(keys)))
    self.assertEqual(ssh.load_private_key(keys), "PRIVATE KEY")


if __name__ == "__main__":
  unittest.main()
