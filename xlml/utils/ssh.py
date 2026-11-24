# Copyright 2023 Google LLC
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

"""Utility for ssh."""

import dataclasses
import datetime
import uuid

from airflow.decorators import task
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.cloud import secretmanager

from dags.common.quarantined_tests import safe_get_from_variable
from dags.common.vm_resource import Project

_SECRET_PROJECT = Project.CLOUD_ML_AUTO_SOLUTIONS.value
# Secret Manager deletes a one-time key after this time. It must be longer
# than the longest test run that uses the key.
_ONE_TIME_KEY_TTL = datetime.timedelta(days=7)


@dataclasses.dataclass
class SshKeys:
  """SSH key info that is safe to store in XCom.

  Task return values are stored in XCom, so this class must never hold a
  private key. Use `load_private_key` inside the task that opens the SSH
  connection.
  """

  public: str
  user: str
  # Secret Manager secret (projects/*/secrets/*) that holds the private key of
  # a one-time key pair. None means the persistent key, which is read from
  # Airflow Variables when needed.
  private_key_secret: str | None = None


def load_private_key(ssh_keys: SshKeys) -> str:
  """Returns the private key of `ssh_keys` in clear text.

  Call this only inside a task, and never return or log the result.
  """
  if ssh_keys.private_key_secret is None:
    return safe_get_from_variable("os-login-ssh-private-key", None)
  client = secretmanager.SecretManagerServiceClient()
  response = client.access_secret_version(
      request=secretmanager.AccessSecretVersionRequest(
          name=f"{ssh_keys.private_key_secret}/versions/latest"
      )
  )
  return response.payload.data.decode()


def _store_one_time_private_key(private_key: bytes) -> str:
  """Stores a private key in Secret Manager and returns the secret name."""
  client = secretmanager.SecretManagerServiceClient()
  secret = client.create_secret(
      request=secretmanager.CreateSecretRequest(
          parent=f"projects/{_SECRET_PROJECT}",
          secret_id=f"xlml-one-time-ssh-key-{uuid.uuid4().hex}",
          secret=secretmanager.Secret(
              replication=secretmanager.Replication(
                  automatic=secretmanager.Replication.Automatic()
              ),
              ttl=_ONE_TIME_KEY_TTL,
          ),
      )
  )
  client.add_secret_version(
      request=secretmanager.AddSecretVersionRequest(
          parent=secret.name,
          payload=secretmanager.SecretPayload(data=private_key),
      )
  )
  return secret.name


@task
def generate_ssh_keys() -> SshKeys:
  """Generate an RSA key pair in the OpenSSH format."""
  key = rsa.generate_private_key(
      public_exponent=65537,
      key_size=2048,
  )

  private_key = key.private_bytes(
      serialization.Encoding.PEM,
      serialization.PrivateFormat.OpenSSH,
      serialization.NoEncryption(),
  )
  public_key = key.public_key().public_bytes(
      serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH
  )

  return SshKeys(
      public=public_key.decode(),
      user="ml-auto-solutions",
      private_key_secret=_store_one_time_private_key(private_key),
  )


@task
def obtain_persist_ssh_keys() -> SshKeys:
  """Obtain persistent SSH keys and user from Airflow Variables."""
  try:
    user = safe_get_from_variable("os-login-ssh-user", None)
    public_key = safe_get_from_variable("os-login-ssh-public-key", None)
  except KeyError as e:
    raise ValueError(
        f"Required Airflow Variable {e} is not set. "
        "Please ensure 'os-login-ssh-user', 'os-login-ssh-private-key', "
        "and 'os-login-ssh-public-key' are configured."
    ) from e

  return SshKeys(public=public_key, user=user)
