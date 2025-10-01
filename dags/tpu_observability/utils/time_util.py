"""Utility class for handling various time representations."""

import datetime
from dataclasses import dataclass
from typing import Union

from google.protobuf import timestamp_pb2


@dataclass
class TimeUtil:
  """A utility that represents time in multiple forms and converting between them."""

  time: int

  @classmethod
  def from_iso_string(cls, time_str: str) -> "TimeUtil":
    """Builds a TimeUtil object from an ISO 8601 formatted string."""
    dt_object = datetime.datetime.fromisoformat(time_str.replace("Z", "+00:00"))
    return cls(int(dt_object.timestamp()))

  @classmethod
  def from_timestamp_pb2(cls, ts_pb: timestamp_pb2.Timestamp) -> "TimeUtil":
    """Builds a TimeUtil object from a Google Protobuf Timestamp."""
    return cls(int(ts_pb.seconds))

  @classmethod
  def from_datetime(cls, dt: datetime.datetime) -> "TimeUtil":
    """Builds a TimeUtil object from a standard datetime object."""
    return cls(int(dt.timestamp()))

  @classmethod
  def from_unix_seconds(cls, unix_seconds: Union[int, float]) -> "TimeUtil":
    """Builds a TimeUtil object from a Unix timestamp (seconds)."""
    return cls(int(unix_seconds))

  def to_unix_seconds(self) -> int:
    return self.time

  def to_timestamp_pb2(self) -> timestamp_pb2.Timestamp:
    timestamp = timestamp_pb2.Timestamp()
    timestamp.FromSeconds(self.time)
    return timestamp

  def to_datetime(self) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(self.time, tz=datetime.timezone.utc)

  def to_iso_string(self) -> str:
    iso_str = self.to_datetime().isoformat()
    return iso_str.replace("+00:00", "Z")


if __name__ == "__main__":
  time = "2025-09-19T04:08:35.951+00:00"
  time_obj = TimeUtil.from_iso_string(time)
  print(time_obj.to_iso_string())
  print(time_obj.to_timestamp_pb2())

  start_time = datetime.datetime.fromisoformat("2025-09-19T04:08:35.951+00:00")
  end_time = start_time + datetime.timedelta(minutes=10)

  start_time_obj = TimeUtil.from_datetime(start_time)
  end_time_obj = TimeUtil.from_datetime(end_time)

  print(start_time_obj.to_timestamp_pb2())
  print(end_time_obj.to_timestamp_pb2())
