"""Utility for parsing the output of the 'tpu-info' command."""

import re
from dataclasses import dataclass
from enum import IntEnum, auto

from airflow.decorators import task

# A type alias for a parsed row, mapping column headers to their values.
_TableRow = dict[str, str]


@dataclass
class Table:
  """Represents a single parsed table from the tpu-info output."""

  name: str
  raw_body: str
  body: list[_TableRow]

  def parse_body(self):
    """Parses the raw_body string to populate the structured body attribute."""

    class TableLineIndex(IntEnum):
      """Below is an example of the text returned by tpu-info, formatted as a
      table.

      TPU Chips
      ┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
      ┃ Chip        ┃ Type         ┃ Devices ┃ PID  ┃
      ┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
      │ /dev/vfio/0 │ TPU v6e chip │ 1       │ 1016 │
      │ /dev/vfio/1 │ TPU v6e chip │ 1       │ 1016 │
      │ /dev/vfio/2 │ TPU v6e chip │ 1       │ 1016 │
      │ /dev/vfio/3 │ TPU v6e chip │ 1       │ 1016 │
      └─────────────┴──────────────┴─────────┴──────┘
      """

      UPPER_BORDER = 0
      HEADER = auto()
      SEPARATOR = auto()
      DATA = auto()
      LOWER_BORDER = -1

    lines = self.raw_body.strip().split("\n")
    if len(lines) < max(TableLineIndex):
      self.body = []
      return

    header_line = lines[TableLineIndex.HEADER]
    headers = [h.strip() for h in header_line.split("┃") if h.strip()]
    data_lines = lines[TableLineIndex.DATA : TableLineIndex.LOWER_BORDER]

    parsed_body = []
    for line in data_lines:
      columns = line.split("│")[1:-1]
      if len(columns) != len(headers):
        continue

      row_data: _TableRow = {
          header: col.strip() for header, col in zip(headers, columns)
      }
      parsed_body.append(row_data)

    self.body = parsed_body


@task
def parse_tpu_info_output(output: str) -> list[Table]:
  """Splits a multi-table string from tpu-info into a structured TpuInfo object.

  Args:
    output: The raw string output from the 'tpu-info' command.

  Returns:
    A TpuInfo object with attributes populated for each found table.
  """
  title_pattern = re.compile(r"(^[^\n].*)\n┏", re.MULTILINE)
  table_block_pattern = re.compile(r"(^┏[\s\S]*?┘)", re.MULTILINE)

  titles = [s.strip() for s in title_pattern.findall(output)]
  blocks = table_block_pattern.findall(output)

  if len(titles) != len(blocks):
    raise ValueError(
        "Mismatch between found table titles and table blocks. "
        f"Found {len(titles)} titles and {len(blocks)} blocks."
    )

  parsed_tables = []
  for name, raw_body in zip(titles, blocks):
    table = Table(name=name, raw_body=raw_body, body=None)
    table.parse_body()
    parsed_tables.append(table)

  return parsed_tables


if __name__ == "__main__":
  full_output = """
                                                                                                                                                   Libtpu version: 0.0.23
                                                                                                                                                   Accelerator type: v6e

TPU Chips
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━┓
┃ Chip        ┃ Type         ┃ Devices ┃ PID  ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━┩
│ /dev/vfio/0 │ TPU v6e chip │ 1       │ 1098 │
│ /dev/vfio/1 │ TPU v6e chip │ 1       │ 1098 │
│ /dev/vfio/2 │ TPU v6e chip │ 1       │ 1098 │
│ /dev/vfio/3 │ TPU v6e chip │ 1       │ 1098 │
└─────────────┴──────────────┴─────────┴──────┘
TPU Runtime Utilization
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Device ┃ HBM Usage (GiB)       ┃ Duty cycle ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ 8      │ 18.45 GiB / 31.25 GiB │ 100.00%    │
│ 9      │ 10.40 GiB / 31.25 GiB │ 100.00%    │
│ 12     │ 10.40 GiB / 31.25 GiB │ 100.00%    │
│ 13     │ 10.40 GiB / 31.25 GiB │ 100.00%    │
└────────┴───────────────────────┴────────────┘
TensorCore Utilization
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Chip ID ┃ TensorCore Utilization ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 0       │ 15.14%                 │
│ 1       │ 14.56%                 │
│ 2       │ 15.53%                 │
│ 3       │ 14.97%                 │
└─────────┴────────────────────────┘
TPU Buffer Transfer Latency
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Buffer Size ┃ P50         ┃ P90         ┃ P95         ┃ P999         ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 8MB+        │ 30154.32 us │ 65472.43 us │ 73990.97 us │ 103220.65 us │
│ 4MB+        │ 16622.62 us │ 33210.22 us │ 36404.47 us │ 50954.72 us  │
└─────────────┴─────────────┴─────────────┴─────────────┴──────────────┘
"""

  tpu_info_output = parse_tpu_info_output(full_output)
  print(tpu_info_output)
