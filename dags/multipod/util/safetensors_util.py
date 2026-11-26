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

"""Utilities and remote TPU pod scripts for MaxText safetensors_dynamic tests.

This module serves two roles without embedding multiline script strings:
1. Airflow DAG command builders (`pod0_barrier_cmd`, `pod0_upload_log_trap_cmd`,
   `pod0_tee_cmd`, `build_prestage_cmds`, `build_stage_summary_cmd`,
   `build_compare_cmds`) imported by `maxtext_e2e_tpu_safetensors_dynamic.py`,
   which stages this standalone Python file onto remote TPU pods
   (`stage_remote_util_cmd`).
2. Remote TPU pod CLI (`/tmp/safetensors_util.py`) supporting subcommands:
   - `prestage`: Atomically caches Hugging Face `*.safetensors` shards into a
     shared GCS directory using a generation-matched GCS lock, recording
     cold vs. warm cache status and transfer timing.
   - `summarize_stages`: Parses per-stage timing logs for `safetensors_dynamic`
     and `to_maxtext`, reads the HF checkpoint dtypes from the shard headers,
     and writes `stage_timings.json` to GCS.
   - `compare`: Compares in-memory `safetensors_dynamic` weights, Step-0 Orbax
     checkpoints, trained Step-(N-1) Orbax checkpoints, and per-step losses
     against the reference `to_maxtext` workflow.
"""

import argparse
import base64
import collections
import concurrent.futures
import datetime
import functools
import json
import os
import pathlib
import random
import re
import shlex
import struct
import sys
import time
import zlib
from typing import Any, Callable

try:
  from google.api_core import exceptions as gcs_exceptions
  from google.cloud import storage
except ImportError:
  gcs_exceptions = None
  storage = None

try:
  import huggingface_hub
except ImportError:
  huggingface_hub = None

try:
  from absl import logging as absl_logging
except ImportError:
  absl_logging = None

try:
  from etils import epath
except ImportError:
  epath = None

try:
  from flax import nnx
  from flax import traverse_util
  from flax.core.spmd import logical_axis_rules as axis_rules
  import jax
  import jax.numpy as jnp
  from maxtext.common import checkpointing
  from maxtext.common import train_state_nnx
  from maxtext.configs import pyconfig
  from maxtext.utils import max_logging
  from maxtext.utils import maxtext_utils
  from maxtext.utils import model_creation_utils
  from maxtext.utils import train_utils
except ImportError:
  nnx = None
  traverse_util = None
  axis_rules = None
  jax = None
  jnp = None
  checkpointing = None
  train_state_nnx = None
  pyconfig = None
  max_logging = None
  maxtext_utils = None
  model_creation_utils = None
  train_utils = None

GCS_CONSOLE_OBJECT_URL = (
    "https://console.cloud.google.com/storage/browser/_details/"
)

POD0_CONDITION = (
    '[[ "${MEGASCALE_SLICE_ID:-${JOB_INDEX:-0}}" == "0" && '
    '"${TPU_WORKER_ID:-${JOB_COMPLETION_INDEX:-0}}" == "0" ]]'
)
REMOTE_UTIL_PATH = "/tmp/safetensors_util.py"
_MAX_SAFETENSORS_HEADER_BYTES = 512 * 1024 * 1024


@functools.lru_cache(maxsize=1)
def _encoded_module_source() -> str:
  """Returns the zlib-compressed, base64-encoded bytes of this module."""
  source_bytes = pathlib.Path(__file__).read_bytes()
  return base64.b64encode(zlib.compress(source_bytes, 9)).decode("ascii")


_UNPACK_UTIL_SNIPPET = (
    "import base64, sys, zlib; "
    "data = zlib.decompress(base64.b64decode(sys.stdin.read())); "
    "open(sys.argv[1], 'wb').write(data)"
)


def stage_remote_util_cmd(remote_path: str = REMOTE_UTIL_PATH) -> str:
  """Returns a shell command that materializes this module on a remote pod.

  The module is embedded in the workload command line, which the kernel caps
  at ~128 KiB per argument, so emit this once per workload and let later steps
  rely on `require_remote_util_cmd`.
  """
  encoded = _encoded_module_source()
  return (
      f"echo '{encoded}' | "
      f'python3 -c "{_UNPACK_UTIL_SNIPPET}" {shlex.quote(remote_path)}'
  )


def require_remote_util_cmd(remote_path: str = REMOTE_UTIL_PATH) -> str:
  """Returns a cheap guard that fails if this module was not staged."""
  quoted = shlex.quote(remote_path)
  return (
      f"{{ test -s {quoted} || "
      f"{{ echo '[safetensors_util] {remote_path} is not staged' >&2; "
      "false; }; }"
  )


def pod0_barrier_cmd(
    pod0_cmds: tuple[str, ...],
    marker_gcs: str,
    poll_attempts: int = 720,
    poll_interval_sec: int = 10,
) -> str:
  """Runs `pod0_cmds` on slice 0 / worker 0 and syncs all pods via GCS."""
  joined_pod0 = " && ".join(pod0_cmds)
  fail_marker_gcs = f"{marker_gcs}.failed"
  return (
      f"if {POD0_CONDITION}; then "
      f"gsutil -q rm -f {marker_gcs} {fail_marker_gcs} 2>/dev/null || true; "
      f"if ({joined_pod0}); then "
      f"echo done | gsutil -q cp - {marker_gcs}; "
      "else "
      "rc=$?; "
      f"echo failed | gsutil -q cp - {fail_marker_gcs} 2>/dev/null || true; "
      "exit $rc; "
      "fi; "
      "else "
      f'echo "Waiting for pod 0 marker {marker_gcs}..."; '
      f"for _ in $(seq 1 {poll_attempts}); do "
      f"gsutil -q stat {marker_gcs} 2>/dev/null && break; "
      f"if gsutil -q stat {fail_marker_gcs} 2>/dev/null; then "
      f'echo "Pod 0 prep failed ({fail_marker_gcs}); aborting." >&2; exit 1; '
      "fi; "
      f"sleep {poll_interval_sec}; "
      "done; "
      f"gsutil -q stat {marker_gcs}; "
      "fi"
  )


def gcs_console_url(gcs_uri: str) -> str:
  """Returns the Cloud Console page of a `gs://` object, or `''`.

  Only the scheme is rewritten, so a Jinja template in the path still renders.
  """
  object_path = gcs_uri.removeprefix("gs://")
  if object_path == gcs_uri or not object_path:
    return ""
  return GCS_CONSOLE_OBJECT_URL + object_path


def pod0_upload_log_trap_cmd(
    local_log: str, gcs_log: str, on_exit_cmd: str = ""
) -> str:
  """Registers an EXIT trap on Pod 0 to upload `local_log` on pass or fail.

  The trap also prints a `[LogLinks]` line with the Cloud Console URL of the
  uploaded log, so a passed or failed job shows a link that opens.
  """
  extra_trap = f"{on_exit_cmd} 2>/dev/null || true; " if on_exit_cmd else ""
  log_url = gcs_console_url(gcs_log)
  print_link = (
      f'; echo "[LogLinks] {gcs_log.rsplit("/", 1)[-1]}={log_url}"'
      if log_url
      else ""
  )
  return (
      f"if {POD0_CONDITION}; then "
      f": > {local_log}; "
      f"trap '{extra_trap}gsutil -q cp {local_log} {gcs_log} 2>/dev/null ||"
      f" true{print_link}' EXIT; "
      "fi"
  )


def pod0_tee_cmd(cmd: str, local_log: str) -> str:
  """Runs `cmd` and tees output to `local_log` on Pod 0."""
  return (
      f"if {POD0_CONDITION}; then "
      f"set -o pipefail; ({cmd}) 2>&1 | tee -a {local_log}; "
      "else "
      f"{cmd}; "
      "fi"
  )


def _quote_path(path: str) -> str:
  """Shell-quotes a local path; a `gs://` URI is left as is.

  A `gs://` URI built by the DAG can hold a Jinja template (the run name),
  which quoting would break before Airflow renders it.
  """
  return path if path.startswith("gs://") else shlex.quote(path)


def build_prestage_cmds(
    hf_repo: str,
    gcs_cache_dir: str,
    timing_out: str | None = None,
    force_download_expr: str = "",
    remote_path: str = REMOTE_UTIL_PATH,
    verify_only: bool = False,
) -> tuple[str, ...]:
  """Builds shell commands to cache HF shards to GCS (util pre-staged).

  With `verify_only` the command downloads nothing: it only checks that the
  shared GCS cache is complete and fails when it is not.
  """
  timing_arg = f" --timing_out={_quote_path(timing_out)}" if timing_out else ""
  force_arg = f" {force_download_expr}" if force_download_expr else ""
  mode_args = " --verify_only" if verify_only else f"{timing_arg}{force_arg}"
  return (
      require_remote_util_cmd(remote_path),
      (
          f"python3 {shlex.quote(remote_path)} prestage "
          f"--hf_repo={shlex.quote(hf_repo)} "
          f"--gcs_cache_dir={shlex.quote(gcs_cache_dir)}"
          f"{mode_args}"
      ),
  )


def build_stage_summary_cmd(
    *,
    dyn_log: str,
    stage_timings_gcs: str,
    weight_dtype: str,
    to_maxtext_save_dtype: str,
    dyn_log_gcs: str,
    prestage_timing: str = "",
    gcs_cache_dir: str = "",
    ref_log: str = "",
    ref_log_gcs: str = "",
    remote_path: str = REMOTE_UTIL_PATH,
) -> str:
  """Builds a Pod-0 command that parses stage timings and writes JSON to GCS."""
  stage_cmd = require_remote_util_cmd(remote_path)
  optional_flags = []
  if prestage_timing:
    optional_flags.append(f"--prestage_timing={_quote_path(prestage_timing)}")
  if gcs_cache_dir:
    optional_flags.append(f"--gcs_cache_dir={gcs_cache_dir}")
  if ref_log:
    optional_flags.append(f"--ref_log={shlex.quote(ref_log)}")
  if ref_log_gcs:
    optional_flags.append(f"--ref_log_gcs={ref_log_gcs}")
  opt_str = (" " + " ".join(optional_flags)) if optional_flags else ""
  return (
      f"if {POD0_CONDITION}; then "
      f"{stage_cmd} && "
      f"python3 {shlex.quote(remote_path)} summarize_stages "
      f"--dyn_log={shlex.quote(dyn_log)} "
      f"--out={stage_timings_gcs} "
      f"--weight_dtype={shlex.quote(weight_dtype)} "
      f"--to_maxtext_save_dtype={shlex.quote(to_maxtext_save_dtype)} "
      f"--dyn_log_gcs={dyn_log_gcs}"
      f"{opt_str}; "
      "fi"
  )


def build_compare_cmds(
    *,
    to_maxtext_ckpt: str,
    ref_step0_ckpt: str,
    dyn_step0_ckpt: str,
    ref_trained_ckpt: str,
    dyn_trained_ckpt: str,
    trained_step: int,
    ref_metrics: str,
    dyn_metrics: str,
    report_gcs: str,
    atol: float,
    maxtext_args: str,
    weight_dtype: str = "",
    to_maxtext_save_dtype: str = "",
    stage_timings_gcs: str = "",
    dyn_log_gcs: str = "",
    ref_log_gcs: str = "",
    cmp_log_gcs: str = "",
    remote_path: str = REMOTE_UTIL_PATH,
) -> tuple[str, ...]:
  """Builds shell commands to compare checkpoints (util pre-staged)."""
  extra_cli = []
  if weight_dtype:
    extra_cli.append(f"--weight_dtype={shlex.quote(weight_dtype)}")
  if to_maxtext_save_dtype:
    extra_cli.append(
        f"--to_maxtext_save_dtype={shlex.quote(to_maxtext_save_dtype)}"
    )
  if stage_timings_gcs:
    extra_cli.append(f"--stage_timings={stage_timings_gcs}")
  if dyn_log_gcs:
    extra_cli.append(f"--dyn_log_gcs={dyn_log_gcs}")
  if ref_log_gcs:
    extra_cli.append(f"--ref_log_gcs={ref_log_gcs}")
  if cmp_log_gcs:
    extra_cli.append(f"--cmp_log_gcs={cmp_log_gcs}")
  extra_cli_str = (" ".join(extra_cli) + " ") if extra_cli else ""
  return (
      require_remote_util_cmd(remote_path),
      (
          f"python3 {shlex.quote(remote_path)} compare "
          f"--to_maxtext_ckpt={to_maxtext_ckpt} "
          f"--ref_step0_ckpt={ref_step0_ckpt} "
          f"--dyn_step0_ckpt={dyn_step0_ckpt} "
          f"--ref_trained_ckpt={ref_trained_ckpt} "
          f"--dyn_trained_ckpt={dyn_trained_ckpt} "
          f"--trained_step={trained_step} "
          f"--ref_metrics={ref_metrics} "
          f"--dyn_metrics={dyn_metrics} "
          f"--report={report_gcs} "
          f"--atol={atol} "
          f"{extra_cli_str}"
          f"{maxtext_args}"
      ),
  )


# ==============================================================================
# Subcommand 1: `prestage` (Hugging Face `*.safetensors` -> shared GCS cache)
# ==============================================================================


def parse_gcs_uri(uri: str) -> tuple[str, str]:
  """Splits a `gs://bucket/prefix` URI into `(bucket_name, prefix)`."""
  if not uri.startswith("gs://"):
    raise ValueError(f"Expected gs:// URI, got {uri}")
  bucket_name, _, prefix = uri[5:].partition("/")
  if not bucket_name or not prefix.strip("/"):
    raise ValueError(f"Expected gs://<bucket>/<prefix>, got {uri}")
  return bucket_name, prefix.rstrip("/")


def list_valid_safetensors(
    client: Any, bucket_name: str, cache_prefix: str
) -> set[str]:
  """Lists non-empty `.safetensors` filenames under `gs://bucket/prefix/`."""
  return {
      os.path.basename(blob.name)
      for blob in client.list_blobs(bucket_name, prefix=f"{cache_prefix}/")
      if blob.name.endswith(".safetensors") and (blob.size or 0) > 0
  }


def cache_one_file(
    fpath: str,
    bucket_name: str,
    cache_prefix: str,
    hf_token: str | None,
    force_download: bool = False,
) -> str:
  """Streams a single `.safetensors` shard from Hugging Face Hub into GCS."""
  if storage is None or huggingface_hub is None:
    raise RuntimeError(
        "google-cloud-storage and huggingface_hub are required for prestage."
    )
  fname = os.path.basename(fpath)
  blob_name = f"{cache_prefix}/{fname}"
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  if not force_download:
    existing_blob = bucket.get_blob(blob_name, client=client)
    if existing_blob is not None and (existing_blob.size or 0) > 0:
      print(f"[Prestage] Cache hit: {fname}", flush=True)
      return fname

  time.sleep(random.uniform(0.0, 2.0))
  fs = huggingface_hub.HfFileSystem(token=hf_token)
  blob = bucket.blob(blob_name)
  max_retries = 8
  for attempt in range(max_retries):
    t0 = time.time()
    try:
      with fs.open(fpath, "rb") as remote_f:
        blob.chunk_size = 32 * 1024 * 1024  # 32 MiB
        blob.upload_from_file(remote_f, client=client, timeout=600)
      print(
          f"[Prestage] Uploaded {fname} in {time.time() - t0:.1f}s",
          flush=True,
      )
      return fname
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if attempt == max_retries - 1:
        raise
      backoff = min(30 * (attempt + 1), 120)
      print(
          f"[Prestage] Retry {attempt + 1}/{max_retries} for {fname} "
          f"after error ({exc}); sleeping {backoff}s...",
          flush=True,
      )
      time.sleep(backoff)
  return fname


def _write_prestage_timing(
    timing_out: str | None, payload: dict[str, Any]
) -> dict[str, Any]:
  """Writes prestage timing metadata to `timing_out` when requested."""
  if timing_out:
    try:
      _write_text(timing_out, json.dumps(payload, indent=2, sort_keys=True))
    except OSError as exc:
      print(
          f"[Prestage] Warning: failed to write timing file {timing_out}: "
          f"{exc}",
          flush=True,
      )
  return payload


def run_prestage(args: argparse.Namespace) -> dict[str, Any]:
  """Atomically stages Hugging Face `.safetensors` shards into GCS."""
  if storage is None or gcs_exceptions is None or huggingface_hub is None:
    raise RuntimeError(
        "google-cloud-storage and huggingface_hub are required for prestage."
    )
  prestage_start = time.time()
  force_download = bool(getattr(args, "force_download", False))
  timing_out = getattr(args, "timing_out", None)

  bucket_name, cache_prefix = parse_gcs_uri(args.gcs_cache_dir)
  parent_prefix = os.path.dirname(cache_prefix)
  safe_repo = args.hf_repo.replace("/", "_")
  complete_blob_name = f"{parent_prefix}/_locks/{safe_repo}.complete"
  lock_blob_name = f"{parent_prefix}/_locks/{safe_repo}.lock"

  client = storage.Client()
  bucket = client.bucket(bucket_name)
  complete_blob = bucket.blob(complete_blob_name)
  lock_blob = bucket.blob(lock_blob_name)

  if bool(getattr(args, "verify_only", False)):
    # A TPU job only checks the cache: no download, no lock, no marker change.
    cached = (
        list_valid_safetensors(client, bucket_name, cache_prefix)
        if complete_blob.exists(client=client)
        else set()
    )
    if not cached:
      raise RuntimeError(
          f"Shared GCS cache {args.gcs_cache_dir} is not complete for "
          f"{args.hf_repo}; run the HF -> GCS step first."
      )
    print(
        f"[Prestage] Shared GCS cache verified ({len(cached)} shards): "
        f"{args.gcs_cache_dir}",
        flush=True,
    )
    return {
        "hf_repo": args.hf_repo,
        "gcs_cache_dir": args.gcs_cache_dir,
        "cache_hit": True,
        "num_total_shards": len(cached),
    }

  if force_download and complete_blob.exists(client=client):
    try:
      complete_blob.delete(client=client)
    except gcs_exceptions.NotFound:
      pass

  wait_start = time.time()
  while True:
    if not force_download and complete_blob.exists(client=client):
      cached = list_valid_safetensors(client, bucket_name, cache_prefix)
      if cached:
        elapsed = round(time.time() - prestage_start, 3)
        print(
            f"[Prestage] Shared GCS cache ready ({len(cached)} shards, "
            f"CACHE_STATUS=WARM, elapsed={elapsed:.2f}s): "
            f"{args.gcs_cache_dir}",
            flush=True,
        )
        return _write_prestage_timing(
            timing_out,
            {
                "hf_repo": args.hf_repo,
                "gcs_cache_dir": args.gcs_cache_dir,
                "cache_hit": True,
                "force_download": False,
                "num_total_shards": len(cached),
                "num_downloaded_files": 0,
                "hf_to_gcs_seconds": elapsed,
            },
        )
      try:
        complete_blob.delete(client=client)
      except gcs_exceptions.NotFound:
        pass

    if time.time() - wait_start > args.wait_timeout_sec:
      elapsed = round(time.time() - prestage_start, 3)
      err_msg = f"Timed out waiting for shared GCS cache {args.gcs_cache_dir}"
      _write_prestage_timing(
          timing_out,
          {
              "hf_repo": args.hf_repo,
              "gcs_cache_dir": args.gcs_cache_dir,
              "cache_hit": False,
              "force_download": force_download,
              "hf_to_gcs_seconds": elapsed,
              "error": err_msg,
          },
      )
      raise TimeoutError(err_msg)

    try:
      lock_payload = json.dumps({"pid": os.getpid(), "ts": time.time()})
      lock_blob.upload_from_string(
          lock_payload, if_generation_match=0, client=client
      )
      break
    except gcs_exceptions.PreconditionFailed:
      existing_lock = bucket.get_blob(lock_blob_name, client=client)
      if existing_lock is not None and existing_lock.time_created is not None:
        age = (
            datetime.datetime.now(datetime.timezone.utc)
            - existing_lock.time_created
        ).total_seconds()
        if age > args.lock_stale_sec:
          print(
              f"[Prestage] Removing stale lock ({age:.0f}s old)...",
              flush=True,
          )
          try:
            existing_lock.delete(
                if_generation_match=existing_lock.generation, client=client
            )
          except (
              gcs_exceptions.NotFound,
              gcs_exceptions.PreconditionFailed,
          ):
            pass
          continue
      print(
          f"[Prestage] Another job is staging {args.hf_repo} to "
          f"{args.gcs_cache_dir}; waiting 15s...",
          flush=True,
      )
      time.sleep(15)

  try:
    hf_token = os.environ.get("HF_TOKEN") or None
    fs = huggingface_hub.HfFileSystem(token=hf_token)
    files = sorted(fs.glob(f"{args.hf_repo}/*.safetensors"))
    if not files:
      raise RuntimeError(f"No .safetensors files found in hf://{args.hf_repo}")

    if force_download:
      missing = list(files)
    else:
      existing = list_valid_safetensors(client, bucket_name, cache_prefix)
      missing = [f for f in files if os.path.basename(f) not in existing]
    print(
        f"[Prestage] {args.hf_repo}: {len(files)} total shards, "
        f"{len(missing)} to download into {args.gcs_cache_dir} "
        f"(force_download={force_download})",
        flush=True,
    )

    if missing:
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=args.max_workers
      ) as executor:
        futures = [
            executor.submit(
                cache_one_file,
                fpath,
                bucket_name,
                cache_prefix,
                hf_token,
                force_download,
            )
            for fpath in missing
        ]
        try:
          for fut in concurrent.futures.as_completed(futures):
            fut.result()
        except Exception:
          for pending in futures:
            pending.cancel()
          raise

    final_cached = list_valid_safetensors(client, bucket_name, cache_prefix)
    expected_names = {os.path.basename(f) for f in files}
    still_missing = sorted(expected_names - final_cached)
    if still_missing:
      raise RuntimeError(
          f"Incomplete GCS cache {args.gcs_cache_dir}; missing: "
          f"{still_missing}"
      )

    complete_blob.upload_from_string(
        json.dumps({
            "hf_repo": args.hf_repo,
            "num_shards": len(files),
            "completed_at": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
        }),
        client=client,
    )
    elapsed = round(time.time() - prestage_start, 3)
    cache_hit = len(missing) == 0
    cache_status = "WARM" if cache_hit else "COLD"
    print(
        f"[Prestage] Completed staging {len(files)} shards to "
        f"{args.gcs_cache_dir} (CACHE_STATUS={cache_status}, "
        f"downloaded={len(missing)}, elapsed={elapsed:.2f}s)",
        flush=True,
    )
    return _write_prestage_timing(
        timing_out,
        {
            "hf_repo": args.hf_repo,
            "gcs_cache_dir": args.gcs_cache_dir,
            "cache_hit": cache_hit,
            "force_download": force_download,
            "num_total_shards": len(files),
            "num_downloaded_files": len(missing),
            "hf_to_gcs_seconds": elapsed,
        },
    )
  except Exception as exc:
    elapsed = round(time.time() - prestage_start, 3)
    _write_prestage_timing(
        timing_out,
        {
            "hf_repo": args.hf_repo,
            "gcs_cache_dir": args.gcs_cache_dir,
            "cache_hit": False,
            "force_download": force_download,
            "hf_to_gcs_seconds": elapsed,
            "error": str(exc),
        },
    )
    raise
  finally:
    try:
      lock_blob.delete(client=client)
    except gcs_exceptions.NotFound:
      pass


# ==============================================================================
# Subcommand 2: `summarize_stages` (Per-Stage Timing Breakdown)
# ==============================================================================

_RE_LOAD_DYN_GCS_CACHE = re.compile(
    r"GCS caching complete in ([0-9.]+)s\.\s*Downloaded (\d+) missing files"
)
_RE_LOAD_DYN_STAGE1 = re.compile(r"\[1/3\] Mappings derived in ([0-9.]+)s")
_RE_LOAD_DYN_STAGE2 = re.compile(
    r"\[2/3\] Distributed Sharded GCS load completed in ([0-9.]+)s"
)
_RE_LOAD_DYN_STAGE3 = re.compile(
    r"\[3/3\] CPU Transformations completed in ([0-9.]+)s"
)
_RE_LOAD_DYN_TOTAL = re.compile(
    r"Total safetensors_dynamic duration:\s*([0-9.]+)s"
)
_RE_TO_MT_HF_DOWNLOAD = re.compile(
    r"\[to_maxtext\] HF snapshot_download completed in ([0-9.]+)s"
)
# Printed when `to_maxtext` gets its input from a GCS copy instead of a Hugging
# Face download, so both loaders start from the same files.
_RE_TO_MT_GCS_COPY = re.compile(
    r"\[to_maxtext\] GCS copy to local completed in ([0-9.]+)s"
)
_RE_TO_MT_TRANSFORM_MIN = re.compile(r"Elapse for transform:\s*([0-9.]+)\s*min")
_RE_TO_MT_OVERALL_MIN = re.compile(r"Overall Elapse:\s*([0-9.]+)\s*min")


def _match_last_float(pattern: re.Pattern[str], text: str) -> float | None:
  """Returns the last regex match group-1 converted to `float`, or `None`."""
  matches = pattern.findall(text)
  if not matches:
    return None
  last = matches[-1]
  val = last[0] if isinstance(last, tuple) else last
  return round(float(val), 3)


def parse_safetensors_dynamic_log(
    log_text: str,
    prestage_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Extracts all `safetensors_dynamic` stage timings from Pod-0 logs."""
  hf_to_gcs_s: float | None = None
  gcs_cache_hit: bool | None = None
  num_total_shards: int | None = None
  num_downloaded_files: int | None = None
  prestage_error: str | None = None

  if prestage_info:
    if prestage_info.get("hf_to_gcs_seconds") is not None:
      hf_to_gcs_s = round(float(prestage_info["hf_to_gcs_seconds"]), 3)
    if "cache_hit" in prestage_info:
      gcs_cache_hit = bool(prestage_info["cache_hit"])
    if prestage_info.get("num_total_shards") is not None:
      num_total_shards = int(prestage_info["num_total_shards"])
    if prestage_info.get("num_downloaded_files") is not None:
      num_downloaded_files = int(prestage_info["num_downloaded_files"])
    if prestage_info.get("error"):
      prestage_error = str(prestage_info["error"])

  cache_matches = _RE_LOAD_DYN_GCS_CACHE.findall(log_text)
  if cache_matches:
    last_sec, last_cnt = cache_matches[-1]
    if hf_to_gcs_s is None:
      hf_to_gcs_s = round(float(last_sec), 3)
    if num_downloaded_files is None:
      num_downloaded_files = int(last_cnt)
    if gcs_cache_hit is None:
      gcs_cache_hit = int(last_cnt) == 0

  s1 = _match_last_float(_RE_LOAD_DYN_STAGE1, log_text)
  s2 = _match_last_float(_RE_LOAD_DYN_STAGE2, log_text)
  s3 = _match_last_float(_RE_LOAD_DYN_STAGE3, log_text)
  total_dyn = _match_last_float(_RE_LOAD_DYN_TOTAL, log_text)
  if total_dyn is None and any(x is not None for x in (s1, s2, s3)):
    total_dyn = round(sum(x or 0.0 for x in (s1, s2, s3)), 3)

  total_e2e = (
      round((hf_to_gcs_s or 0.0) + total_dyn, 3)
      if total_dyn is not None
      else None
  )

  result: dict[str, Any] = {
      "hf_to_gcs_seconds": hf_to_gcs_s,
      "gcs_cache_hit": gcs_cache_hit,
      "cache_status": (
          None
          if gcs_cache_hit is None
          else ("WARM" if gcs_cache_hit else "COLD")
      ),
      "num_total_shards": num_total_shards,
      "num_downloaded_files": num_downloaded_files,
      "derive_mappings_seconds": s1,
      "sharded_gcs_load_seconds": s2,
      "in_memory_transform_seconds": s3,
      "total_dynamic_load_seconds": total_dyn,
      "total_no_step1_seconds": total_dyn,
      "total_e2e_seconds": total_e2e,
  }
  if prestage_error is not None:
    result["prestage_error"] = prestage_error
  return result


def parse_to_maxtext_log(log_text: str) -> dict[str, Any]:
  """Extracts `to_maxtext` stage timings from Pod-0 logs."""
  hf_download_s = _match_last_float(_RE_TO_MT_HF_DOWNLOAD, log_text)
  gcs_to_local_s = _match_last_float(_RE_TO_MT_GCS_COPY, log_text)
  step1_s = hf_download_s if hf_download_s is not None else gcs_to_local_s
  transform_min = _match_last_float(_RE_TO_MT_TRANSFORM_MIN, log_text)
  overall_min = _match_last_float(_RE_TO_MT_OVERALL_MIN, log_text)

  transform_s = (
      round(transform_min * 60.0, 3) if transform_min is not None else None
  )
  overall_s = round(overall_min * 60.0, 3) if overall_min is not None else None
  save_orbax_s = (
      round(max(0.0, overall_s - transform_s), 3)
      if (overall_s is not None and transform_s is not None)
      else None
  )
  return {
      "hf_download_seconds": hf_download_s,
      "gcs_to_local_seconds": gcs_to_local_s,
      "offline_convert_transform_seconds": transform_s,
      "offline_convert_save_orbax_seconds": save_orbax_s,
      "offline_convert_total_seconds": overall_s,
      "orbax_restore_seconds": None,
      "total_no_step1_seconds": overall_s,
      "total_e2e_seconds": (
          round((step1_s or 0.0) + overall_s, 3)
          if overall_s is not None
          else None
      ),
  }


def _safe_read_json(path_str: str) -> dict[str, Any] | None:
  """Reads and parses a JSON dict from `path_str`, returning `None` on error."""
  if not path_str:
    return None
  try:
    if not _path_exists(path_str):
      return None
    data = json.loads(_read_text(path_str))
    return data if isinstance(data, dict) else None
  except (OSError, ValueError, TypeError) as exc:
    _log(f"[StageSummary] Warning: failed to read JSON from {path_str}: {exc}")
    return None


def _safe_read_text(path_str: str) -> str:
  """Reads UTF-8 text from `path_str`, returning `''` on missing/error."""
  if not path_str:
    return ""
  try:
    if not _path_exists(path_str):
      return ""
    return _read_text(path_str)
  except OSError as exc:
    _log(f"[StageSummary] Warning: failed to read text from {path_str}: {exc}")
    return ""


def _read_safetensors_header_dtypes(blob: Any) -> list[str]:
  """Returns the dtype of every tensor listed in one shard's header."""
  (header_len,) = struct.unpack("<Q", blob.download_as_bytes(start=0, end=7))
  if not 0 < header_len <= _MAX_SAFETENSORS_HEADER_BYTES:
    raise ValueError(
        f"Implausible safetensors header length {header_len} in {blob.name}"
    )
  header = json.loads(blob.download_as_bytes(start=8, end=8 + header_len - 1))
  header.pop("__metadata__", None)
  return [tensor["dtype"] for tensor in header.values()]


def read_hf_dtypes(gcs_cache_dir: str, max_workers: int = 8) -> dict[str, int]:
  """Counts tensor dtypes in the cached `.safetensors` shards.

  Only each shard's JSON header is downloaded (8-byte length + header), e.g.
  `{"BF16": 723}` for Llama-3.1-70B.
  """
  if storage is None:
    raise RuntimeError("google-cloud-storage is required to read HF dtypes.")
  bucket_name, cache_prefix = parse_gcs_uri(gcs_cache_dir)
  client = storage.Client()
  shards = [
      blob
      for blob in client.list_blobs(bucket_name, prefix=f"{cache_prefix}/")
      if blob.name.endswith(".safetensors")
  ]
  if not shards:
    raise FileNotFoundError(f"No .safetensors shards under {gcs_cache_dir}")
  counts: collections.Counter[str] = collections.Counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
    for dtypes in pool.map(_read_safetensors_header_dtypes, shards):
      counts.update(dtypes)
  return dict(sorted(counts.items()))


def build_dtype_record(
    hf_checkpoint: dict[str, int] | None,
    weight_dtype: str | None,
    to_maxtext_save_dtype: str | None,
) -> dict[str, Any]:
  """Builds the `dtype` block written to stage timings and the report."""
  return {
      "hf_checkpoint": hf_checkpoint,
      "weight_dtype": weight_dtype,
      "to_maxtext_save_dtype": to_maxtext_save_dtype,
  }


def format_dtype_line(record: dict[str, Any]) -> str:
  """Formats `[dtype] hf=BF16 weight=bfloat16 save=bfloat16` for the log."""
  hf_counts = record.get("hf_checkpoint")
  if not hf_counts:
    hf_text = "unknown"
  elif len(hf_counts) == 1:
    hf_text = next(iter(hf_counts))
  else:
    hf_text = ",".join(f"{name}:{count}" for name, count in hf_counts.items())
  return (
      f"[dtype] hf={hf_text} weight={record.get('weight_dtype')} "
      f"save={record.get('to_maxtext_save_dtype')}"
  )


def build_log_urls(log_links: dict[str, str]) -> dict[str, str]:
  """Maps each `gs://` log link to its Cloud Console URL."""
  urls = {name: gcs_console_url(uri) for name, uri in log_links.items()}
  return {name: url for name, url in urls.items() if url}


def run_summarize_stages(args: argparse.Namespace) -> dict[str, Any]:
  """Parses `safetensors_dynamic` and optional `to_maxtext` stage timings."""
  prestage_info = _safe_read_json(getattr(args, "prestage_timing", ""))
  dyn_log_text = _safe_read_text(args.dyn_log)
  dyn_stages = parse_safetensors_dynamic_log(dyn_log_text, prestage_info)

  to_mt_stages = None
  ref_log_text = _safe_read_text(getattr(args, "ref_log", ""))
  if ref_log_text:
    to_mt_stages = parse_to_maxtext_log(ref_log_text)

  log_links: dict[str, str] = {}
  if getattr(args, "dyn_log_gcs", None):
    log_links["safetensors_dynamic_log"] = args.dyn_log_gcs
  if getattr(args, "ref_log_gcs", None):
    log_links["to_maxtext_log"] = args.ref_log_gcs

  weight_dtype = getattr(args, "weight_dtype", "") or None
  to_maxtext_save_dtype = (
      getattr(args, "to_maxtext_save_dtype", "") or weight_dtype
  )

  # The HF checkpoint dtype never changes for a run, so reuse it when this
  # summary is rewritten (the EXIT trap runs this step again).
  previous = _safe_read_json(args.out) or {}
  hf_checkpoint = (previous.get("dtype") or {}).get("hf_checkpoint")
  dtype_error: str | None = None
  gcs_cache_dir = getattr(args, "gcs_cache_dir", "")
  if not hf_checkpoint and gcs_cache_dir:
    try:
      hf_checkpoint = read_hf_dtypes(gcs_cache_dir)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      dtype_error = f"{type(exc).__name__}: {exc}"
      _log(f"[dtype] Warning: could not read HF checkpoint dtypes: {exc}")
  dtype_record = build_dtype_record(
      hf_checkpoint, weight_dtype, to_maxtext_save_dtype
  )
  if dtype_error is not None:
    dtype_record["hf_checkpoint_error"] = dtype_error

  summary: dict[str, Any] = {
      "weight_dtype": weight_dtype,
      "to_maxtext_save_dtype": to_maxtext_save_dtype,
      "dtype": dtype_record,
      "safetensors_dynamic": dyn_stages,
      "to_maxtext": to_mt_stages,
      "log_links": log_links,
      "log_urls": build_log_urls(log_links),
  }
  _write_text(args.out, json.dumps(summary, indent=2, sort_keys=True))
  _log(format_dtype_line(dtype_record))
  _log(
      "[StageSummary] safetensors_dynamic="
      f"{json.dumps(dyn_stages, sort_keys=True)}"
  )
  return summary


# ==============================================================================
# Subcommand 2: `compare` (Checkpoint & Metrics Comparison on TPU)
# ==============================================================================


def _log(message: str) -> None:
  """Logs via `max_logging` on TPU pods or falls back to stdout."""
  if max_logging is not None:
    max_logging.log(message)
  else:
    print(message, flush=True)


def _path_exists(path_str: str) -> bool:
  """Checks whether a local or GCS path exists."""
  if epath is not None:
    return epath.Path(path_str).exists()
  if path_str.startswith("gs://") and storage is not None:
    bucket_name, blob_name = parse_gcs_uri(path_str)
    return bool(storage.Client().bucket(bucket_name).blob(blob_name).exists())
  return pathlib.Path(path_str).exists()


def _read_text(path_str: str) -> str:
  """Reads UTF-8 text from a local or GCS path."""
  if epath is not None:
    return epath.Path(path_str).read_text()
  if path_str.startswith("gs://") and storage is not None:
    bucket_name, blob_name = parse_gcs_uri(path_str)
    return (
        storage.Client()
        .bucket(bucket_name)
        .blob(blob_name)
        .download_as_text(encoding="utf-8")
    )
  return pathlib.Path(path_str).read_text(encoding="utf-8")


def _write_text(path_str: str, content: str) -> None:
  """Writes UTF-8 text to a local or GCS path."""
  if epath is not None:
    epath.Path(path_str).write_text(content)
  elif path_str.startswith("gs://") and storage is not None:
    bucket_name, blob_name = parse_gcs_uri(path_str)
    content_type = (
        "application/json"
        if path_str.endswith(".json")
        else "text/plain; charset=utf-8"
    )
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        content, content_type=content_type
    )
  else:
    pathlib.Path(path_str).write_text(content, encoding="utf-8")


def _abstract_train_state(config: Any, mesh: Any) -> Any:
  """Constructs the abstract NNX training state for checkpoint loading."""
  (
      create_model_partial,
      abstract_model,
  ) = model_creation_utils.create_nnx_abstract_model(config, mesh)
  _, tx = train_utils.create_training_optimizer(
      config, abstract_model, mesh=mesh
  )

  def init_state_fn():
    model = create_model_partial()
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    return train_state_nnx.TrainStateNNX(model, optimizer)

  abstract_state, _, _ = maxtext_utils.get_abstract_state(
      config, mesh, init_state_fn, is_training=True
  )
  return abstract_state


def _load_params(
    config: Any, abstract_state: Any, path: str, layout: str = "orbax"
) -> Any:
  """Loads sharded model parameters from an Orbax or safetensors checkpoint."""
  with axis_rules(config.logical_axis_rules):
    _, params = checkpointing.load_state_if_possible(
        None,
        None,
        path,
        "",
        config.checkpoint_storage_concurrent_gb,
        abstract_state,
        use_ocdbt=config.checkpoint_storage_use_ocdbt,
        use_zarr3=config.checkpoint_storage_use_zarr3,
        source_checkpoint_layout=layout,
        maxtext_config=config,
    )
  pure = params.to_pure_dict() if hasattr(params, "to_pure_dict") else params
  jax.block_until_ready(pure)
  return pure


@functools.lru_cache(maxsize=1)
def _get_leaf_stats_fn() -> Callable[[Any, Any], tuple[Any, Any]]:
  """Returns a JIT-compiled leaf comparison function."""

  def _leaf_stats(a, b):
    a32 = a.astype(jnp.float32)
    b32 = b.astype(jnp.float32)
    equal = jnp.all((a == b) | (jnp.isnan(a) & jnp.isnan(b)))
    return jnp.max(jnp.abs(a32 - b32)), equal

  return jax.jit(_leaf_stats)


def clear_caches() -> None:
  """Clears internal LRU caches (used by unit tests)."""
  _get_leaf_stats_fn.cache_clear()
  _encoded_module_source.cache_clear()


def compare_checkpoint_pair(
    config: Any,
    abstract_state: Any,
    label: str,
    ref_path: str,
    dyn_path: str,
    atol: float,
    ref_layout: str = "orbax",
    dyn_layout: str = "orbax",
) -> dict[str, Any]:
  """Compares every parameter leaf between reference and dynamic checkpoints."""
  try:
    t0 = time.time()
    ref_params = _load_params(
        config, abstract_state, ref_path, layout=ref_layout
    )
    ref_load_s = time.time() - t0

    t0 = time.time()
    dyn_params = _load_params(
        config, abstract_state, dyn_path, layout=dyn_layout
    )
    dyn_load_s = time.time() - t0
  except Exception as exc:  # pylint: disable=broad-exception-caught
    _log(f"[{label}] FAIL: error loading checkpoint: {exc}")
    return {
        "label": label,
        "ref_path": ref_path,
        "dyn_path": dyn_path,
        "ref_layout": ref_layout,
        "dyn_layout": dyn_layout,
        "ref_load_seconds": None,
        "dyn_load_seconds": None,
        "num_common_leaves": 0,
        "only_in_ref": [],
        "only_in_dyn": [],
        "num_mismatches": 0,
        "max_abs_diff_across_all_leaves": None,
        "atol": atol,
        "result": "FAIL",
        "error": str(exc),
        "mismatches": [],
        "leaves": [],
    }

  rflat = traverse_util.flatten_dict(ref_params)
  dflat = traverse_util.flatten_dict(dyn_params)

  only_ref = sorted("/".join(map(str, k)) for k in rflat.keys() - dflat.keys())
  only_dyn = sorted("/".join(map(str, k)) for k in dflat.keys() - rflat.keys())

  leaf_stats_fn = _get_leaf_stats_fn()
  leaves = []
  mismatches = []
  for key in sorted(rflat.keys() & dflat.keys()):
    a = dflat[key]
    b = rflat[key]
    name = "/".join(map(str, key))
    shape_ok = a.shape == b.shape
    dtype_ok = a.dtype == b.dtype
    sharding_ok = getattr(a, "sharding", None) == getattr(b, "sharding", None)
    entry: dict[str, Any] = {
        "name": name,
        "dyn_shape": list(a.shape),
        "ref_shape": list(b.shape),
        "dyn_dtype": str(a.dtype),
        "ref_dtype": str(b.dtype),
        "shape_match": shape_ok,
        "dtype_match": dtype_ok,
        "sharding_match": sharding_ok,
    }
    if shape_ok:
      max_abs_arr, exact_arr = leaf_stats_fn(a, b)  # pylint: disable=not-callable
      max_abs = float(max_abs_arr)
      exact = bool(exact_arr)
      within_tol = exact if atol == 0.0 else (max_abs <= atol)
      entry["max_abs_diff"] = max_abs
      entry["exact_equal"] = exact
      entry["pass"] = bool(dtype_ok and sharding_ok and within_tol)
    else:
      entry["max_abs_diff"] = None
      entry["exact_equal"] = False
      entry["pass"] = False
    leaves.append(entry)
    if not entry["pass"]:
      mismatches.append(entry)

  max_diff = max(
      (x["max_abs_diff"] for x in leaves if x["max_abs_diff"] is not None),
      default=0.0,
  )
  passed = not only_ref and not only_dyn and not mismatches
  summary = {
      "label": label,
      "ref_path": ref_path,
      "dyn_path": dyn_path,
      "ref_layout": ref_layout,
      "dyn_layout": dyn_layout,
      "ref_load_seconds": round(ref_load_s, 3),
      "dyn_load_seconds": round(dyn_load_s, 3),
      "num_common_leaves": len(leaves),
      "only_in_ref": only_ref,
      "only_in_dyn": only_dyn,
      "num_mismatches": len(mismatches),
      "max_abs_diff_across_all_leaves": max_diff,
      "atol": atol,
      "result": "PASS" if passed else "FAIL",
      "mismatches": mismatches,
      "leaves": leaves,
  }
  _log(
      f"[{label}] {summary['result']}: common_leaves={len(leaves)}, "
      f"mismatches={len(mismatches)}, max_abs_diff={max_diff}, "
      f"ref_load={ref_load_s:.2f}s, dyn_load={dyn_load_s:.2f}s"
  )
  return summary


def load_losses(metrics_path: str) -> list[tuple[int, float]]:
  """Parses `(step, learning/loss)` pairs from a MaxText JSONL metrics file."""
  text = _safe_read_text(metrics_path)
  if not text:
    return []
  losses = []
  for idx, line in enumerate(text.splitlines()):
    line = line.strip()
    if not line:
      continue
    try:
      row = json.loads(line)
    except ValueError as exc:
      _log(
          f"[Metrics] Warning: skipping malformed JSON line {idx + 1} in "
          f"{metrics_path}: {exc}"
      )
      continue
    if isinstance(row, dict) and "learning/loss" in row:
      losses.append((int(row.get("step", idx)), float(row["learning/loss"])))
  return losses


def compare_metrics(
    ref_metrics_path: str, dyn_metrics_path: str, atol: float
) -> dict[str, Any]:
  """Compares per-step training losses between reference and dynamic runs."""
  ref_losses = load_losses(ref_metrics_path)
  dyn_losses = load_losses(dyn_metrics_path)
  diffs = []
  for (s_ref, l_ref), (s_dyn, l_dyn) in zip(ref_losses, dyn_losses):
    abs_diff = abs(l_dyn - l_ref)
    tol = max(atol, 1e-5)
    diffs.append({
        "ref_step": s_ref,
        "dyn_step": s_dyn,
        "ref_loss": l_ref,
        "dyn_loss": l_dyn,
        "abs_diff": abs_diff,
        "pass": (s_ref == s_dyn) and (abs_diff <= tol),
    })
  max_loss_diff = max((d["abs_diff"] for d in diffs), default=0.0)
  passed = (
      bool(ref_losses)
      and len(ref_losses) == len(dyn_losses)
      and all(d["pass"] for d in diffs)
  )
  summary = {
      "ref_metrics": ref_metrics_path,
      "dyn_metrics": dyn_metrics_path,
      "num_steps_ref": len(ref_losses),
      "num_steps_dyn": len(dyn_losses),
      "max_loss_abs_diff": max_loss_diff,
      "steps": diffs,
      "result": "PASS" if passed else "FAIL",
  }
  _log(
      f"METRICS_COMPARE: {summary['result']} "
      f"(steps={len(diffs)}, max_loss_abs_diff={max_loss_diff})"
  )
  return summary


def run_compare(
    args: argparse.Namespace, model_rest: list[str]
) -> dict[str, Any]:
  """Runs 4-way checkpoint and metrics parity verification on TPU."""
  if jax is None or pyconfig is None or maxtext_utils is None:
    raise RuntimeError("JAX and MaxText are required for compare.")

  config = pyconfig.initialize(["compare_ckpts"] + model_rest)
  mesh = maxtext_utils.get_mesh_from_config(config)
  abstract_state = _abstract_train_state(config, mesh)

  in_memory_summary = compare_checkpoint_pair(
      config,
      abstract_state,
      "in_memory_safetensors_vs_to_maxtext",
      args.to_maxtext_ckpt,
      config.load_parameters_path,
      args.atol,
      ref_layout="orbax",
      dyn_layout="safetensors_dynamic",
  )
  trained_label = f"step{args.trained_step}_trained_ckpt"
  step0_summary = compare_checkpoint_pair(
      config,
      abstract_state,
      "step0_ckpt",
      args.ref_step0_ckpt,
      args.dyn_step0_ckpt,
      args.atol,
  )
  trained_summary = compare_checkpoint_pair(
      config,
      abstract_state,
      trained_label,
      args.ref_trained_ckpt,
      args.dyn_trained_ckpt,
      args.atol,
  )
  metrics_summary = compare_metrics(
      args.ref_metrics, args.dyn_metrics, args.atol
  )

  stage_timings: dict[str, Any] = (
      _safe_read_json(getattr(args, "stage_timings", "")) or {}
  )
  if not stage_timings.get("to_maxtext"):
    ref_log_text = _safe_read_text(getattr(args, "ref_log_gcs", ""))
    if ref_log_text:
      stage_timings["to_maxtext"] = parse_to_maxtext_log(ref_log_text)
  if isinstance(stage_timings.get("to_maxtext"), dict):
    to_mt = stage_timings["to_maxtext"]
    if (
        to_mt.get("orbax_restore_seconds") is None
        and in_memory_summary.get("ref_load_seconds") is not None
    ):
      to_mt["orbax_restore_seconds"] = in_memory_summary["ref_load_seconds"]
    conv_s = to_mt.get("offline_convert_total_seconds")
    rest_s = to_mt.get("orbax_restore_seconds")
    if conv_s is not None or rest_s is not None:
      no_s1 = round((conv_s or 0.0) + (rest_s or 0.0), 3)
      to_mt["total_no_step1_seconds"] = no_s1
      step1_s = to_mt.get("hf_download_seconds")
      if step1_s is None:
        step1_s = to_mt.get("gcs_to_local_seconds")
      to_mt["total_e2e_seconds"] = round((step1_s or 0.0) + no_s1, 3)
    dyn_st = stage_timings.get("safetensors_dynamic") or {}
    dyn_no_s1 = dyn_st.get("total_no_step1_seconds") or dyn_st.get(
        "total_dynamic_load_seconds"
    )
    dyn_e2e = dyn_st.get("total_e2e_seconds")
    if (
        to_mt.get("total_no_step1_seconds") is not None
        and dyn_no_s1 is not None
        and dyn_no_s1 > 0
    ):
      stage_timings["speedup_no_step1"] = round(
          to_mt["total_no_step1_seconds"] / dyn_no_s1, 2
      )
    if (
        to_mt.get("total_e2e_seconds") is not None
        and dyn_e2e is not None
        and dyn_e2e > 0
    ):
      stage_timings["speedup_e2e"] = round(
          to_mt["total_e2e_seconds"] / dyn_e2e, 2
      )

  log_links: dict[str, str] = {}
  if getattr(args, "dyn_log_gcs", None):
    log_links["safetensors_dynamic_log"] = args.dyn_log_gcs
  if getattr(args, "ref_log_gcs", None):
    log_links["to_maxtext_log"] = args.ref_log_gcs
  if getattr(args, "cmp_log_gcs", None):
    log_links["compare_log"] = args.cmp_log_gcs
  log_urls = build_log_urls(log_links)
  if stage_timings and log_links:
    stage_timings["log_links"] = log_links
    stage_timings["log_urls"] = log_urls

  cfg_weight_dtype = str(getattr(config, "weight_dtype", ""))
  weight_dtype = getattr(args, "weight_dtype", "") or cfg_weight_dtype or None
  to_maxtext_save_dtype = (
      getattr(args, "to_maxtext_save_dtype", "") or weight_dtype
  )

  overall_pass = (
      in_memory_summary["result"] == "PASS"
      and step0_summary["result"] == "PASS"
      and trained_summary["result"] == "PASS"
      and metrics_summary["result"] == "PASS"
  )
  report = {
      "model_name": config.model_name,
      "weight_dtype": weight_dtype,
      "to_maxtext_save_dtype": to_maxtext_save_dtype,
      "dtype": build_dtype_record(
          (stage_timings.get("dtype") or {}).get("hf_checkpoint"),
          weight_dtype,
          to_maxtext_save_dtype,
      ),
      "overall_result": "PASS" if overall_pass else "FAIL",
      "stage_timings": stage_timings,
      "log_links": log_links,
      "log_urls": log_urls,
      "in_memory_safetensors_vs_to_maxtext": in_memory_summary,
      "step0_checkpoint_comparison": step0_summary,
      f"step{args.trained_step}_trained_checkpoint_comparison": (
          trained_summary
      ),
      "training_metrics_comparison": metrics_summary,
  }
  if jax.process_index() == 0:
    if getattr(args, "stage_timings", None) and stage_timings:
      _write_text(
          args.stage_timings,
          json.dumps(stage_timings, indent=2, sort_keys=True),
      )
    _write_text(args.report, json.dumps(report, indent=2, sort_keys=True))
  _log("OVERALL_COMPARE_RESULT: " + report["overall_result"])
  if not overall_pass:
    sys.exit(1)
  return report


def build_arg_parser() -> argparse.ArgumentParser:
  """Constructs the CLI argument parser for remote TPU pod subcommands."""
  parser = argparse.ArgumentParser(
      description="MaxText safetensors_dynamic E2E helper utilities."
  )
  subparsers = parser.add_subparsers(dest="subcommand", required=True)

  prestage_parser = subparsers.add_parser(
      "prestage",
      help="Atomically stage Hugging Face *.safetensors shards into GCS.",
  )
  prestage_parser.add_argument("--hf_repo", required=True)
  prestage_parser.add_argument("--gcs_cache_dir", required=True)
  prestage_parser.add_argument("--max_workers", type=int, default=8)
  prestage_parser.add_argument("--lock_stale_sec", type=int, default=2700)
  prestage_parser.add_argument("--wait_timeout_sec", type=int, default=3600)
  prestage_parser.add_argument("--force_download", action="store_true")
  prestage_parser.add_argument("--verify_only", action="store_true")
  prestage_parser.add_argument("--timing_out", default=None)

  summary_parser = subparsers.add_parser(
      "summarize_stages",
      help="Parse per-stage timing logs and write stage_timings.json.",
  )
  summary_parser.add_argument("--dyn_log", required=True)
  summary_parser.add_argument("--prestage_timing", default="")
  summary_parser.add_argument("--ref_log", default="")
  summary_parser.add_argument("--out", required=True)
  summary_parser.add_argument("--gcs_cache_dir", default="")
  summary_parser.add_argument("--weight_dtype", default="")
  summary_parser.add_argument("--to_maxtext_save_dtype", default="")
  summary_parser.add_argument("--dyn_log_gcs", default="")
  summary_parser.add_argument("--ref_log_gcs", default="")

  compare_parser = subparsers.add_parser(
      "compare",
      help="Compare reference to_maxtext and safetensors_dynamic checkpoints.",
  )
  compare_parser.add_argument("--to_maxtext_ckpt", required=True)
  compare_parser.add_argument("--ref_step0_ckpt", required=True)
  compare_parser.add_argument("--dyn_step0_ckpt", required=True)
  compare_parser.add_argument("--ref_trained_ckpt", required=True)
  compare_parser.add_argument("--dyn_trained_ckpt", required=True)
  compare_parser.add_argument("--trained_step", type=int, default=1)
  compare_parser.add_argument("--ref_metrics", required=True)
  compare_parser.add_argument("--dyn_metrics", required=True)
  compare_parser.add_argument("--report", required=True)
  compare_parser.add_argument("--atol", type=float, default=0.0)
  compare_parser.add_argument("--weight_dtype", default="")
  compare_parser.add_argument("--to_maxtext_save_dtype", default="")
  compare_parser.add_argument("--stage_timings", default="")
  compare_parser.add_argument("--dyn_log_gcs", default="")
  compare_parser.add_argument("--ref_log_gcs", default="")
  compare_parser.add_argument("--cmp_log_gcs", default="")

  return parser


def main(argv: list[str] | None = None) -> None:
  """Entrypoint when executed as a standalone script on TPU pods."""
  parser = build_arg_parser()
  args, extra_args = parser.parse_known_args(argv)
  # `max_logging` logs at INFO, which absl drops by default outside `app.run`.
  if absl_logging is not None:
    absl_logging.set_verbosity(absl_logging.INFO)
  if args.subcommand == "prestage":
    if extra_args:
      parser.error(f"Unrecognized arguments for prestage: {extra_args}")
    run_prestage(args)
  elif args.subcommand == "summarize_stages":
    if extra_args:
      parser.error(f"Unrecognized arguments for summarize_stages: {extra_args}")
    run_summarize_stages(args)
  elif args.subcommand == "compare":
    run_compare(args, extra_args)


if __name__ == "__main__" and len(sys.argv) > 1:
  main()
