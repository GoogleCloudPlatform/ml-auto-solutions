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

"""MaxText E2E TPU safetensors_dynamic vs. to_maxtext Checkpoint Comparison DAG.

Supports two execution modes (`run_mode`) across configurable models and
one- and two-slice TPU v6e topologies (`v6e-8`, `v6e-64`):
1. `e2e_only` mode:
   - Standalone `safetensors_dynamic` loading (`source_checkpoint_layout=
     safetensors_dynamic`) with 2 pre-training steps on C4 (`to-mt` and
     `cmp-ckpt` are short-circuited).
2. `compare` mode:
   - Offline conversion via `maxtext.checkpoint_conversion.to_maxtext` plus
     deterministic pre-training from both `to_maxtext` and
     `safetensors_dynamic`, followed by 4-way weight & loss verification
     (`safetensors_util.py compare`: in-memory weights leaf-by-leaf, Step-0
     Orbax checkpoints, trained Step-(N-1) Orbax checkpoints, and per-step
     training losses).

In both modes an `hf-to-gcs` job per model first copies the Hugging Face
`.safetensors` shards into the shared GCS cache on a CPU node. The TPU jobs
only check that cache, so they never download and a restarted TPU job never
repeats the download.
"""

import dataclasses
import datetime
import re

from airflow import models
from airflow.decorators import task
from airflow.models.baseoperator import chain
from airflow.models.param import Param
from airflow.operators.python import get_current_context
from airflow.utils.task_group import TaskGroup
from dags import gcs_bucket
from dags.common import test_owner
from dags.common.vm_resource import DockerImage, GkeClusters
from dags.multipod.configs import gke_config
from dags.multipod.util import safetensors_util
from xlml.apis.gke_cluster_config import GkeClusterConfig

# Environment variable -> Airflow Variable that holds its value. The value
# is read when the task runs; it must not be put in a command.
SECRET_ENV = {"HF_TOKEN": "HF_TOKEN"}

# The v6e queue has no TPU quota of its own and only uses chips that other
# queues leave idle, so a job can wait for longer than the default 60 minutes
# before it starts.
WORKLOAD_PROVISION_TIMEOUT = datetime.timedelta(hours=4)

# The HF -> GCS download runs once per model on a CPU node, before the TPU
# jobs of that model.
HF_TO_GCS_CLUSTER = GkeClusters.CPU_N2_HIGHMEM_64_CLUSTER
HF_TO_GCS_TIME_OUT_IN_MIN = 180

# The slices of one job can start minutes apart, because each slice waits for
# its own nodes. The default 300 s for `jax.distributed.initialize` is too short
# for that: the first slice aborts before the last slice has joined.
JAX_DISTRIBUTED_INIT_TIMEOUT_SEC = 1800

RUN_NAME_TEMPLATE = (
    "{{ params.run_name if params.run_name else 'safetensors-dynamic-' ~"
    " ts_nodash }}"
)
FORCE_DOWNLOAD_TEMPLATE = (
    "{{ '--force_download' if params.get('force_hf_download', False) else"
    " '' }}"
)

RUN_MODE_E2E_ONLY = "e2e_only"
RUN_MODE_COMPARE = "compare"
VALID_RUN_MODES = (RUN_MODE_E2E_ONLY, RUN_MODE_COMPARE)


def resolve_comparison_enabled(
    params: dict[str, object] | None = None,
    conf: dict[str, object] | None = None,
) -> bool:
  """Resolves whether `to-mt` + `cmp-ckpt` should run for this DAG run."""
  mode = (conf or {}).get("run_mode") or (params or {}).get("run_mode")
  if mode == RUN_MODE_COMPARE:
    return True
  if mode == RUN_MODE_E2E_ONLY:
    return False
  raise ValueError(f"run_mode must be one of {VALID_RUN_MODES}; got {mode!r}")


@task.short_circuit(task_id="check_comparison_enabled")
def check_comparison_enabled() -> bool:
  """Skips `to-mt` and `cmp-ckpt` when running in `e2e_only` mode."""
  context = get_current_context()
  dag_run = context.get("dag_run")
  conf = (dag_run.conf if dag_run else {}) or {}
  params = context.get("params") or {}
  return resolve_comparison_enabled(params=params, conf=conf)


@dataclasses.dataclass(frozen=True)
class SafetensorsComparisonSpec:
  """Declarative specification for a model & slice-topology safetensors test."""

  model_name: str
  hf_repo: str
  cluster: GkeClusterConfig = dataclasses.field(
      default_factory=lambda: GkeClusters.TPU_V6E_BODABORG_NAP_CLUSTER
  )
  core_count: int = 8
  num_slices: int = 1
  scan_layers: bool = True
  weight_dtype: str = "float32"
  to_maxtext_save_dtype: str | None = None
  steps: int = 2
  per_device_batch_size: int = 1
  max_target_length: int = 2048
  atol: float = 0.0
  comparison_supported: bool = True
  extra_maxtext_flags: tuple[str, ...] = ()
  extra_to_maxtext_flags: tuple[str, ...] = ()
  time_out_in_min: int = 60
  priority: str = "medium"

  @property
  def effective_to_maxtext_save_dtype(self) -> str:
    """Returns `--save_dtype` for `to_maxtext` (defaults to `weight_dtype`)."""
    return self.to_maxtext_save_dtype or self.weight_dtype

  def dyn_ckpt_save_flags_template(self) -> str:
    """Returns Jinja-templated checkpoint flags for `safetensors-dyn`."""
    save_flags = "checkpoint_period=1 save_checkpoint_on_start=true"
    skip_flags = (
        "save_checkpoint_on_start=false save_checkpoint_on_completion=false"
    )
    if not self.comparison_supported:
      return skip_flags
    return (
        "{% if params.get('run_mode') == 'compare' %}"
        f"{save_flags}"
        "{% else %}"
        f"{skip_flags}"
        "{% endif %}"
    )

  @property
  def group_id(self) -> str:
    """Returns an Airflow-safe TaskGroup ID (alphanumeric, dashes, `_`)."""
    safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", self.model_name)
    tpu_ver = f"v{self.cluster.device_version.value}"
    if self.num_slices > 1:
      return f"{safe_model}-{self.num_slices}x-{tpu_ver}-{self.core_count}"
    return f"{safe_model}-{tpu_ver}-{self.core_count}"


def shared_hf_gcs_cache_dir(hf_repo: str) -> str:
  """Returns the shared GCS cache directory of a Hugging Face repo."""
  return (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/hf_safetensors_cache/"
      f"{hf_repo.replace('/', '_')}"
  )


def hf_to_gcs_output_dir(model_name: str) -> str:
  """Returns the per-run GCS directory of a model's `hf-to-gcs` job."""
  return (
      f"{gcs_bucket.BASE_OUTPUT_DIR}/{model_name}/safetensors_compare/"
      f"hf_to_gcs/{RUN_NAME_TEMPLATE}"
  )


def hf_to_gcs_timing_path(model_name: str) -> str:
  """Returns the GCS path of the download timing an `hf-to-gcs` job writes."""
  return f"{hf_to_gcs_output_dir(model_name)}/prestage_timing.json"


def create_hf_to_gcs_task_group(
    model_name: str,
    hf_repo: str,
    owner: str = test_owner.KEVIN_Y,
) -> TaskGroup:
  """Builds the CPU job that copies a model's HF shards into the GCS cache."""
  safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)
  with TaskGroup(group_id=f"hf-to-gcs-{safe_model}") as hf_to_gcs_group:
    local_log = "/tmp/hf_to_gcs.log"
    log_gcs = f"{hf_to_gcs_output_dir(model_name)}/hf_to_gcs.log"
    prestage_cmd = " && ".join(
        safetensors_util.build_prestage_cmds(
            hf_repo=hf_repo,
            gcs_cache_dir=shared_hf_gcs_cache_dir(hf_repo),
            timing_out=hf_to_gcs_timing_path(model_name),
            force_download_expr=FORCE_DOWNLOAD_TEMPLATE,
        )
    )
    hf_to_gcs_cmds = (
        safetensors_util.stage_remote_util_cmd(),
        safetensors_util.pod0_upload_log_trap_cmd(local_log, log_gcs),
        safetensors_util.pod0_tee_cmd(prestage_cmd, local_log),
    )
    gke_config.get_maxtext_cpu_end_to_end_gke_config(
        time_out_in_min=HF_TO_GCS_TIME_OUT_IN_MIN,
        test_name="hf-to-gcs",
        run_model_cmds=hf_to_gcs_cmds,
        secret_env=SECRET_ENV,
        docker_image="{{ params.docker_image }}",
        cluster=HF_TO_GCS_CLUSTER,
        test_owner=owner,
        priority="medium",
        use_gcluster=True,
    ).run(skip_post_process=True)
  return hf_to_gcs_group


def create_safetensors_comparison_task_group(
    spec: SafetensorsComparisonSpec,
    owner: str = test_owner.KEVIN_Y,
    hf_to_gcs_group: TaskGroup | None = None,
) -> TaskGroup:
  """Builds the `safetensors-dyn` (and optional `to-mt` + `cmp-ckpt`) group.

  `hf_to_gcs_group` is the model's `hf-to-gcs` job; `safetensors-dyn` starts
  after it.
  """
  with TaskGroup(group_id=spec.group_id) as model_group:
    run_name = RUN_NAME_TEMPLATE
    model = spec.model_name
    hf_repo = spec.hf_repo
    weight_dtype = spec.weight_dtype
    to_mt_save_dtype = spec.effective_to_maxtext_save_dtype
    scan_layers = "true" if spec.scan_layers else "false"
    trained_step = spec.steps - 1
    extra_flags_str = (
        " " + " ".join(spec.extra_maxtext_flags)
        if spec.extra_maxtext_flags
        else ""
    )
    extra_to_mt_str = (
        " " + " ".join(spec.extra_to_maxtext_flags)
        if spec.extra_to_maxtext_flags
        else ""
    )

    cluster = spec.cluster.override(core_count=spec.core_count)
    tpu_ver = f"v{cluster.device_version.value}"

    topology_tag = f"{spec.num_slices}x_{tpu_ver}_{spec.core_count}"
    base_dir = (
        f"{gcs_bucket.BASE_OUTPUT_DIR}/{model}/safetensors_compare/"
        f"{topology_tag}/{run_name}"
    )
    markers_dir = f"{base_dir}/markers"
    logs_dir = f"{base_dir}/logs"
    dyn_log_gcs = f"{logs_dir}/safetensors_dynamic.log"
    ref_log_gcs = f"{logs_dir}/to_maxtext.log"
    cmp_log_gcs = f"{logs_dir}/compare_ckpts.log"
    stage_timings_gcs = f"{base_dir}/stage_timings.json"

    to_maxtext_dir = f"{base_dir}/to_maxtext"
    to_maxtext_step0_ckpt = f"{to_maxtext_dir}/0/items"

    train_ref_dir = f"{base_dir}/train_to_maxtext"
    train_ref_run = "run"
    train_ref_step0_ckpt = (
        f"{train_ref_dir}/{train_ref_run}/checkpoints/0/items"
    )
    train_ref_trained_ckpt = (
        f"{train_ref_dir}/{train_ref_run}/checkpoints/{trained_step}/items"
    )
    ref_metrics_gcs = f"{train_ref_dir}/metrics_to_maxtext.txt"

    train_dyn_dir = f"{base_dir}/train_safetensors_dynamic"
    train_dyn_run = "run"
    train_dyn_step0_ckpt = (
        f"{train_dyn_dir}/{train_dyn_run}/checkpoints/0/items"
    )
    train_dyn_trained_ckpt = (
        f"{train_dyn_dir}/{train_dyn_run}/checkpoints/{trained_step}/items"
    )
    dyn_metrics_gcs = f"{train_dyn_dir}/metrics_safetensors_dynamic.txt"

    report_gcs = f"{base_dir}/comparison_report.json"

    shared_hf_gcs_cache = shared_hf_gcs_cache_dir(hf_repo)
    dyn_load_parameters_path = shared_hf_gcs_cache

    # Keep `enable_checkpointing=true` even when not writing checkpoints so
    # multi-host / multi-slice TPU runs initialize `jax.distributed` on `main`.
    dyn_ckpt_save_flags = spec.dyn_ckpt_save_flags_template()
    ref_ckpt_save_flags = "checkpoint_period=1 save_checkpoint_on_start=true"

    base_train_flags = (
        "src/maxtext/configs/base.yml "
        f"model_name={model} "
        f"scan_layers={scan_layers} "
        f"weight_dtype={weight_dtype} "
        f"tokenizer_path={hf_repo} "
        "tokenizer_type=huggingface "
        "dataset_type=hf "
        "hf_path=allenai/c4 "
        "hf_data_dir=en "
        "train_split=train "
        "enable_data_shuffling=false "
        f"per_device_batch_size={spec.per_device_batch_size} "
        f"max_target_length={spec.max_target_length} "
        f"steps={spec.steps} "
        "enable_checkpointing=true "
        "async_checkpointing=false "
        "jax_distributed_initialization_timeout="
        f"{JAX_DISTRIBUTED_INIT_TIMEOUT_SEC} "
    )
    dyn_train_flags = (
        f"{base_train_flags}{dyn_ckpt_save_flags}{extra_flags_str}"
    )
    ref_train_flags = (
        f"{base_train_flags}{ref_ckpt_save_flags}{extra_flags_str}"
    )

    # `safetensors_dynamic` workflow (always runs)
    dyn_local_log = "/tmp/safetensors_dynamic.log"
    # The model's `hf-to-gcs` job wrote the download timing of this run.
    prestage_timing_gcs = hf_to_gcs_timing_path(model)
    dyn_pod0_prep = (
        (
            f"if gsutil ls {train_dyn_dir} >/dev/null 2>&1; then "
            f"gsutil -m rm -rf {train_dyn_dir} || true; "
            "fi"
        ),
        # No download on the TPU job: fail early when the cache is not ready.
        *safetensors_util.build_prestage_cmds(
            hf_repo=hf_repo,
            gcs_cache_dir=shared_hf_gcs_cache,
            verify_only=True,
        ),
    )

    stage_summary_cmd = safetensors_util.build_stage_summary_cmd(
        dyn_log=dyn_local_log,
        prestage_timing=prestage_timing_gcs,
        gcs_cache_dir=shared_hf_gcs_cache,
        stage_timings_gcs=stage_timings_gcs,
        weight_dtype=weight_dtype,
        to_maxtext_save_dtype=to_mt_save_dtype,
        dyn_log_gcs=dyn_log_gcs,
    )
    stage_summary_trap_cmd = (
        f"python3 {safetensors_util.REMOTE_UTIL_PATH} summarize_stages "
        f"--dyn_log={dyn_local_log} "
        f"--out={stage_timings_gcs} "
        f"--weight_dtype={weight_dtype} "
        f"--to_maxtext_save_dtype={to_mt_save_dtype} "
        f"--dyn_log_gcs={dyn_log_gcs} "
        f"--prestage_timing={prestage_timing_gcs} "
        f"--gcs_cache_dir={shared_hf_gcs_cache}"
    )

    safetensors_dynamic_cmds = (
        'export HF_HOME="/mnt/shm/hf_cache"',
        'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
        safetensors_util.stage_remote_util_cmd(),
        safetensors_util.pod0_upload_log_trap_cmd(
            dyn_local_log, dyn_log_gcs, on_exit_cmd=stage_summary_trap_cmd
        ),
        safetensors_util.pod0_tee_cmd(
            safetensors_util.pod0_barrier_cmd(
                dyn_pod0_prep, f"{markers_dir}/safetensors_dynamic_prep.done"
            ),
            dyn_local_log,
        ),
        safetensors_util.pod0_tee_cmd(
            "python3 -m maxtext.trainers.pre_train.train "
            f"{dyn_train_flags} "
            f"base_output_directory={train_dyn_dir} "
            f"run_name={train_dyn_run} "
            f"load_parameters_path={dyn_load_parameters_path} "
            "source_checkpoint_layout=safetensors_dynamic "
            "metrics_file=/tmp/metrics_safetensors_dynamic.txt",
            dyn_local_log,
        ),
        safetensors_util.pod0_tee_cmd(stage_summary_cmd, dyn_local_log),
        (
            f"if {safetensors_util.POD0_CONDITION}; then "
            f"gsutil cp /tmp/metrics_safetensors_dynamic.txt {dyn_metrics_gcs};"
            " fi"
        ),
    )

    safetensors_dynamic_task = gke_config.get_gke_config(
        time_out_in_min=spec.time_out_in_min,
        test_name="safetensors-dyn",
        workload_provision_timeout=WORKLOAD_PROVISION_TIMEOUT,
        run_model_cmds=safetensors_dynamic_cmds,
        secret_env=SECRET_ENV,
        docker_image="{{ params.docker_image }}",
        cluster=cluster,
        num_slices=spec.num_slices,
        test_owner=owner,
        priority=spec.priority,
        use_gcluster=True,
        mounts="/dev/shm;/mnt/shm;rw",
    ).run(skip_post_process=True)
    if hf_to_gcs_group is not None:
      chain(hf_to_gcs_group, safetensors_dynamic_task)

    if spec.comparison_supported:
      comparison_gate = check_comparison_enabled()
      local_hf_dir = f"/mnt/shm/{model}_hf"
      ref_local_log = "/tmp/to_maxtext.log"
      cmp_local_log = "/tmp/compare_ckpts.log"

      # Task 1: Original `to_maxtext` workflow (Pod 0 converts; all pods run
      # `train.py` from the converted checkpoint).
      to_mt_pod0_prep = (
          (
              f"if gsutil ls {to_maxtext_dir} >/dev/null 2>&1; then "
              f"gsutil -m rm -rf {to_maxtext_dir} {train_ref_dir} || true; "
              "fi"
          ),
          (
              'python3 -c "import os, time, huggingface_hub as h; '
              "t0 = time.time(); "
              f"h.snapshot_download('{hf_repo}', local_dir='{local_hf_dir}', "
              "allow_patterns=["
              "'*.safetensors','*.json','*.model','*.txt','*.jinja','*.py'"
              "], "
              "token=os.environ.get('HF_TOKEN'), max_workers=16); "
              "print(f'[to_maxtext] HF snapshot_download completed in "
              "{time.time() - t0:.2f}s', flush=True)\""
          ),
          (
              "env JAX_PLATFORMS=cpu python3 -m "
              "maxtext.checkpoint_conversion.to_maxtext "
              "src/maxtext/configs/base.yml "
              f"model_name={model} "
              f"base_output_directory={to_maxtext_dir} "
              f"scan_layers={scan_layers} "
              "hardware=cpu "
              f"skip_jax_distributed_system=True{extra_to_mt_str} "
              f"--hf_model_path={local_hf_dir} "
              "--lazy_load_tensors=True "
              f"--save_dtype={to_mt_save_dtype}"
          ),
          f"rm -rf {local_hf_dir}",
      )

      to_mt_body_cmd = (
          "python3 -m maxtext.trainers.pre_train.train "
          f"{ref_train_flags} "
          f"base_output_directory={train_ref_dir} "
          f"run_name={train_ref_run} "
          f"load_parameters_path={to_maxtext_step0_ckpt} "
          "metrics_file=/tmp/metrics_ref.jsonl"
      )
      to_mt_post_cmd = (
          f"if {safetensors_util.POD0_CONDITION}; then "
          f"gsutil cp /tmp/metrics_ref.jsonl {ref_metrics_gcs} && "
          f"gsutil -q cp {ref_local_log} {ref_log_gcs}; "
          "fi"
      )

      to_maxtext_cmds = (
          'export HF_HOME="/mnt/shm/hf_cache"',
          'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
          safetensors_util.stage_remote_util_cmd(),
          safetensors_util.pod0_upload_log_trap_cmd(ref_local_log, ref_log_gcs),
          safetensors_util.pod0_tee_cmd(
              safetensors_util.pod0_barrier_cmd(
                  to_mt_pod0_prep, f"{markers_dir}/to_maxtext_prep.done"
              ),
              ref_local_log,
          ),
          safetensors_util.pod0_tee_cmd(to_mt_body_cmd, ref_local_log),
          to_mt_post_cmd,
      )

      to_maxtext_task = gke_config.get_gke_config(
          time_out_in_min=spec.time_out_in_min,
          test_name="to-mt",
          workload_provision_timeout=WORKLOAD_PROVISION_TIMEOUT,
          run_model_cmds=to_maxtext_cmds,
          secret_env=SECRET_ENV,
          docker_image="{{ params.docker_image }}",
          cluster=cluster,
          num_slices=spec.num_slices,
          test_owner=owner,
          priority=spec.priority,
          use_gcluster=True,
          mounts="/dev/shm;/mnt/shm;rw",
      ).run(skip_post_process=True)

      # Task 3: Compare weights, checkpoints and losses.
      compare_maxtext_args = (
          "src/maxtext/configs/base.yml "
          f"model_name={model} "
          f"base_output_directory={base_dir} "
          "run_name=compare_ckpts "
          f"load_parameters_path={dyn_load_parameters_path} "
          "source_checkpoint_layout=safetensors_dynamic "
          f"scan_layers={scan_layers} "
          f"weight_dtype={weight_dtype} "
          "dtype=float32 "
          f"per_device_batch_size={spec.per_device_batch_size} "
          "max_target_length=512 "
          "enable_checkpointing=true "
          "async_checkpointing=false "
          # Slices can start minutes apart; the 300 s default is too short.
          "jax_distributed_initialization_timeout="
          f"{JAX_DISTRIBUTED_INIT_TIMEOUT_SEC}{extra_flags_str}"
      )
      raw_compare_cmds = safetensors_util.build_compare_cmds(
          to_maxtext_ckpt=to_maxtext_step0_ckpt,
          ref_step0_ckpt=train_ref_step0_ckpt,
          dyn_step0_ckpt=train_dyn_step0_ckpt,
          ref_trained_ckpt=train_ref_trained_ckpt,
          dyn_trained_ckpt=train_dyn_trained_ckpt,
          trained_step=trained_step,
          ref_metrics=ref_metrics_gcs,
          dyn_metrics=dyn_metrics_gcs,
          report_gcs=report_gcs,
          atol=spec.atol,
          maxtext_args=compare_maxtext_args,
          weight_dtype=weight_dtype,
          to_maxtext_save_dtype=to_mt_save_dtype,
          stage_timings_gcs=stage_timings_gcs,
          dyn_log_gcs=dyn_log_gcs,
          ref_log_gcs=ref_log_gcs,
          cmp_log_gcs=cmp_log_gcs,
      )
      compare_cmds = (
          'export HF_HOME="/mnt/shm/hf_cache"',
          'export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=20480"',
          safetensors_util.stage_remote_util_cmd(),
          safetensors_util.pod0_upload_log_trap_cmd(cmp_local_log, cmp_log_gcs),
          safetensors_util.pod0_tee_cmd(raw_compare_cmds[1], cmp_local_log),
      )

      compare_checkpoints_task = gke_config.get_gke_config(
          time_out_in_min=spec.time_out_in_min,
          test_name="cmp-ckpt",
          workload_provision_timeout=WORKLOAD_PROVISION_TIMEOUT,
          run_model_cmds=compare_cmds,
          secret_env=SECRET_ENV,
          docker_image="{{ params.docker_image }}",
          cluster=cluster,
          num_slices=spec.num_slices,
          test_owner=owner,
          priority=spec.priority,
          use_gcluster=True,
          mounts="/dev/shm;/mnt/shm;rw",
      ).run(skip_post_process=True)

      chain(comparison_gate, to_maxtext_task)
      chain(
          [to_maxtext_task, safetensors_dynamic_task],
          compare_checkpoints_task,
      )

  return model_group


# TODO(KevinYenky): Add qwen3-30b-a3b back (`v6e-16`, 1 and 2 slices,
# `sparse_matmul=false`, 120 min) once slow safetensors read of checkpoints with
# many small tensors fixed.
TEST_SPECS: tuple[SafetensorsComparisonSpec, ...] = (
    SafetensorsComparisonSpec(
        model_name="qwen3-0.6b",
        hf_repo="Qwen/Qwen3-0.6B",
        core_count=8,
        num_slices=1,
    ),
    SafetensorsComparisonSpec(
        model_name="qwen3-0.6b",
        hf_repo="Qwen/Qwen3-0.6B",
        core_count=8,
        num_slices=2,
        extra_maxtext_flags=("dcn_data_parallelism=2",),
    ),
    SafetensorsComparisonSpec(
        model_name="gemma2-2b",
        hf_repo="google/gemma-2-2b",
        core_count=8,
        num_slices=1,
        weight_dtype="bfloat16",
        to_maxtext_save_dtype="float32",
    ),
    SafetensorsComparisonSpec(
        model_name="gemma2-2b",
        hf_repo="google/gemma-2-2b",
        core_count=8,
        num_slices=2,
        weight_dtype="bfloat16",
        to_maxtext_save_dtype="float32",
        extra_maxtext_flags=("dcn_data_parallelism=2",),
    ),
    SafetensorsComparisonSpec(
        model_name="llama3.1-70b",
        hf_repo="meta-llama/Llama-3.1-70B",
        core_count=64,
        num_slices=1,
        weight_dtype="bfloat16",
        time_out_in_min=90,
    ),
    SafetensorsComparisonSpec(
        model_name="llama3.1-70b",
        hf_repo="meta-llama/Llama-3.1-70B",
        core_count=64,
        num_slices=2,
        weight_dtype="bfloat16",
        extra_maxtext_flags=("dcn_data_parallelism=2",),
        time_out_in_min=90,
    ),
)

with models.DAG(
    dag_id="maxtext_e2e_tpu_safetensors_dynamic",
    schedule=None,
    tags=[
        "multipod_team",
        "maxtext",
        "safetensors_dynamic",
        "checkpoint",
        "e2e",
        "TPU",
    ],
    start_date=datetime.datetime(2026, 6, 10),
    catchup=False,
    default_args={
        "owner": test_owner.KEVIN_Y,
    },
    params={
        "docker_image": Param(
            default=DockerImage.MAXTEXT_TPU_JAX_STABLE.value,
            type="string",
            description="MaxText Docker image URI to test",
        ),
        "run_name": Param(
            default="",
            type="string",
            description=(
                "Unique run name for checkpoints and comparison outputs "
                "(defaults to safetensors-dynamic-{{ ts_nodash }})"
            ),
        ),
        "run_mode": Param(
            type="string",
            enum=list(VALID_RUN_MODES),
            description=(
                "Workflow mode (required): 'e2e_only' runs only standalone "
                "safetensors-dyn; 'compare' runs to-mt + safetensors-dyn + "
                "cmp-ckpt (weight/ckpt/loss comparison)."
            ),
        ),
        "force_hf_download": Param(
            default=False,
            type="boolean",
            description=(
                "If true, the hf-to-gcs job of each model forces a cold HF "
                "-> GCS download even when the shared GCS safetensors cache "
                "is already populated. Use it for benchmark runs; otherwise "
                "only a model's first run measures HF -> GCS and later runs "
                "report WARM."
            ),
        ),
    },
) as dag:
  hf_to_gcs_groups: dict[tuple[str, str], TaskGroup] = {}
  for test_spec in TEST_SPECS:
    model_key = (test_spec.model_name, test_spec.hf_repo)
    if model_key not in hf_to_gcs_groups:
      hf_to_gcs_groups[model_key] = create_hf_to_gcs_task_group(*model_key)
    create_safetensors_comparison_task_group(
        test_spec, hf_to_gcs_group=hf_to_gcs_groups[model_key]
    )
