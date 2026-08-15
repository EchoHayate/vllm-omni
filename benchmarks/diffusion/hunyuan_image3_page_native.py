# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODES = ("dense_legacy", "paged_scheduler")
WORKLOADS = ("cold", "repeated", "mixed")
CONCURRENCY_LEVELS = (1, 4, 8)
MIN_REPETITIONS = 5

REQUIRED_RECORD_FIELDS = {
    "commit",
    "model_revision",
    "mode",
    "workload",
    "concurrency",
    "request_id",
    "latency_ms",
    "transferred_bytes",
    "transfer_wait_ms",
    "scheduler_ready_ms",
    "scheduler_waiting_ms",
    "reference_gather_ms",
    "peak_memory_mb",
}
REQUIRED_SUMMARY_FIELDS = {
    "throughput_qps",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_median_ms",
    "latency_mad_ms",
    "gpu_idle_percent",
    "hbm_high_water_mb",
}

_REPEATED_PROMPT = "A brown and white dog is running on the grass."
_COLD_PROMPTS = (
    "A red fox crossing a frozen river under soft morning light.",
    "A glass greenhouse filled with tropical plants during a rainstorm.",
    "A wooden sailboat passing a lighthouse at sunset.",
    "A ceramic teapot beside fresh lemons on a linen tablecloth.",
    "A snow-covered mountain village beneath a clear night sky.",
    "A blue vintage bicycle parked beside a brick wall.",
    "A hummingbird hovering over purple flowers in a garden.",
    "A small robot arranging books in a quiet library.",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled HunyuanImage3 dense_legacy/paged_scheduler end-to-end waves. "
            "Run the same command on the same commit for both modes."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--workload", choices=WORKLOADS, required=True)
    parser.add_argument(
        "--concurrency",
        type=int,
        choices=CONCURRENCY_LEVELS,
        required=True,
    )
    parser.add_argument("--warmup", type=_non_negative_int, required=True)
    parser.add_argument("--repetitions", type=_positive_int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--height", type=_positive_int, required=True)
    parser.add_argument("--width", type=_positive_int, required=True)
    parser.add_argument("--steps", type=_positive_int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--tensor-parallel-size",
        type=_positive_int,
        default=None,
        help="Defaults to the number of visible CUDA devices, or 1.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--sample-interval-ms",
        type=_positive_int,
        default=100,
        help="nvidia-smi utilization/HBM sampling interval.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repetitions < MIN_REPETITIONS:
        parser.error(f"--repetitions must be at least {MIN_REPETITIONS}; got {args.repetitions}")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    return args


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile for an empty sequence")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_records(
    records: Sequence[dict[str, Any]],
    *,
    measured_duration_s: float,
    gpu_idle_percent: float | None,
    sampled_hbm_high_water_mb: float | None,
) -> dict[str, float | None]:
    if not records:
        raise ValueError("at least one measured request record is required")
    if measured_duration_s <= 0:
        raise ValueError("measured_duration_s must be positive")

    latencies = [float(record["latency_ms"]) for record in records]
    median = float(statistics.median(latencies))
    mad = float(statistics.median(abs(value - median) for value in latencies))
    request_peak_memory = max(
        (float(record["peak_memory_mb"]) for record in records),
        default=0.0,
    )
    hbm_candidates = [request_peak_memory]
    if sampled_hbm_high_water_mb is not None:
        hbm_candidates.append(float(sampled_hbm_high_water_mb))

    summary: dict[str, float | None] = {
        "throughput_qps": len(records) / measured_duration_s,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_p99_ms": _percentile(latencies, 99),
        "latency_median_ms": median,
        "latency_mad_ms": mad,
        "gpu_idle_percent": (float(gpu_idle_percent) if gpu_idle_percent is not None else None),
        "hbm_high_water_mb": max(hbm_candidates),
    }
    if set(summary) != REQUIRED_SUMMARY_FIELDS:
        raise AssertionError("summary schema does not match the required fields")
    return summary


def write_results(
    output_path: Path,
    *,
    metadata: dict[str, Any],
    records: Sequence[dict[str, Any]],
    waves: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    for index, record in enumerate(records):
        if set(record) != REQUIRED_RECORD_FIELDS:
            missing = sorted(REQUIRED_RECORD_FIELDS - set(record))
            extra = sorted(set(record) - REQUIRED_RECORD_FIELDS)
            raise ValueError(f"record {index} has an invalid schema: missing={missing}, extra={extra}")
    if set(summary) != REQUIRED_SUMMARY_FIELDS:
        missing = sorted(REQUIRED_SUMMARY_FIELDS - set(summary))
        extra = sorted(set(summary) - REQUIRED_SUMMARY_FIELDS)
        raise ValueError(f"summary has an invalid schema: missing={missing}, extra={extra}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "records": list(records),
        "waves": list(waves),
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _visible_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip() and item.strip() != "-1"]
        if devices:
            return len(devices)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return 1
    count = len([line for line in result.stdout.splitlines() if line.strip()])
    return max(count, 1)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _write_deploy_config(
    path: Path,
    *,
    args: argparse.Namespace,
    tensor_parallel_size: int,
) -> None:
    config = {
        "pipeline": "hunyuan_image3_dit",
        "async_chunk": False,
        "trust_remote_code": True,
        "stages": [
            {
                "stage_id": 0,
                "max_num_seqs": args.concurrency,
                "max_model_len": 32768,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "enforce_eager": True,
                "trust_remote_code": True,
                "revision": args.model_revision,
                "devices": ",".join(str(index) for index in range(tensor_parallel_size)),
                "distributed_executor_backend": "mp",
                "diffusion_kv_mode": args.mode,
                "vae_use_slicing": False,
                "vae_use_tiling": False,
                "moe_backend": "flashinfer_cutlass",
                "parallel_config": {
                    "pipeline_parallel_size": 1,
                    "data_parallel_size": 1,
                    "tensor_parallel_size": tensor_parallel_size,
                    "enable_expert_parallel": True,
                    "sequence_parallel_size": 1,
                    "ulysses_degree": 1,
                    "ring_degree": 1,
                    "allgather_degree": 1,
                    "cfg_parallel_size": 1,
                    "vae_patch_parallel_size": 1,
                    "use_hsdp": False,
                    "hsdp_shard_size": -1,
                    "hsdp_replicate_size": 1,
                },
                "default_sampling_params": {
                    "seed": args.seed,
                },
            }
        ],
    }
    path.write_text(json.dumps(config, indent=2) + "\n")


def _prompt_for(
    workload: str,
    *,
    request_index: int,
) -> str:
    if workload == "repeated":
        return _REPEATED_PROMPT
    if workload == "mixed" and request_index % 2 == 0:
        return _REPEATED_PROMPT
    base = _COLD_PROMPTS[request_index % len(_COLD_PROMPTS)]
    return f"{base} Request variant {request_index}."


def _collect_rank_snapshots(value: object) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"rank", "enabled", "metrics", "active_bindings"}.issubset(value):
            snapshots.append(value)
        else:
            for nested in value.values():
                snapshots.extend(_collect_rank_snapshots(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            snapshots.extend(_collect_rank_snapshots(nested))
    return snapshots


def _page_metric_totals(omni: Any) -> dict[str, float]:
    rpc_results = omni.engine.collective_rpc(
        method="get_diffusion_page_debug_snapshots",
        timeout=120,
        stage_ids=[0],
    )
    totals = {
        "transferred_bytes": 0.0,
        "local_install_latency_s": 0.0,
        "local_kv_wait_s": 0.0,
        "reference_gather_bytes": 0.0,
        "reference_gather_latency_s": 0.0,
    }
    for snapshot in _collect_rank_snapshots(rpc_results):
        metrics = snapshot.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for field in totals:
            totals[field] += float(metrics.get(field, 0.0) or 0.0)
    return totals


def _metric_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    return {field: max(0.0, after[field] - before[field]) for field in before}


def _duration_ms(
    output: Any,
    *keys: str,
) -> float | None:
    stage_durations = getattr(output, "stage_durations", None)
    if not isinstance(stage_durations, dict):
        return None
    for key in keys:
        value = stage_durations.get(key)
        if value is None:
            continue
        numeric = float(value)
        return numeric if key.endswith("_ms") else numeric * 1000.0
    return None


class _GpuSampler:
    def __init__(
        self,
        *,
        interval_s: float,
        idle_threshold_percent: float = 5.0,
    ) -> None:
        self.interval_s = interval_s
        self.idle_threshold_percent = idle_threshold_percent
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._utilization_samples: list[float] = []
        self._memory_samples: list[float] = []

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU sampler already started")
        self._thread = threading.Thread(
            target=self._run,
            name="hunyuan-page-native-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_s * 4))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                rows = [
                    [float(item.strip()) for item in line.split(",")]
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
                if rows:
                    self._utilization_samples.append(sum(row[0] for row in rows) / len(rows))
                    self._memory_samples.append(sum(row[1] for row in rows))
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                return
            self._stop.wait(self.interval_s)

    @property
    def idle_percent(self) -> float | None:
        if not self._utilization_samples:
            return None
        idle_samples = sum(utilization <= self.idle_threshold_percent for utilization in self._utilization_samples)
        return 100.0 * idle_samples / len(self._utilization_samples)

    @property
    def hbm_high_water_mb(self) -> float | None:
        if not self._memory_samples:
            return None
        return max(self._memory_samples)

    @property
    def sample_count(self) -> int:
        return len(self._utilization_samples)


def _run_wave(
    omni: Any,
    *,
    args: argparse.Namespace,
    wave_index: int,
    measured: bool,
    commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    phase = "measured" if measured else "warmup"
    wave_id = f"{phase}-{wave_index}"
    request_offset = wave_index * args.concurrency
    prompts = [
        {
            "prompt": _prompt_for(
                args.workload,
                request_index=request_offset + slot,
            )
        }
        for slot in range(args.concurrency)
    ]
    params = OmniDiffusionSamplingParams(
        height=args.height,
        width=args.width,
        seed=args.seed + wave_index,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_provided=True,
    )

    before = _page_metric_totals(omni)
    wave_started = time.perf_counter()
    completed: list[tuple[Any, float]] = []
    outputs = omni.generate(
        prompts,
        params,
        py_generator=True,
        use_tqdm=False,
    )
    for output in outputs:
        if not bool(getattr(output, "finished", False)):
            continue
        if not getattr(output, "images", None):
            continue
        completed.append((output, time.perf_counter()))
    wave_finished = time.perf_counter()
    after = _page_metric_totals(omni)
    delta = _metric_delta(before, after)

    if len(completed) != args.concurrency:
        raise RuntimeError(f"{wave_id} completed {len(completed)} image requests; expected {args.concurrency}")

    attributable = args.concurrency == 1
    records: list[dict[str, Any]] = []
    for slot, (output, completed_at) in enumerate(completed):
        latency_ms = (completed_at - wave_started) * 1000.0
        scheduler_waiting_ms = _duration_ms(
            output,
            "scheduler_waiting_ms",
            "queue_wait_ms",
        )
        transfer_wait_ms = delta["local_kv_wait_s"] * 1000.0 if attributable else None
        non_ready_ms = (scheduler_waiting_ms or 0.0) + (transfer_wait_ms or 0.0)
        scheduler_ready_ms = max(0.0, latency_ms - non_ready_ms)
        records.append(
            {
                "commit": commit,
                "model_revision": args.model_revision,
                "mode": args.mode,
                "workload": args.workload,
                "concurrency": args.concurrency,
                "request_id": f"{wave_id}-{slot}",
                "latency_ms": latency_ms,
                "transferred_bytes": (int(delta["transferred_bytes"]) if attributable else None),
                "transfer_wait_ms": transfer_wait_ms,
                "scheduler_ready_ms": scheduler_ready_ms,
                "scheduler_waiting_ms": scheduler_waiting_ms,
                "reference_gather_ms": (delta["reference_gather_latency_s"] * 1000.0 if attributable else None),
                "peak_memory_mb": float(getattr(output, "peak_memory_mb", 0.0) or 0.0),
            }
        )

    wave = {
        "wave_id": wave_id,
        "measured": measured,
        "request_count": args.concurrency,
        "duration_ms": (wave_finished - wave_started) * 1000.0,
        "page_metrics_scope": ("request" if attributable else "wave_only_not_attributed_to_requests"),
        "page_metric_delta": {
            "transferred_bytes": int(delta["transferred_bytes"]),
            "transfer_wait_ms": delta["local_kv_wait_s"] * 1000.0,
            "local_install_ms": delta["local_install_latency_s"] * 1000.0,
            "reference_gather_bytes": int(delta["reference_gather_bytes"]),
            "reference_gather_ms": (delta["reference_gather_latency_s"] * 1000.0),
        },
    }
    return records, wave, wave_finished - wave_started


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from vllm_omni.entrypoints.omni import Omni

    tensor_parallel_size = (
        args.tensor_parallel_size if args.tensor_parallel_size is not None else _visible_device_count()
    )
    commit = _git_commit()
    measured_records: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    measured_duration_s = 0.0

    old_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
    sampler = _GpuSampler(interval_s=args.sample_interval_ms / 1000.0)

    with tempfile.TemporaryDirectory(prefix="hunyuan-page-native-benchmark-") as temp_dir:
        deploy_config = Path(temp_dir) / "deploy.json"
        _write_deploy_config(
            deploy_config,
            args=args,
            tensor_parallel_size=tensor_parallel_size,
        )
        omni = Omni(
            model=args.model,
            deploy_config=str(deploy_config),
            mode="text-to-image",
            trust_remote_code=True,
            revision=args.model_revision,
            log_stats=False,
        )
        try:
            for wave_index in range(args.warmup):
                _, wave, _ = _run_wave(
                    omni,
                    args=args,
                    wave_index=wave_index,
                    measured=False,
                    commit=commit,
                )
                waves.append(wave)

            sampler.start()
            for repetition in range(args.repetitions):
                records, wave, duration_s = _run_wave(
                    omni,
                    args=args,
                    wave_index=args.warmup + repetition,
                    measured=True,
                    commit=commit,
                )
                measured_records.extend(records)
                waves.append(wave)
                measured_duration_s += duration_s
        finally:
            sampler.stop()
            omni.close()
            if old_backend is None:
                os.environ.pop("DIFFUSION_ATTENTION_BACKEND", None)
            else:
                os.environ["DIFFUSION_ATTENTION_BACKEND"] = old_backend

    summary = summarize_records(
        measured_records,
        measured_duration_s=measured_duration_s,
        gpu_idle_percent=sampler.idle_percent,
        sampled_hbm_high_water_mb=sampler.hbm_high_water_mb,
    )
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "model": args.model,
        "model_revision": args.model_revision,
        "mode": args.mode,
        "workload": args.workload,
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gpu_sampler": {
            "sample_interval_ms": args.sample_interval_ms,
            "sample_count": sampler.sample_count,
            "idle_definition": "fraction of samples with mean visible-GPU utilization <= 5%",
            "hbm_definition": "maximum summed memory.used across visible GPUs",
        },
        "metric_semantics": {
            "latency_ms": "common wave submission to individual final image output",
            "scheduler_waiting_ms": (
                "exact scheduler_waiting_ms when exposed, otherwise orchestrator queue_wait_ms proxy"
            ),
            "scheduler_ready_ms": ("derived E2E residency proxy: latency - scheduler_waiting - transfer_wait"),
            "page_metrics": (
                "worker cumulative-counter delta; exact per request only at concurrency=1, "
                "otherwise retained at wave scope and request fields are null"
            ),
        },
        "command": [sys.executable, *sys.argv],
    }
    write_results(
        args.output_json,
        metadata=metadata,
        records=measured_records,
        waves=waves,
        summary=summary,
    )
    return {
        "metadata": metadata,
        "records": measured_records,
        "waves": waves,
        "summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(args)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Raw benchmark artifact: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
