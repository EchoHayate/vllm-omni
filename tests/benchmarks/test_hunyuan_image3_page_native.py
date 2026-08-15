# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARK_PATH = _REPO_ROOT / "benchmarks" / "diffusion" / "hunyuan_image3_page_native.py"


def _load_benchmark_module():
    assert _BENCHMARK_PATH.exists(), f"missing benchmark module: {_BENCHMARK_PATH}"
    spec = importlib.util.spec_from_file_location(
        "benchmarks.diffusion.hunyuan_image3_page_native",
        _BENCHMARK_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_args(*, repetitions: int, output_json: Path) -> list[str]:
    return [
        "--model",
        "tencent/HunyuanImage-3.0-Instruct",
        "--model-revision",
        "revision",
        "--mode",
        "paged_scheduler",
        "--workload",
        "mixed",
        "--concurrency",
        "4",
        "--warmup",
        "2",
        "--repetitions",
        str(repetitions),
        "--seed",
        "0",
        "--height",
        "1024",
        "--width",
        "1024",
        "--steps",
        "4",
        "--output-json",
        str(output_json),
    ]


def test_parse_args_rejects_fewer_than_five_measured_repetitions(tmp_path: Path) -> None:
    benchmark = _load_benchmark_module()

    with pytest.raises(SystemExit):
        benchmark.parse_args(
            _required_args(
                repetitions=4,
                output_json=tmp_path / "result.json",
            )
        )


def test_summarize_records_reports_percentiles_median_mad_and_throughput() -> None:
    benchmark = _load_benchmark_module()
    records = [
        {
            "latency_ms": latency_ms,
            "peak_memory_mb": peak_memory_mb,
        }
        for latency_ms, peak_memory_mb in zip(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [100.0, 110.0, 120.0, 130.0, 140.0],
        )
    ]

    summary = benchmark.summarize_records(
        records,
        measured_duration_s=0.2,
        gpu_idle_percent=12.5,
        sampled_hbm_high_water_mb=135.0,
    )

    assert summary == pytest.approx(
        {
            "throughput_qps": 25.0,
            "latency_p50_ms": 30.0,
            "latency_p95_ms": 48.0,
            "latency_p99_ms": 49.6,
            "latency_median_ms": 30.0,
            "latency_mad_ms": 10.0,
            "gpu_idle_percent": 12.5,
            "hbm_high_water_mb": 140.0,
        }
    )


def test_write_results_emits_required_record_and_summary_schema(tmp_path: Path) -> None:
    benchmark = _load_benchmark_module()
    output_path = tmp_path / "result.json"
    record = {
        "commit": "abc123",
        "model_revision": "revision",
        "mode": "paged_scheduler",
        "workload": "cold",
        "concurrency": 1,
        "request_id": "measured-0-0",
        "latency_ms": 123.0,
        "transferred_bytes": 4096,
        "transfer_wait_ms": 4.0,
        "scheduler_ready_ms": 100.0,
        "scheduler_waiting_ms": 19.0,
        "reference_gather_ms": 7.0,
        "peak_memory_mb": 1024.0,
    }
    summary = {
        "throughput_qps": 1.0,
        "latency_p50_ms": 123.0,
        "latency_p95_ms": 123.0,
        "latency_p99_ms": 123.0,
        "latency_median_ms": 123.0,
        "latency_mad_ms": 0.0,
        "gpu_idle_percent": 5.0,
        "hbm_high_water_mb": 1024.0,
    }

    benchmark.write_results(
        output_path,
        metadata={"model": "model", "warmup": 2, "repetitions": 5},
        records=[record],
        waves=[{"wave_id": "measured-0", "request_count": 1}],
        summary=summary,
    )

    payload = json.loads(output_path.read_text())
    assert payload["schema_version"] == 1
    assert payload["metadata"]["model"] == "model"
    assert payload["records"] == [record]
    assert payload["waves"] == [{"wave_id": "measured-0", "request_count": 1}]
    assert payload["summary"] == summary
    assert set(record) == benchmark.REQUIRED_RECORD_FIELDS
    assert set(summary) == benchmark.REQUIRED_SUMMARY_FIELDS
