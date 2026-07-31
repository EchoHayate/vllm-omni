# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure future-ready latency for asynchronous KV prefetch H2D copies."""

import argparse
import statistics
import time

import torch

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import (
    OmniKVCacheConfig,
    OmniKVTransferManager,
)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def _format_metrics(name: str, values: list[float]) -> str:
    return (
        f"{name}: median={statistics.median(values):.3f} ms, "
        f"p95={_percentile(values, 0.95):.3f} ms, "
        f"range=[{min(values):.3f}, {max(values):.3f}] ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size-mib", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.accelerator.set_device_index(device.index)
    element_size = torch.empty((), dtype=torch.float32).element_size()
    element_count = args.size_mib * 1024 * 1024 // element_size
    source = torch.empty(element_count, dtype=torch.float32, pin_memory=True)
    source.normal_()

    manager = OmniKVTransferManager(
        OmniKVCacheConfig(need_recv_cache=True),
        async_prefetch=True,
    )

    def fake_receive(
        request_id,
        target_device=None,
        *,
        sender_info=None,
        non_blocking_copy=False,
        copy_stream=None,
        deferred_copy_sources=None,
        deferred_pool_buffers=None,
    ):
        assert non_blocking_copy
        assert copy_stream is not None
        assert deferred_copy_sources is not None
        assert deferred_pool_buffers is not None
        destination = source.to(target_device, non_blocking=True)
        deferred_copy_sources.append(source)
        return {"layer_blocks": {"key_cache": [destination], "value_cache": []}}, source.nbytes

    manager.receive_kv_cache_for_request = fake_receive
    submit_ms: list[float] = []
    blocking_ready_ms: list[float] = []
    peak_memory_mib: list[float] = []
    event_incomplete = 0

    for iteration in range(args.warmup + args.iterations):
        torch.accelerator.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        started = time.perf_counter()
        result = manager._prefetch_payload(
            f"bench-{iteration}",
            None,
            device,
        )
        submitted = time.perf_counter()
        event_incomplete += int(not result.ready_event.query())
        result.ready_event.synchronize()
        completed = time.perf_counter()
        peak_mib = torch.accelerator.max_memory_allocated() / (1024 * 1024)
        manager._release_prefetch_result(result)

        if iteration >= args.warmup:
            submit_ms.append((submitted - started) * 1000)
            blocking_ready_ms.append((completed - started) * 1000)
            peak_memory_mib.append(peak_mib)
        del result

    manager.shutdown_prefetch()
    print(f"device={device}, size={args.size_mib} MiB, warmup={args.warmup}, iterations={args.iterations}")
    print(_format_metrics("event_future_ready", submit_ms))
    print(_format_metrics("blocking_future_ready", blocking_ready_ms))
    print(
        f"event_incomplete_at_return={event_incomplete}/{args.warmup + args.iterations}, "
        f"peak_memory={max(peak_memory_mib):.1f} MiB"
    )


if __name__ == "__main__":
    main()
