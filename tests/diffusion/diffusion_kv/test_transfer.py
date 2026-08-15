# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionPageMetrics
from vllm_omni.diffusion.diffusion_kv.page import DiffusionPageBinding, PageState
from vllm_omni.diffusion.diffusion_kv.transfer import (
    PageCopy,
    PageEndpoint,
    PageTransferPlan,
    PageTransferSessionManager,
    TransferState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class FakeEvent:
    def __init__(self, complete: bool = True) -> None:
        self.complete = complete

    def query(self) -> bool:
        return self.complete


class FakeClock:
    def __init__(self, now: float = 1.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _endpoint(
    tensor: torch.Tensor,
    *,
    stage_id: str,
    block_id: int,
    kv_kind: str = "key",
    byte_offset: int = 0,
) -> PageEndpoint:
    return PageEndpoint(
        stage_id=stage_id,
        replica_id=0,
        tp_rank=0,
        cache_role="primary",
        layer_name="layer0",
        kv_kind=kv_kind,
        block_id=block_id,
        byte_offset=byte_offset,
        num_bytes=tensor.numel() * tensor.element_size(),
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        device_type=tensor.device.type,
    )


def _binding(*, generation: int = 1, block_id: int = 5) -> DiffusionPageBinding:
    return DiffusionPageBinding(
        request_id="req",
        allocation_generation=generation,
        sequences=(),
        page_states={block_id: PageState.INSTALLING_LOCAL},
        externally_required=frozenset({block_id}),
    )


def _direct_manager(
    *,
    event: FakeEvent | None = None,
    generation: int = 1,
    clock: FakeClock | None = None,
    copy_fn=None,
    metrics: DiffusionPageMetrics | None = None,
):
    event = event or FakeEvent()
    clock = clock or FakeClock()
    source = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    destination = torch.zeros_like(source)
    source_endpoint = _endpoint(source, stage_id="source", block_id=3)
    destination_endpoint = _endpoint(destination, stage_id="target", block_id=5)
    plan = PageTransferPlan(
        session_id="session",
        request_id="req",
        allocation_generation=generation,
        route_epoch=2,
        op_id=9,
        source_stage="source",
        target_stage="target",
        source_tp_rank=0,
        target_tp_rank=0,
        pages=(PageCopy(source_endpoint, destination_endpoint),),
        deadline_monotonic_s=10.0,
    )
    binding = _binding(generation=generation)
    manager = PageTransferSessionManager(
        registry=None,
        timeout_s=5.0,
        clock=clock,
        event_factory=lambda: event,
        copy_fn=copy_fn,
        metrics=metrics,
    )
    manager.register_binding(binding)
    manager.register_endpoint(source_endpoint, source)
    manager.register_endpoint(destination_endpoint, destination)
    return manager, binding, plan, source, destination


def test_direct_copy_commits_only_after_event_completion() -> None:
    event = FakeEvent(complete=False)
    manager, binding, plan, source, destination = _direct_manager(event=event)

    handle = manager.send_pages(plan, manager.prepare_receive(plan))

    assert manager.poll_completion(handle) is None
    assert not binding.is_compute_ready
    assert torch.equal(destination, source)
    event.complete = True
    result = manager.poll_completion(handle)
    assert result is not None
    assert result.status == "completed"
    assert binding.is_compute_ready


def test_completed_transfer_reports_lifecycle_metrics_and_terminal_snapshot() -> None:
    event = FakeEvent(complete=False)
    clock = FakeClock(now=1.0)
    metrics = DiffusionPageMetrics()
    manager, _, plan, _, _ = _direct_manager(
        event=event,
        generation=3,
        clock=clock,
        metrics=metrics,
    )

    reservation = manager.prepare_receive(plan)
    expected_bytes = plan.pages[0].destination.num_bytes
    assert metrics.staging_bytes == expected_bytes
    assert metrics.staging_bytes_high_water == expected_bytes

    clock.now = 2.5
    handle = manager.send_pages(plan, reservation)
    assert metrics.local_kv_wait_s == pytest.approx(1.5)
    assert metrics.in_flight_bytes == expected_bytes
    assert metrics.in_flight_bytes_high_water == expected_bytes

    clock.now = 4.0
    event.complete = True
    result = manager.poll_completion(handle)

    assert result is not None
    assert result.status == "completed"
    assert metrics.stable_pages_imported == 1
    assert metrics.stable_pages_committed == 1
    assert metrics.transferred_bytes == expected_bytes
    assert metrics.local_install_latency_s == pytest.approx(1.5)
    assert metrics.staging_bytes == 0
    assert metrics.in_flight_bytes == 0
    assert metrics.terminal_snapshots[-1] == {
        "request_id": "req",
        "allocation_generation": 3,
        "session_id": "session",
        "tp_rank": 0,
        "page_count": 1,
        "transferred_bytes": expected_bytes,
        "terminal_status": "completed",
    }


def test_transfer_reports_stale_duplicate_cancellation_and_timeout_counts() -> None:
    stale_metrics = DiffusionPageMetrics()
    stale_manager, _, stale_plan, _, _ = _direct_manager(metrics=stale_metrics)
    stale_manager.open_session(stale_plan)
    stale_manager.unregister_binding("req", allocation_generation=1)
    assert stale_manager.complete_for_test(stale_plan.identity).status == "stale"
    assert stale_metrics.stale_completions == 1

    duplicate_metrics = DiffusionPageMetrics()
    duplicate_manager, _, duplicate_plan, _, _ = _direct_manager(metrics=duplicate_metrics)
    duplicate_handle = duplicate_manager.send_pages(
        duplicate_plan,
        duplicate_manager.prepare_receive(duplicate_plan),
    )
    assert duplicate_manager.poll_completion(duplicate_handle) is not None
    assert duplicate_manager.poll_completion(duplicate_handle) is not None
    assert duplicate_metrics.duplicate_completions == 1

    cancellation_metrics = DiffusionPageMetrics()
    cancellation_manager, _, cancellation_plan, _, _ = _direct_manager(metrics=cancellation_metrics)
    cancellation_manager.cancel(cancellation_manager.prepare_receive(cancellation_plan))
    assert cancellation_metrics.cancellations == 1
    assert cancellation_metrics.staging_bytes == 0
    assert cancellation_metrics.terminal_snapshots[-1]["page_count"] == 1
    assert cancellation_metrics.terminal_snapshots[-1]["transferred_bytes"] == 0

    timeout_metrics = DiffusionPageMetrics()
    timeout_clock = FakeClock(now=1.0)
    timeout_manager, _, timeout_plan, _, _ = _direct_manager(
        event=FakeEvent(complete=False),
        clock=timeout_clock,
        metrics=timeout_metrics,
    )
    timeout_handle = timeout_manager.send_pages(
        timeout_plan,
        timeout_manager.prepare_receive(timeout_plan),
    )
    timeout_clock.now = 11.0
    assert timeout_manager.poll_completion(timeout_handle) is not None
    assert timeout_metrics.timeouts == 1
    assert timeout_metrics.staging_bytes == 0
    assert timeout_metrics.in_flight_bytes == 0


def test_reference_gather_metrics_are_recorded_without_tensor_reads() -> None:
    metrics = DiffusionPageMetrics()
    manager, _, _, _, _ = _direct_manager(metrics=metrics)

    manager.record_reference_gather(num_bytes=8192, latency_s=0.25)

    assert metrics.reference_gather_bytes == 8192
    assert metrics.reference_gather_latency_s == pytest.approx(0.25)


def test_stale_generation_never_commits_new_binding() -> None:
    manager, old_binding, plan, _, _ = _direct_manager(generation=1)
    manager.open_session(plan)
    manager.close_session(plan.session_id)
    new_binding = _binding(generation=2)
    manager.register_binding(new_binding)

    result = manager.complete_for_test(plan.identity)

    assert result.status == "stale"
    assert not old_binding.is_compute_ready
    assert not new_binding.is_compute_ready


def test_duplicate_terminal_completion_is_idempotent() -> None:
    manager, _, plan, _, _ = _direct_manager()
    handle = manager.send_pages(plan, manager.prepare_receive(plan))

    first = manager.poll_completion(handle)
    second = manager.poll_completion(handle)

    assert first == second


@pytest.mark.parametrize("state", ["allocated", "target_ready", "transferring"])
def test_cancellation_is_terminal_in_every_nonterminal_state(state: str) -> None:
    event = FakeEvent(complete=False)
    manager, binding, plan, _, _ = _direct_manager(event=event)
    if state == "allocated":
        handle = manager.open_session(plan)
    elif state == "target_ready":
        handle = manager.prepare_receive(plan)
    else:
        handle = manager.send_pages(plan, manager.prepare_receive(plan))

    result = manager.cancel(handle)

    assert result.status == "cancelled"
    assert manager.poll_completion(handle) == result
    assert not binding.is_compute_ready


def test_monotonic_timeout_never_commits_pages() -> None:
    event = FakeEvent(complete=False)
    clock = FakeClock(now=1.0)
    manager, binding, plan, _, _ = _direct_manager(event=event, clock=clock)
    handle = manager.send_pages(plan, manager.prepare_receive(plan))
    clock.now = 11.0

    result = manager.poll_completion(handle)

    assert result is not None
    assert result.status == "timed_out"
    assert not binding.is_compute_ready


@pytest.mark.parametrize(
    "destination_transform",
    [
        lambda tensor: torch.zeros(tensor.shape, dtype=torch.float64),
        lambda tensor: torch.zeros((4, 4), dtype=tensor.dtype),
        lambda tensor: torch.zeros_like(tensor).transpose(0, 1),
    ],
    ids=["dtype", "shape", "stride"],
)
def test_copy_rejects_endpoint_geometry_mismatch(destination_transform) -> None:
    manager, _, plan, source, _ = _direct_manager()
    destination = destination_transform(source)
    destination_endpoint = _endpoint(destination, stage_id="target", block_id=5)
    mismatched_plan = replace(
        plan,
        pages=(PageCopy(plan.pages[0].source, destination_endpoint),),
    )
    manager.register_endpoint(destination_endpoint, destination)

    with pytest.raises(ValueError, match="geometry mismatch"):
        manager.prepare_receive(mismatched_plan)


def test_registration_rejects_endpoint_byte_count_mismatch() -> None:
    manager, _, plan, _, destination = _direct_manager()
    endpoint = replace(plan.pages[0].destination, num_bytes=plan.pages[0].destination.num_bytes + 4)

    with pytest.raises(ValueError, match="byte count"):
        manager.register_endpoint(endpoint, destination)


def test_copy_rejects_source_destination_overlap() -> None:
    manager, _, plan, source, _ = _direct_manager()
    destination_endpoint = _endpoint(source, stage_id="target", block_id=5)
    overlapping_plan = replace(
        plan,
        pages=(PageCopy(plan.pages[0].source, destination_endpoint),),
    )
    manager.unregister_endpoint(plan.pages[0].destination)
    manager.register_endpoint(destination_endpoint, source)

    with pytest.raises(ValueError, match="overlap"):
        manager.prepare_receive(overlapping_plan)


def test_prepare_receive_rejects_missing_destination_registration() -> None:
    manager, _, plan, _, _ = _direct_manager()
    missing = replace(plan.pages[0].destination, block_id=7)
    missing_plan = replace(plan, pages=(PageCopy(plan.pages[0].source, missing),))

    with pytest.raises(KeyError, match="destination endpoint"):
        manager.prepare_receive(missing_plan)


def test_copy_failure_is_terminal_and_leaves_page_uncommitted() -> None:
    def fail_copy(destination: torch.Tensor, source: torch.Tensor) -> None:
        raise RuntimeError("injected copy failure")

    manager, binding, plan, _, _ = _direct_manager(copy_fn=fail_copy)
    handle = manager.send_pages(plan, manager.prepare_receive(plan))

    result = manager.poll_completion(handle)

    assert result is not None
    assert result.status == "failed"
    assert "injected copy failure" in result.message
    assert not binding.is_compute_ready
    assert manager.state(handle) is TransferState.FAILED
