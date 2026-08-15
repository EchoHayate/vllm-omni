# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import replace
from multiprocessing import shared_memory as shm_pkg

import pytest
import torch

from vllm_omni.diffusion.diffusion_kv.page import DiffusionPageBinding, PageState
from vllm_omni.diffusion.diffusion_kv.transfer import (
    PageCopy,
    PageEndpoint,
    PageTransferPlan,
    PageTransferSessionManager,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)
from vllm_omni.distributed.omni_connectors.page_shm import SharedMemoryPageAdapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _endpoint(
    tensor: torch.Tensor,
    *,
    stage_id: str,
    block_id: int,
    kv_kind: str,
) -> PageEndpoint:
    return PageEndpoint(
        stage_id=stage_id,
        replica_id=0,
        tp_rank=0,
        cache_role="primary",
        layer_name="layer0",
        kv_kind=kv_kind,
        block_id=block_id,
        byte_offset=0,
        num_bytes=tensor.numel() * tensor.element_size(),
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        device_type=tensor.device.type,
    )


def _adapter():
    source_key = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    source_value = source_key + 100
    destination_key = torch.zeros_like(source_key)
    destination_value = torch.zeros_like(source_value)
    copies = tuple(
        PageCopy(
            _endpoint(source, stage_id="source", block_id=3, kv_kind=kv_kind),
            _endpoint(destination, stage_id="target", block_id=5, kv_kind=kv_kind),
        )
        for source, destination, kv_kind in (
            (source_key, destination_key, "key"),
            (source_value, destination_value, "value"),
        )
    )
    plan = PageTransferPlan(
        session_id="page-shm-session",
        request_id="req",
        allocation_generation=1,
        route_epoch=0,
        op_id=4,
        source_stage="source",
        target_stage="target",
        source_tp_rank=0,
        target_tp_rank=0,
        pages=copies,
        deadline_monotonic_s=1_000_000_000.0,
    )
    binding = DiffusionPageBinding(
        request_id="req",
        allocation_generation=1,
        sequences=(),
        page_states={5: PageState.INSTALLING_LOCAL},
        externally_required=frozenset({5}),
    )
    manager = PageTransferSessionManager(registry=None, timeout_s=30.0)
    manager.register_binding(binding)
    manager.register_endpoint(copies[0].destination, destination_key)
    manager.register_endpoint(copies[1].destination, destination_value)
    connector = SharedMemoryConnector({})
    adapter = SharedMemoryPageAdapter(
        connector=connector,
        transfer_manager=manager,
    )
    return (
        adapter,
        connector,
        manager,
        binding,
        plan,
        (source_key, source_value),
        (destination_key, destination_value),
    )


def _segment_exists(name: str) -> bool:
    try:
        segment = shm_pkg.SharedMemory(name=name)
    except FileNotFoundError:
        return False
    segment.close()
    return True


def test_pack_two_page_spans_and_copy_into_preallocated_pages(mocker) -> None:
    adapter, connector, _, binding, plan, sources, destinations = _adapter()
    deserialize = mocker.patch.object(
        connector,
        "deserialize_obj",
        side_effect=AssertionError("page payload must not deserialize model objects"),
    )
    try:
        registration, payload = adapter.put_pages(plan, sources)

        result = adapter.copy_into_pages(plan, registration)

        assert payload.page_count == 2
        assert result.status == "completed"
        assert binding.is_compute_ready
        assert torch.equal(destinations[0], sources[0])
        assert torch.equal(destinations[1], sources[1])
        deserialize.assert_not_called()
    finally:
        adapter.close()
        connector.close()


def test_payload_larger_than_reserved_span_is_rejected() -> None:
    adapter, connector, _, binding, plan, sources, _ = _adapter()
    smaller_pages = tuple(
        PageCopy(
            replace(page.source, num_bytes=page.source.num_bytes - 4),
            replace(page.destination, num_bytes=page.destination.num_bytes - 4),
        )
        for page in plan.pages
    )
    smaller_plan = replace(plan, pages=smaller_pages)
    try:
        with pytest.raises(ValueError, match="reserved"):
            adapter.put_pages(smaller_plan, sources)
        assert not binding.is_compute_ready
    finally:
        adapter.close()
        connector.close()


@pytest.mark.parametrize(
    "transform",
    [
        lambda tensor: tensor.to(torch.float64),
        lambda tensor: tensor.reshape(4, 4),
        lambda tensor: tensor.transpose(0, 1),
    ],
    ids=["dtype", "shape", "stride"],
)
def test_put_pages_rejects_tensor_geometry_mismatch(transform) -> None:
    adapter, connector, _, binding, plan, sources, _ = _adapter()
    bad_sources = (transform(sources[0]), sources[1])
    try:
        with pytest.raises(ValueError, match="geometry mismatch"):
            adapter.put_pages(plan, bad_sources)
        assert not binding.is_compute_ready
    finally:
        adapter.close()
        connector.close()


def test_duplicate_copy_into_pages_returns_same_terminal_result() -> None:
    adapter, connector, _, _, plan, sources, _ = _adapter()
    try:
        registration, _ = adapter.put_pages(plan, sources)

        first = adapter.copy_into_pages(plan, registration)
        second = adapter.copy_into_pages(plan, registration)

        assert first == second
    finally:
        adapter.close()
        connector.close()


def test_cancellation_unlinks_segment_and_keeps_pages_uncommitted() -> None:
    adapter, connector, _, binding, plan, sources, _ = _adapter()
    try:
        registration, _ = adapter.put_pages(plan, sources)
        assert _segment_exists(registration.name)

        result = adapter.cancel(plan, registration)

        assert result.status == "cancelled"
        assert not _segment_exists(registration.name)
        assert not binding.is_compute_ready
    finally:
        adapter.close()
        connector.close()


def test_sender_failure_leaves_destination_pages_uncommitted() -> None:
    adapter, connector, _, binding, plan, sources, _ = _adapter()
    try:
        with pytest.raises(ValueError, match="tensor count"):
            adapter.put_pages(plan, sources[:1])
        assert not binding.is_compute_ready
    finally:
        adapter.close()
        connector.close()


def test_receiver_close_cleans_unconsumed_segment() -> None:
    adapter, connector, _, binding, plan, sources, _ = _adapter()
    registration, _ = adapter.put_pages(plan, sources)
    assert _segment_exists(registration.name)

    adapter.close()

    assert not _segment_exists(registration.name)
    assert not binding.is_compute_ready
    connector.close()
