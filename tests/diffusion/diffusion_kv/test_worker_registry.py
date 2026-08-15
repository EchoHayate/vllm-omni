# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)

import vllm_omni.diffusion.diffusion_kv.worker_registry as worker_registry_module
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.diffusion_kv.page import DiffusionPageRange, PageState
from vllm_omni.diffusion.diffusion_kv.worker_registry import WorkerPageRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCK_SIZE = 4


class _FakeBlockTable:
    def __init__(self) -> None:
        self.rows: dict[int, list[int]] = {}

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.rows[row_idx] = list(block_ids)

    def clear_row(self, row_idx: int) -> None:
        self.rows.pop(row_idx, None)


def _spec(
    *,
    block_size: int = BLOCK_SIZE,
    num_heads: int = 2,
    head_size: int = 8,
) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _config(
    *,
    num_blocks: int = 8,
    spec: FullAttentionSpec | None = None,
    layer_names: tuple[str, ...] = ("layer0",),
) -> KVCacheConfig:
    spec = spec or _spec()
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            KVCacheTensor(
                size=spec.page_size_bytes * num_blocks,
                shared_by=list(layer_names),
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=list(layer_names),
                kv_cache_spec=spec,
            )
        ],
    )


def _registry(
    monkeypatch,
    *,
    num_blocks: int = 8,
    block_size: int = BLOCK_SIZE,
    num_heads: int = 2,
    head_size: int = 8,
    layer_names: tuple[str, ...] = ("layer0",),
) -> WorkerPageRegistry:
    spec = _spec(
        block_size=block_size,
        num_heads=num_heads,
        head_size=head_size,
    )
    monkeypatch.setattr(
        worker_registry_module,
        "_make_native_block_table",
        lambda **_: _FakeBlockTable(),
    )
    return WorkerPageRegistry(
        kv_cache_config=_config(
            num_blocks=num_blocks,
            spec=spec,
            layer_names=layer_names,
        ),
        layer_specs={layer_name: spec for layer_name in layer_names},
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_model_len=32,
    )


def _metadata(
    *,
    request_id: str = "req",
    generation: int = 1,
    block_ids: list[int] | None = None,
    prefix_len: int = 4,
    target_len: int = 4,
    seq_len: int = 8,
    imported_prefix_token_count: int = 0,
) -> DiffusionKVMetadata:
    block_ids = block_ids or [3, 5]
    stable_block_count = (prefix_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    dynamic_end = prefix_len + target_len
    dynamic_block_count = (dynamic_end + BLOCK_SIZE - 1) // BLOCK_SIZE
    return DiffusionKVMetadata(
        request_id=request_id,
        allocation_generation=generation,
        sequences=(
            DiffusionKVSequenceMetadata(
                sequence_id=0,
                prefix_len=prefix_len,
                target_len=target_len,
                seq_len=seq_len,
                block_ids=(block_ids,),
                page_ranges=(
                    DiffusionPageRange(
                        "primary",
                        0,
                        prefix_len,
                        tuple(block_ids[:stable_block_count]),
                        mutable=False,
                    ),
                    DiffusionPageRange(
                        "primary",
                        prefix_len,
                        target_len,
                        tuple(block_ids[prefix_len // BLOCK_SIZE : dynamic_block_count]),
                        mutable=True,
                    ),
                ),
                cacheable_prefix_block_count=prefix_len // BLOCK_SIZE,
                imported_prefix_token_count=imported_prefix_token_count,
            ),
        ),
    )


def test_registry_allocates_native_shape_for_each_group(monkeypatch) -> None:
    registry = _registry(
        monkeypatch,
        num_blocks=8,
        block_size=4,
        num_heads=2,
        head_size=8,
    )

    cache = registry.get_layer_cache("layer0")

    assert cache.shape == (2, 8, 4, 2, 8)
    assert cache.dtype is torch.bfloat16


def test_registry_maps_shared_layers_to_one_physical_storage(monkeypatch) -> None:
    registry = _registry(monkeypatch, layer_names=("layer0", "layer1"))

    assert registry.get_layer_cache("layer0") is registry.get_layer_cache("layer1")


def test_registry_binds_scheduler_ids_without_allocating_new_ids(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)

    binding = registry.bind_request(_metadata(block_ids=[3, 5]))

    assert binding.sequences[0].stable.block_ids == (3,)
    assert binding.sequences[0].dynamic.block_ids == (5,)
    assert binding.sequences[0].slot_mapping.tolist() == list(range(12, 16)) + list(range(20, 24))


def test_registry_zeroes_only_pages_entering_reserved_state(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)
    cache = registry.get_layer_cache("layer0")
    cache.fill_(7)

    registry.bind_request(_metadata(block_ids=[3, 5]))

    assert torch.count_nonzero(cache[:, 3]) == 0
    assert torch.count_nonzero(cache[:, 5]) == 0
    assert torch.all(cache[:, 4] == 7)


def test_registry_supports_zero_length_stable_prefix(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)

    binding = registry.bind_request(
        _metadata(
            block_ids=[3],
            prefix_len=0,
            target_len=4,
            seq_len=4,
        )
    )

    assert binding.sequences[0].stable.block_ids == ()
    assert binding.sequences[0].dynamic.block_ids == (3,)
    assert binding.page_states == {3: PageState.RESERVED}


def test_registry_assigns_one_native_block_table_row_per_cfg_sequence(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)
    first = _metadata(request_id="cfg", block_ids=[1, 2]).sequences[0]
    second = _metadata(request_id="cfg", block_ids=[3, 4]).sequences[0]
    second.sequence_id = 1
    metadata = DiffusionKVMetadata(
        request_id="cfg",
        allocation_generation=1,
        sequences=(first, second),
    )

    registry.bind_request(metadata)

    assert registry._block_tables[0].rows == {
        0: [1, 2],
        1: [3, 4],
    }
    registry.release_request("cfg", allocation_generation=1)
    assert registry._block_tables[0].rows == {}


def test_registry_rejects_generation_reuse_until_release(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)
    registry.bind_request(_metadata(request_id="req", generation=1))

    with pytest.raises(ValueError, match="already has an active binding"):
        registry.bind_request(_metadata(request_id="req", generation=2))


def test_registry_rejects_configured_layer_spec_mismatch(monkeypatch) -> None:
    spec = _spec()
    monkeypatch.setattr(
        worker_registry_module,
        "_make_native_block_table",
        lambda **_: _FakeBlockTable(),
    )

    with pytest.raises(ValueError, match="layer/spec mismatch"):
        WorkerPageRegistry(
            kv_cache_config=_config(spec=spec, layer_names=("layer0",)),
            layer_specs={"other_layer": spec},
            device=torch.device("cpu"),
            max_num_reqs=4,
            max_model_len=32,
        )


def test_registry_rejects_out_of_range_block_id(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)

    with pytest.raises(ValueError, match="out of range"):
        registry.bind_request(_metadata(block_ids=[3, 8]))


def test_registry_rejects_duplicate_block_id_within_one_rank(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)

    with pytest.raises(ValueError, match="duplicate block ID"):
        registry.bind_request(_metadata(block_ids=[3, 3]))


def test_registry_rejects_overlapping_active_ownership(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)
    registry.bind_request(_metadata(request_id="first", block_ids=[3, 5]))

    with pytest.raises(ValueError, match="already owned"):
        registry.bind_request(_metadata(request_id="second", block_ids=[5, 6]))


def test_registry_rejects_sequence_geometry_mismatch(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)
    metadata = _metadata(prefix_len=4, target_len=4, seq_len=8)
    sequence = metadata.sequences[0]
    sequence.page_ranges = (
        sequence.page_ranges[0],
        DiffusionPageRange("primary", 5, 3, (5,), mutable=True),
    )

    with pytest.raises(ValueError, match="sequence geometry"):
        registry.bind_request(metadata)


def test_registry_rejects_stale_release(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)
    registry.bind_request(_metadata(request_id="req", generation=2))

    with pytest.raises(ValueError, match="stale release"):
        registry.release_request("req", allocation_generation=1)


def test_release_transitions_owned_pages_to_free(monkeypatch) -> None:
    registry = _registry(monkeypatch, num_blocks=8)
    binding = registry.bind_request(
        _metadata(
            request_id="req",
            generation=2,
            imported_prefix_token_count=4,
        )
    )
    assert binding.page_states == {
        3: PageState.INSTALLING_LOCAL,
        5: PageState.RESERVED,
    }

    registry.release_request("req", allocation_generation=2)

    assert binding.page_states == {
        3: PageState.FREE,
        5: PageState.FREE,
    }
    replacement = registry.bind_request(
        _metadata(
            request_id="replacement",
            generation=1,
            block_ids=[3, 5],
        )
    )
    assert replacement.page_states == {
        3: PageState.RESERVED,
        5: PageState.RESERVED,
    }
