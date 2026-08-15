# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from vllm_omni.diffusion.data import DiffusionPageMetrics
from vllm_omni.diffusion.diffusion_kv.page import (
    DiffusionPageBinding,
    DiffusionSequenceBinding,
    PageState,
)


@dataclass(slots=True)
class HunyuanPagedKVBatch:
    request_id: str
    allocation_generation: int
    block_size: int
    sequences: tuple[DiffusionSequenceBinding, ...]
    query_lens: tuple[int, ...]
    seq_lens: tuple[int, ...]
    logical_positions: tuple[torch.Tensor, ...]
    slot_mappings: tuple[torch.Tensor, ...]
    stable_write_masks: tuple[torch.Tensor, ...]
    dynamic_write_masks: tuple[torch.Tensor, ...]
    cacheable_block_ids: tuple[tuple[int, ...], ...]
    imported_prefix_token_counts: tuple[int, ...]
    _binding: DiffusionPageBinding = field(repr=False, compare=False)
    gather_count: int = 0
    gathered_bytes: int = 0
    gather_latency_s: float = 0.0

    @property
    def batch_size(self) -> int:
        return len(self.sequences)

    @property
    def query_len(self) -> int:
        return self.query_lens[0]

    @property
    def seq_len(self) -> int:
        return self.seq_lens[0]


def _imported_prefix_token_count(
    binding: DiffusionPageBinding,
    sequence: DiffusionSequenceBinding,
    *,
    block_size: int,
) -> int:
    imported_block_count = 0
    seen_local_page = False
    for block_id in sequence.stable.block_ids:
        is_imported = block_id in binding.externally_required
        if is_imported and seen_local_page:
            raise ValueError(
                "imported stable pages must form a leading prefix: "
                f"request={binding.request_id!r}, sequence={sequence.sequence_id}, block={block_id}"
            )
        if not is_imported:
            seen_local_page = True
            continue
        state = binding.page_states[block_id]
        if state is not PageState.COMMITTED:
            raise ValueError(f"imported stable page {block_id} is not committed for request {binding.request_id!r}")
        imported_block_count += 1

    imported_tokens = imported_block_count * block_size
    if imported_tokens > sequence.stable.token_count:
        raise ValueError(
            "imported stable pages exceed the stable prefix: "
            f"request={binding.request_id!r}, sequence={sequence.sequence_id}"
        )
    return imported_tokens


def build_hunyuan_paged_kv_batch(
    binding: DiffusionPageBinding,
    *,
    query_lens: list[int],
    seq_lens: list[int],
    block_size: int,
    position_ids: torch.Tensor | None = None,
) -> HunyuanPagedKVBatch:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    row_count = len(binding.sequences)
    if len(query_lens) != row_count or len(seq_lens) != row_count:
        raise ValueError(
            "Hunyuan paged KV row count mismatch: "
            f"bindings={row_count}, query_lens={len(query_lens)}, seq_lens={len(seq_lens)}"
        )
    if row_count == 0:
        raise ValueError("Hunyuan paged KV requires at least one sequence row")
    if any(query_len <= 0 for query_len in query_lens):
        raise ValueError("Hunyuan paged KV query lengths must be positive")
    if len(set(query_lens)) != 1:
        raise ValueError(f"Hunyuan paged KV requires a uniform query length, got {query_lens}")
    if len(set(seq_lens)) != 1:
        raise ValueError(f"Hunyuan paged KV sequence length mismatch: rows must be uniform, got {seq_lens}")

    logical_positions: list[torch.Tensor] = []
    slot_mappings: list[torch.Tensor] = []
    stable_write_masks: list[torch.Tensor] = []
    dynamic_write_masks: list[torch.Tensor] = []
    cacheable_block_ids: list[tuple[int, ...]] = []
    imported_prefix_token_counts: list[int] = []

    for row_index, (sequence, query_len, seq_len) in enumerate(zip(binding.sequences, query_lens, seq_lens)):
        if sequence.sequence_id != row_index:
            raise ValueError(
                "Hunyuan paged KV row order must match contiguous sequence IDs: "
                f"row={row_index}, sequence_id={sequence.sequence_id}"
            )
        if seq_len != sequence.seq_len:
            raise ValueError(
                "Hunyuan paged KV sequence length mismatch: "
                f"row={row_index}, binding={sequence.seq_len}, runtime={seq_len}"
            )
        if query_len > seq_len:
            raise ValueError(f"Hunyuan paged KV query length {query_len} exceeds sequence length {seq_len}")

        device = sequence.slot_mapping.device
        row_positions = torch.arange(
            seq_len - query_len,
            seq_len,
            dtype=torch.long,
            device=device,
        )
        if position_ids is not None:
            if position_ids.shape != (row_count, query_lens[0]):
                raise ValueError(
                    "Hunyuan paged KV position ID shape mismatch: "
                    f"expected={(row_count, query_lens[0])}, got={tuple(position_ids.shape)}"
                )
            runtime_positions = position_ids[row_index].to(device=device, dtype=torch.long)
            if not torch.equal(runtime_positions, row_positions):
                raise ValueError(
                    "Hunyuan paged KV position IDs do not match dense logical token order: "
                    f"row={row_index}, expected={row_positions.tolist()}, "
                    f"got={runtime_positions.tolist()}"
                )

        imported_tokens = _imported_prefix_token_count(
            binding,
            sequence,
            block_size=block_size,
        )
        stable_end = sequence.stable.token_start + sequence.stable.token_count
        stable_mask = (row_positions >= imported_tokens) & (row_positions < stable_end)
        dynamic_mask = row_positions >= sequence.dynamic.token_start
        dynamic_blocks = set(sequence.dynamic.block_ids)

        logical_positions.append(row_positions)
        slot_mappings.append(sequence.slot_mapping.index_select(0, row_positions))
        stable_write_masks.append(stable_mask)
        dynamic_write_masks.append(dynamic_mask)
        cacheable_block_ids.append(
            tuple(block_id for block_id in sequence.stable.block_ids if block_id not in dynamic_blocks)
        )
        imported_prefix_token_counts.append(imported_tokens)

    return HunyuanPagedKVBatch(
        request_id=binding.request_id,
        allocation_generation=binding.allocation_generation,
        block_size=block_size,
        sequences=binding.sequences,
        query_lens=tuple(query_lens),
        seq_lens=tuple(seq_lens),
        logical_positions=tuple(logical_positions),
        slot_mappings=tuple(slot_mappings),
        stable_write_masks=tuple(stable_write_masks),
        dynamic_write_masks=tuple(dynamic_write_masks),
        cacheable_block_ids=tuple(cacheable_block_ids),
        imported_prefix_token_counts=tuple(imported_prefix_token_counts),
        _binding=binding,
    )


def _validate_layer_cache(layer_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_cache.ndim != 5 or layer_cache.shape[0] != 2:
        raise ValueError("Hunyuan page storage must have shape [2, num_blocks, block_size, num_kv_heads, head_dim]")
    return layer_cache[0].flatten(0, 1), layer_cache[1].flatten(0, 1)


def write_hunyuan_kv(
    layer_cache: torch.Tensor,
    batch: HunyuanPagedKVBatch,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    stable_already_committed: bool = False,
) -> None:
    if layer_cache.ndim >= 3 and layer_cache.shape[2] != batch.block_size:
        raise ValueError(
            f"Hunyuan page storage block size mismatch: binding={batch.block_size}, storage={layer_cache.shape[2]}"
        )
    flat_key_pages, flat_value_pages = _validate_layer_cache(layer_cache)
    if key.shape != value.shape:
        raise ValueError(f"Hunyuan key/value shape mismatch: key={tuple(key.shape)}, value={tuple(value.shape)}")
    if key.ndim != 4:
        raise ValueError("Hunyuan key/value rows must have shape [batch, query, kv_heads, head_dim]")
    expected_shape = (
        batch.batch_size,
        batch.query_len,
        flat_key_pages.shape[1],
        flat_key_pages.shape[2],
    )
    if tuple(key.shape) != expected_shape:
        raise ValueError(
            f"Hunyuan key/value geometry does not match page binding: expected={expected_shape}, got={tuple(key.shape)}"
        )
    if key.dtype != layer_cache.dtype or value.dtype != layer_cache.dtype:
        raise ValueError(
            "Hunyuan key/value dtype must match page storage: "
            f"cache={layer_cache.dtype}, key={key.dtype}, value={value.dtype}"
        )
    if key.device != layer_cache.device or value.device != layer_cache.device:
        raise ValueError(
            "Hunyuan key/value device must match page storage: "
            f"cache={layer_cache.device}, key={key.device}, value={value.device}"
        )

    for row_index in range(batch.batch_size):
        write_mask = batch.dynamic_write_masks[row_index]
        if not stable_already_committed:
            write_mask = write_mask | batch.stable_write_masks[row_index]
        if not bool(write_mask.any()):
            continue
        slots = batch.slot_mappings[row_index][write_mask]
        flat_key_pages.index_copy_(0, slots, key[row_index][write_mask])
        flat_value_pages.index_copy_(0, slots, value[row_index][write_mask])


def gather_hunyuan_kv_reference(
    layer_cache: torch.Tensor,
    batch: HunyuanPagedKVBatch,
    *,
    metrics: DiffusionPageMetrics | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer_cache.ndim >= 3 and layer_cache.shape[2] != batch.block_size:
        raise ValueError(
            f"Hunyuan page storage block size mismatch: binding={batch.block_size}, storage={layer_cache.shape[2]}"
        )
    flat_key_pages, flat_value_pages = _validate_layer_cache(layer_cache)
    started = time.perf_counter()
    gathered_keys = [flat_key_pages.index_select(0, sequence.slot_mapping) for sequence in batch.sequences]
    gathered_values = [flat_value_pages.index_select(0, sequence.slot_mapping) for sequence in batch.sequences]
    key = torch.stack(gathered_keys, dim=0)
    value = torch.stack(gathered_values, dim=0)
    gathered_bytes = key.numel() * key.element_size() + value.numel() * value.element_size()
    gather_latency_s = time.perf_counter() - started
    batch.gather_count += 1
    batch.gathered_bytes += gathered_bytes
    batch.gather_latency_s += gather_latency_s
    if metrics is not None:
        metrics.record_reference_gather(
            num_bytes=gathered_bytes,
            latency_s=gather_latency_s,
        )
    return key, value
