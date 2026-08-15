# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
)
from vllm.v1.worker.block_table import BlockTable

from vllm_omni.diffusion.data import DiffusionPageMetrics
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.diffusion_kv.page import (
    DiffusionPageBinding,
    DiffusionSequenceBinding,
    PageState,
    build_slot_mapping,
)


@dataclass(frozen=True, slots=True)
class LayerPageStorage:
    tensor: torch.Tensor
    layer_names: tuple[str, ...]
    group_index: int
    spec: KVCacheSpec


@dataclass(slots=True)
class _ActiveBinding:
    binding: DiffusionPageBinding
    row_indices: tuple[int, ...]
    owned_pages: tuple[tuple[int, int], ...]


def _make_native_block_table(
    *,
    block_size: int,
    max_num_reqs: int,
    max_num_blocks_per_req: int,
    device: torch.device,
) -> BlockTable:
    return BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=max_num_reqs * max_num_blocks_per_req * block_size,
        pin_memory=False,
        device=device,
        kernel_block_size=block_size,
        cp_kv_cache_interleave_size=1,
    )


def _full_attention_cache_shape(
    spec: KVCacheSpec,
    *,
    num_blocks: int,
) -> tuple[int, ...]:
    if not isinstance(spec, FullAttentionSpec):
        raise ValueError(
            f"paged diffusion Worker storage currently supports only FullAttentionSpec, got {type(spec).__name__}"
        )
    if spec.head_size_v != spec.head_size:
        raise ValueError(
            "paged diffusion Worker storage requires equal key/value head sizes, "
            f"got key={spec.head_size}, value={spec.head_size_v}"
        )
    if spec.page_size_padded is not None:
        raise ValueError("paged diffusion Worker storage does not support padded cache pages")
    if spec.real_page_size_bytes != spec.page_size_bytes:
        raise ValueError("paged diffusion Worker storage does not support auxiliary per-page data")
    return (
        2,
        num_blocks,
        spec.block_size,
        spec.num_kv_heads,
        spec.head_size,
    )


class WorkerPageRegistry:
    def __init__(
        self,
        *,
        kv_cache_config: KVCacheConfig,
        layer_specs: Mapping[str, KVCacheSpec],
        device: torch.device,
        max_num_reqs: int,
        max_model_len: int,
        metrics: DiffusionPageMetrics | None = None,
    ) -> None:
        if kv_cache_config.num_blocks <= 0:
            raise ValueError("kv_cache_config.num_blocks must be positive")
        if max_num_reqs <= 0:
            raise ValueError("max_num_reqs must be positive")
        if max_model_len <= 0:
            raise ValueError("max_model_len must be positive")

        self.kv_cache_config = kv_cache_config
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self._layer_specs = dict(layer_specs)
        self._layer_to_group = self._validate_layer_specs()
        self._storages, self._layer_storages = self._allocate_storages()
        self._block_tables = self._allocate_block_tables()
        self._active_bindings: dict[str, _ActiveBinding] = {}
        self._page_owners: dict[tuple[int, int], tuple[str, int, int]] = {}
        self._free_rows = list(reversed(range(max_num_reqs)))
        self.metrics = metrics or DiffusionPageMetrics()
        self._update_page_pool_metrics()

    def _validate_layer_specs(self) -> dict[str, int]:
        configured_layers: set[str] = set()
        layer_to_group: dict[str, int] = {}
        for group_index, group in enumerate(self.kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                if layer_name in configured_layers:
                    raise ValueError(f"configured layer {layer_name!r} belongs to multiple cache groups")
                configured_layers.add(layer_name)
                layer_to_group[layer_name] = group_index

        discovered_layers = set(self._layer_specs)
        if configured_layers != discovered_layers:
            raise ValueError(
                "configured layer/spec mismatch: "
                f"configured={sorted(configured_layers)}, discovered={sorted(discovered_layers)}"
            )

        for group in self.kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if self._layer_specs[layer_name] != group.kv_cache_spec:
                    raise ValueError(
                        "configured layer/spec mismatch for "
                        f"{layer_name!r}: configured={group.kv_cache_spec!r}, "
                        f"discovered={self._layer_specs[layer_name]!r}"
                    )
        return layer_to_group

    def _allocate_storages(
        self,
    ) -> tuple[tuple[LayerPageStorage, ...], dict[str, LayerPageStorage]]:
        configured_layers = set(self._layer_to_group)
        mapped_layers: set[str] = set()
        storages: list[LayerPageStorage] = []
        layer_storages: dict[str, LayerPageStorage] = {}

        for tensor_config in self.kv_cache_config.kv_cache_tensors:
            layer_names = tuple(tensor_config.shared_by)
            if not layer_names:
                raise ValueError("KVCacheTensor.shared_by must contain at least one layer")
            unknown = set(layer_names).difference(configured_layers)
            if unknown:
                raise ValueError(f"KVCacheTensor.shared_by contains unknown layers: {sorted(unknown)}")
            duplicates = set(layer_names).intersection(mapped_layers)
            if duplicates:
                raise ValueError(f"layers map to multiple physical KV tensors: {sorted(duplicates)}")
            group_indices = {self._layer_to_group[layer_name] for layer_name in layer_names}
            if len(group_indices) != 1:
                raise ValueError("one physical KV tensor cannot span multiple cache groups")
            if tensor_config.offset != 0 or tensor_config.block_stride != 0:
                raise ValueError("paged diffusion Worker storage does not support packed KVCacheTensor layouts")

            group_index = next(iter(group_indices))
            spec = self.kv_cache_config.kv_cache_groups[group_index].kv_cache_spec
            shape = _full_attention_cache_shape(
                spec,
                num_blocks=self.kv_cache_config.num_blocks,
            )
            tensor = torch.empty(shape, dtype=spec.dtype, device=self.device)
            if tensor.numel() * tensor.element_size() != tensor_config.size:
                raise ValueError(
                    "configured KV tensor size does not match derived physical shape: "
                    f"configured={tensor_config.size}, derived={tensor.numel() * tensor.element_size()}"
                )
            storage = LayerPageStorage(
                tensor=tensor,
                layer_names=layer_names,
                group_index=group_index,
                spec=spec,
            )
            storages.append(storage)
            for layer_name in layer_names:
                layer_storages[layer_name] = storage
            mapped_layers.update(layer_names)

        missing = configured_layers.difference(mapped_layers)
        if missing:
            raise ValueError(f"configured layers lack physical KV tensors: {sorted(missing)}")
        return tuple(storages), layer_storages

    def _allocate_block_tables(self) -> tuple[BlockTable, ...]:
        tables: list[BlockTable] = []
        for group in self.kv_cache_config.kv_cache_groups:
            block_size = group.kv_cache_spec.block_size
            max_num_blocks_per_req = (self.max_model_len + block_size - 1) // block_size
            tables.append(
                _make_native_block_table(
                    block_size=block_size,
                    max_num_reqs=self.max_num_reqs,
                    max_num_blocks_per_req=max_num_blocks_per_req,
                    device=self.device,
                )
            )
        return tuple(tables)

    def get_layer_cache(self, layer_name: str) -> torch.Tensor:
        try:
            return self._layer_storages[layer_name].tensor
        except KeyError as exc:
            raise KeyError(f"unknown paged diffusion KV layer {layer_name!r}") from exc

    def bind_request(self, metadata: DiffusionKVMetadata) -> DiffusionPageBinding:
        if metadata.request_id in self._active_bindings:
            active = self._active_bindings[metadata.request_id]
            raise ValueError(
                f"request {metadata.request_id!r} already has an active binding "
                f"for allocation generation {active.binding.allocation_generation}"
            )
        if metadata.allocation_generation <= 0:
            raise ValueError("allocation_generation must be positive")
        if not metadata.sequences:
            raise ValueError("Diffusion KV metadata must contain at least one sequence")
        if len(self._free_rows) < len(metadata.sequences):
            raise ValueError(
                "Worker page registry has insufficient free request rows: "
                f"required={len(metadata.sequences)}, available={len(self._free_rows)}"
            )

        expected_sequence_ids = list(range(len(metadata.sequences)))
        sequence_ids = [sequence.sequence_id for sequence in metadata.sequences]
        if sequence_ids != expected_sequence_ids:
            raise ValueError(
                "metadata sequences must use contiguous sequence IDs: "
                f"expected={expected_sequence_ids}, got={sequence_ids}"
            )

        owned_pages: list[tuple[int, int]] = []
        seen_pages: set[tuple[int, int]] = set()
        sequence_bindings: list[DiffusionSequenceBinding] = []
        primary_states: dict[int, PageState] = {}
        externally_required: set[int] = set()

        for sequence in metadata.sequences:
            self._validate_sequence_geometry(sequence)
            if len(sequence.block_ids) != len(self.kv_cache_config.kv_cache_groups):
                raise ValueError(
                    "metadata cache group count does not match Worker configuration: "
                    f"metadata={len(sequence.block_ids)}, "
                    f"configured={len(self.kv_cache_config.kv_cache_groups)}"
                )

            for group_index, (group, group_block_ids) in enumerate(
                zip(self.kv_cache_config.kv_cache_groups, sequence.block_ids)
            ):
                expected_blocks = (sequence.seq_len + group.kv_cache_spec.block_size - 1) // (
                    group.kv_cache_spec.block_size
                )
                if len(group_block_ids) != expected_blocks:
                    raise ValueError(
                        "metadata sequence geometry does not match allocated block count: "
                        f"sequence_id={sequence.sequence_id}, group={group_index}, "
                        f"expected={expected_blocks}, got={len(group_block_ids)}"
                    )
                for block_id in group_block_ids:
                    page = (group_index, block_id)
                    if block_id < 0 or block_id >= self.kv_cache_config.num_blocks:
                        raise ValueError(
                            f"block ID {block_id} is out of range for {self.kv_cache_config.num_blocks} blocks"
                        )
                    if page in seen_pages:
                        raise ValueError(f"duplicate block ID {block_id} in cache group {group_index} within one rank")
                    if owner := self._page_owners.get(page):
                        raise ValueError(
                            f"block ID {block_id} in cache group {group_index} is already owned by "
                            f"request={owner[0]!r}, generation={owner[1]}, sequence={owner[2]}"
                        )
                    seen_pages.add(page)
                    owned_pages.append(page)

            stable, dynamic = sequence.page_ranges
            slot_mapping = build_slot_mapping(
                block_ids=tuple(sequence.block_ids[0]),
                block_size=self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size,
                token_start=0,
                token_count=sequence.seq_len,
                device=self.device,
            )
            sequence_bindings.append(
                DiffusionSequenceBinding(
                    sequence_id=sequence.sequence_id,
                    seq_len=sequence.seq_len,
                    stable=stable,
                    dynamic=dynamic,
                    slot_mapping=slot_mapping,
                )
            )

            block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
            imported_block_count = sequence.imported_prefix_token_count // block_size
            imported_ids = set(stable.block_ids[:imported_block_count])
            externally_required.update(imported_ids)
            for block_id in sequence.block_ids[0]:
                state = PageState.INSTALLING_LOCAL if block_id in imported_ids else PageState.RESERVED
                existing = primary_states.setdefault(block_id, state)
                if existing is not state:
                    raise ValueError(f"block ID {block_id} has conflicting initial page states")

        row_indices = tuple(self._free_rows.pop() for _ in metadata.sequences)
        owner_records = tuple(
            (
                page,
                (
                    metadata.request_id,
                    metadata.allocation_generation,
                    sequence.sequence_id,
                ),
            )
            for sequence in metadata.sequences
            for page in (
                (group_index, block_id)
                for group_index, group_block_ids in enumerate(sequence.block_ids)
                for block_id in group_block_ids
            )
        )
        try:
            for page, owner in owner_records:
                self._page_owners[page] = owner
            for row_index, sequence in zip(row_indices, metadata.sequences):
                for group_index, table in enumerate(self._block_tables):
                    table.add_row(list(sequence.block_ids[group_index]), row_index)
            self._zero_owned_pages(owned_pages)
            binding = DiffusionPageBinding(
                request_id=metadata.request_id,
                allocation_generation=metadata.allocation_generation,
                sequences=tuple(sequence_bindings),
                page_states=primary_states,
                externally_required=frozenset(externally_required),
            )
            self._active_bindings[metadata.request_id] = _ActiveBinding(
                binding=binding,
                row_indices=row_indices,
                owned_pages=tuple(owned_pages),
            )
            self.metrics.stable_pages_requested += len(externally_required)
            self._update_page_pool_metrics()
            return binding
        except Exception:
            for row_index in row_indices:
                for table in self._block_tables:
                    table.clear_row(row_index)
            for page, _ in owner_records:
                self._page_owners.pop(page, None)
            self._free_rows.extend(row_indices)
            raise

    def _validate_sequence_geometry(self, sequence: DiffusionKVSequenceMetadata) -> None:
        if sequence.seq_len <= 0 or sequence.prefix_len < 0 or sequence.target_len <= 0:
            raise ValueError("metadata sequence geometry contains non-positive lengths")
        if sequence.prefix_len + sequence.target_len > sequence.seq_len:
            raise ValueError("metadata sequence geometry exceeds seq_len")
        if len(sequence.page_ranges) != 2:
            raise ValueError("metadata sequence geometry requires stable and dynamic page ranges")
        stable, dynamic = sequence.page_ranges
        if (
            stable.cache_role != "primary"
            or stable.token_start != 0
            or stable.token_count != sequence.prefix_len
            or stable.mutable
            or dynamic.cache_role != "primary"
            or dynamic.token_start != sequence.prefix_len
            or dynamic.token_count != sequence.target_len
            or not dynamic.mutable
        ):
            raise ValueError("metadata sequence geometry does not match stable/dynamic page ranges")

        block_size = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size
        stable_block_count = (sequence.prefix_len + block_size - 1) // block_size
        dynamic_end = sequence.prefix_len + sequence.target_len
        dynamic_block_count = (dynamic_end + block_size - 1) // block_size
        primary_ids = tuple(sequence.block_ids[0])
        if (
            stable.block_ids != primary_ids[:stable_block_count]
            or dynamic.block_ids != primary_ids[sequence.prefix_len // block_size : dynamic_block_count]
        ):
            raise ValueError("metadata sequence geometry does not match Scheduler block IDs")
        if sequence.cacheable_prefix_block_count != sequence.prefix_len // block_size:
            raise ValueError("metadata sequence geometry has an invalid cacheable prefix boundary")
        if (
            sequence.imported_prefix_token_count < 0
            or sequence.imported_prefix_token_count > sequence.prefix_len
            or sequence.imported_prefix_token_count % block_size
        ):
            raise ValueError("metadata sequence geometry has an invalid imported prefix boundary")

    def _zero_owned_pages(self, owned_pages: list[tuple[int, int]]) -> None:
        for storage in self._storages:
            block_ids = [block_id for group_index, block_id in owned_pages if group_index == storage.group_index]
            if block_ids:
                block_index = torch.tensor(block_ids, dtype=torch.long, device=storage.tensor.device)
                storage.tensor.index_fill_(1, block_index, 0)

    def release_request(self, request_id: str, allocation_generation: int) -> None:
        try:
            active = self._active_bindings[request_id]
        except KeyError as exc:
            raise ValueError(f"stale release for inactive request {request_id!r}") from exc
        if active.binding.allocation_generation != allocation_generation:
            raise ValueError(
                f"stale release for request {request_id!r}: "
                f"active_generation={active.binding.allocation_generation}, "
                f"release_generation={allocation_generation}"
            )

        for block_id in tuple(active.binding.page_states):
            active.binding.transition_page(block_id, PageState.RELEASING)
            active.binding.transition_page(block_id, PageState.FREE)
        for row_index in active.row_indices:
            for table in self._block_tables:
                table.clear_row(row_index)
        for page in active.owned_pages:
            self._page_owners.pop(page, None)
        self._active_bindings.pop(request_id)
        self._free_rows.extend(active.row_indices)
        self._update_page_pool_metrics()

    def _update_page_pool_metrics(self) -> None:
        self.metrics.update_page_pool(
            pages_in_use=len(self._page_owners),
            total_pages=self.kv_cache_config.num_blocks * len(self.kv_cache_config.kv_cache_groups),
        )
