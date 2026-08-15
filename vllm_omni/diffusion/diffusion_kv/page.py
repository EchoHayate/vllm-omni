# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class PageState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    INSTALLING_LOCAL = "installing_local"
    INSTALLING_REMOTE = "installing_remote"
    COMMITTED = "committed"
    RELEASING = "releasing"


_ALLOWED_PAGE_TRANSITIONS = {
    PageState.FREE: {PageState.RESERVED},
    PageState.RESERVED: {
        PageState.INSTALLING_LOCAL,
        PageState.INSTALLING_REMOTE,
        PageState.COMMITTED,
        PageState.RELEASING,
    },
    PageState.INSTALLING_LOCAL: {PageState.COMMITTED, PageState.RELEASING},
    PageState.INSTALLING_REMOTE: {PageState.COMMITTED, PageState.RELEASING},
    PageState.COMMITTED: {PageState.RELEASING},
    PageState.RELEASING: {PageState.FREE},
}


@dataclass(frozen=True, slots=True)
class DiffusionPageRange:
    cache_role: str
    token_start: int
    token_count: int
    block_ids: tuple[int, ...]
    mutable: bool

    def __post_init__(self) -> None:
        if not self.cache_role:
            raise ValueError("cache_role must be non-empty")
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if any(block_id < 0 for block_id in self.block_ids):
            raise ValueError("block_ids must be non-negative")


@dataclass(frozen=True, slots=True)
class DiffusionSequenceBinding:
    sequence_id: int
    seq_len: int
    stable: DiffusionPageRange
    dynamic: DiffusionPageRange
    slot_mapping: torch.Tensor

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stable.mutable:
            raise ValueError("stable range must be immutable")
        if not self.dynamic.mutable:
            raise ValueError("dynamic range must be mutable")
        if self.stable.token_start != 0:
            raise ValueError("stable range must start at token zero")
        if self.dynamic.token_start != self.stable.token_count:
            raise ValueError("dynamic range must immediately follow stable range")
        if self.dynamic.token_start + self.dynamic.token_count > self.seq_len:
            raise ValueError("stable and dynamic ranges exceed seq_len")
        if self.slot_mapping.ndim != 1 or self.slot_mapping.numel() != self.seq_len:
            raise ValueError("slot_mapping must contain one slot per logical token")


@dataclass(slots=True)
class DiffusionPageBinding:
    request_id: str
    allocation_generation: int
    sequences: tuple[DiffusionSequenceBinding, ...]
    page_states: dict[int, PageState] = field(default_factory=dict)
    externally_required: frozenset[int] = frozenset()
    locally_produced_pages: frozenset[int] = frozenset()
    local_write_event: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.allocation_generation <= 0:
            raise ValueError("allocation_generation must be positive")
        missing = self.externally_required.difference(self.page_states)
        if missing:
            raise ValueError(f"externally required pages lack state: {sorted(missing)}")

    def record_local_write_completion(
        self,
        block_ids: set[int],
        event: object,
    ) -> None:
        unknown = block_ids.difference(self.page_states)
        if unknown:
            raise ValueError(f"locally produced pages lack state: {sorted(unknown)}")
        imported = block_ids.intersection(self.externally_required)
        if imported:
            raise ValueError(f"externally required pages cannot be locally produced: {sorted(imported)}")
        self.locally_produced_pages = self.locally_produced_pages.union(block_ids)
        self.local_write_event = event

    @property
    def is_compute_ready(self) -> bool:
        return all(self.page_states[block_id] is PageState.COMMITTED for block_id in self.externally_required)

    def transition_page(self, block_id: int, target: PageState) -> None:
        current = self.page_states[block_id]
        if target not in _ALLOWED_PAGE_TRANSITIONS[current]:
            raise ValueError(f"invalid page transition {current.value} -> {target.value} for block {block_id}")
        self.page_states[block_id] = target


def build_slot_mapping(
    *,
    block_ids: tuple[int, ...],
    block_size: int,
    token_start: int,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if token_start < 0 or token_count < 0:
        raise ValueError("token_start and token_count must be non-negative")
    logical_positions = torch.arange(token_start, token_start + token_count, dtype=torch.long, device=device)
    logical_blocks = torch.div(logical_positions, block_size, rounding_mode="floor")
    if logical_blocks.numel() and int(logical_blocks.max().item()) >= len(block_ids):
        raise ValueError("block_ids do not cover the requested token range")
    block_tensor = torch.tensor(block_ids, dtype=torch.long, device=device)
    return block_tensor[logical_blocks] * block_size + logical_positions.remainder(block_size)
