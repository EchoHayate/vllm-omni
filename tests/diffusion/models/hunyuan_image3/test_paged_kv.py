# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.diffusion_kv.page import (
    DiffusionPageBinding,
    DiffusionPageRange,
    DiffusionSequenceBinding,
    PageState,
    build_slot_mapping,
)
from vllm_omni.diffusion.models.hunyuan_image3.paged_kv import (
    build_hunyuan_paged_kv_batch,
    gather_hunyuan_kv_reference,
    write_hunyuan_kv,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

BLOCK_SIZE = 4
NUM_BLOCKS = 12
NUM_KV_HEADS = 2
HEAD_DIM = 3


def _sequence(
    sequence_id: int,
    *,
    block_ids: tuple[int, ...],
    prefix_len: int,
    target_len: int,
    seq_len: int,
) -> DiffusionSequenceBinding:
    stable_block_count = (prefix_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    dynamic_end = prefix_len + target_len
    dynamic_block_count = (dynamic_end + BLOCK_SIZE - 1) // BLOCK_SIZE
    return DiffusionSequenceBinding(
        sequence_id=sequence_id,
        seq_len=seq_len,
        stable=DiffusionPageRange(
            "primary",
            0,
            prefix_len,
            block_ids[:stable_block_count],
            mutable=False,
        ),
        dynamic=DiffusionPageRange(
            "primary",
            prefix_len,
            target_len,
            block_ids[prefix_len // BLOCK_SIZE : dynamic_block_count],
            mutable=True,
        ),
        slot_mapping=build_slot_mapping(
            block_ids=block_ids,
            block_size=BLOCK_SIZE,
            token_start=0,
            token_count=seq_len,
            device=torch.device("cpu"),
        ),
    )


def _binding(
    *sequences: DiffusionSequenceBinding,
    page_states: dict[int, PageState] | None = None,
    externally_required: frozenset[int] = frozenset(),
) -> DiffusionPageBinding:
    return DiffusionPageBinding(
        request_id="req",
        allocation_generation=7,
        sequences=sequences,
        page_states=page_states or {},
        externally_required=externally_required,
    )


def _pages() -> torch.Tensor:
    return torch.zeros(
        2,
        NUM_BLOCKS,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )


def _kv(
    *,
    batch_size: int,
    query_len: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn(
        batch_size,
        query_len,
        NUM_KV_HEADS,
        HEAD_DIM,
        generator=generator,
    )
    value = torch.randn(
        batch_size,
        query_len,
        NUM_KV_HEADS,
        HEAD_DIM,
        generator=generator,
    )
    return key, value


def _batch(
    binding: DiffusionPageBinding,
    *,
    query_len: int,
    seq_len: int,
    position_ids: torch.Tensor | None = None,
):
    rows = len(binding.sequences)
    return build_hunyuan_paged_kv_batch(
        binding,
        query_lens=[query_len] * rows,
        seq_lens=[seq_len] * rows,
        block_size=BLOCK_SIZE,
        position_ids=position_ids,
    )


def test_no_cfg_row_maps_stable_and_dynamic_tokens_in_dense_order() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 3),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        )
    )

    batch = _batch(
        binding,
        query_len=8,
        seq_len=8,
        position_ids=torch.arange(8).reshape(1, 8),
    )

    assert batch.logical_positions[0].tolist() == list(range(8))
    assert batch.stable_write_masks[0].tolist() == [True] * 4 + [False] * 4
    assert batch.dynamic_write_masks[0].tolist() == [False] * 4 + [True] * 4
    assert batch.cacheable_block_ids == ((1,),)


def test_cfg_rows_write_to_distinct_scheduler_pages() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 2),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        ),
        _sequence(
            1,
            block_ids=(7, 8),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        ),
    )
    batch = _batch(binding, query_len=8, seq_len=8)
    pages = _pages()
    key, value = _kv(batch_size=2, query_len=8, seed=1)

    write_hunyuan_kv(pages, batch, key, value)
    gathered_key, gathered_value = gather_hunyuan_kv_reference(pages, batch)

    torch.testing.assert_close(gathered_key, key)
    torch.testing.assert_close(gathered_value, value)
    assert not torch.equal(pages[:, 1:3], pages[:, 7:9])


def test_partial_stable_block_is_not_cacheable_or_overwritten_as_stable() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(2, 5),
            prefix_len=5,
            target_len=3,
            seq_len=8,
        )
    )

    batch = _batch(binding, query_len=8, seq_len=8)

    assert batch.cacheable_block_ids == ((2,),)
    assert batch.stable_write_masks[0].tolist() == [True] * 5 + [False] * 3
    assert batch.dynamic_write_masks[0].tolist() == [False] * 5 + [True] * 3
    assert batch.slot_mappings[0].tolist() == [8, 9, 10, 11, 20, 21, 22, 23]


def test_imported_complete_stable_pages_are_preserved_while_local_tail_is_written() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 4),
            prefix_len=6,
            target_len=2,
            seq_len=8,
        ),
        page_states={
            1: PageState.COMMITTED,
            4: PageState.RESERVED,
        },
        externally_required=frozenset({1}),
    )
    batch = _batch(
        binding,
        query_len=4,
        seq_len=8,
        position_ids=torch.arange(4, 8).reshape(1, 4),
    )
    pages = _pages()
    pages[:, 1].fill_(9)
    imported_before = pages[:, 1].clone()
    key, value = _kv(batch_size=1, query_len=4, seed=2)

    write_hunyuan_kv(pages, batch, key, value)
    gathered_key, gathered_value = gather_hunyuan_kv_reference(pages, batch)

    torch.testing.assert_close(pages[:, 1], imported_before)
    torch.testing.assert_close(gathered_key[:, :4], torch.full_like(gathered_key[:, :4], 9))
    torch.testing.assert_close(gathered_key[:, 4:], key)
    torch.testing.assert_close(gathered_value[:, :4], torch.full_like(gathered_value[:, :4], 9))
    torch.testing.assert_close(gathered_value[:, 4:], value)
    assert batch.imported_prefix_token_counts == (4,)


def test_imported_page_must_be_committed_before_batch_is_compute_ready() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 4),
            prefix_len=6,
            target_len=2,
            seq_len=8,
        ),
        page_states={
            1: PageState.INSTALLING_LOCAL,
            4: PageState.RESERVED,
        },
        externally_required=frozenset({1}),
    )

    with pytest.raises(ValueError, match="imported stable page 1 is not committed"):
        _batch(binding, query_len=4, seq_len=8)


def test_second_step_reuses_stable_slots_and_overwrites_dynamic_slots() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(2, 6),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        )
    )
    first_batch = _batch(binding, query_len=8, seq_len=8)
    second_batch = _batch(
        binding,
        query_len=8,
        seq_len=8,
        position_ids=torch.arange(8).reshape(1, 8),
    )
    pages = _pages()
    first_key, first_value = _kv(batch_size=1, query_len=8, seed=3)
    write_hunyuan_kv(pages, first_batch, first_key, first_value)
    stable_before = pages[:, 2].clone()
    dynamic_before = pages[:, 6].clone()

    second_key, second_value = _kv(batch_size=1, query_len=8, seed=4)
    write_hunyuan_kv(
        pages,
        second_batch,
        second_key,
        second_value,
        stable_already_committed=True,
    )

    torch.testing.assert_close(pages[:, 2], stable_before)
    assert not torch.equal(pages[:, 6], dynamic_before)
    gathered_key, gathered_value = gather_hunyuan_kv_reference(pages, first_batch)
    torch.testing.assert_close(gathered_key[:, :4], first_key[:, :4])
    torch.testing.assert_close(gathered_value[:, :4], first_value[:, :4])
    torch.testing.assert_close(gathered_key[:, 4:], second_key[:, 4:])
    torch.testing.assert_close(gathered_value[:, 4:], second_value[:, 4:])


def test_page_storage_block_size_mismatch_fails_before_write() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 2),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        )
    )
    batch = _batch(binding, query_len=8, seq_len=8)
    wrong_pages = torch.zeros(
        2,
        NUM_BLOCKS * 2,
        BLOCK_SIZE // 2,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    key, value = _kv(batch_size=1, query_len=8, seed=6)

    with pytest.raises(ValueError, match="block size mismatch"):
        write_hunyuan_kv(wrong_pages, batch, key, value)


def test_target_blocks_are_never_published_as_cacheable_prefix_blocks() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(3, 5, 9),
            prefix_len=5,
            target_len=4,
            seq_len=12,
        )
    )

    batch = _batch(binding, query_len=12, seq_len=12)

    assert batch.cacheable_block_ids == ((3,),)
    assert set(batch.cacheable_block_ids[0]).isdisjoint(binding.sequences[0].dynamic.block_ids)


@pytest.mark.parametrize(
    ("query_lens", "seq_lens", "match"),
    [
        ([8], [8], "row count"),
        ([8, 7], [8, 8], "uniform query length"),
        ([8, 8], [8, 7], "sequence length mismatch"),
    ],
)
def test_row_geometry_mismatch_fails(
    query_lens: list[int],
    seq_lens: list[int],
    match: str,
) -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 2),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        ),
        _sequence(
            1,
            block_ids=(3, 4),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        ),
    )

    with pytest.raises(ValueError, match=match):
        build_hunyuan_paged_kv_batch(
            binding,
            query_lens=query_lens,
            seq_lens=seq_lens,
            block_size=BLOCK_SIZE,
        )


def test_position_ids_must_match_logical_rope_token_order() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 2),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        )
    )

    with pytest.raises(ValueError, match="position IDs do not match"):
        _batch(
            binding,
            query_len=4,
            seq_len=8,
            position_ids=torch.tensor([[4, 6, 5, 7]]),
        )


def test_reference_gather_records_bounded_scratch_cost() -> None:
    binding = _binding(
        _sequence(
            0,
            block_ids=(1, 2),
            prefix_len=4,
            target_len=4,
            seq_len=8,
        )
    )
    batch = _batch(binding, query_len=8, seq_len=8)
    pages = _pages()
    key, value = _kv(batch_size=1, query_len=8, seed=5)
    write_hunyuan_kv(pages, batch, key, value)

    gathered_key, gathered_value = gather_hunyuan_kv_reference(pages, batch)

    expected_bytes = (gathered_key.numel() + gathered_value.numel()) * gathered_key.element_size()
    assert batch.gather_count == 1
    assert batch.gathered_bytes == expected_bytes
    assert batch.gather_latency_s >= 0
