# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.page import (
    DiffusionPageBinding,
    DiffusionPageRange,
    DiffusionSequenceBinding,
    PageState,
    build_slot_mapping,
)
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_TRANSFORMER_MODULE = "vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer"
NUM_HEADS = 4
NUM_KV_HEADS = 2
HEAD_DIM = 8
BLOCK_SIZE = 4


class RecordingAttention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.paged_kv_cache: torch.Tensor | None = None
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def forward(self, query, key, value, attn_metadata=None, **kwargs):
        del attn_metadata, kwargs
        self.calls.append((query.detach().clone(), key.detach().clone(), value.detach().clone()))
        return query


@contextmanager
def _patched_manager_environment():
    patches = [
        patch(f"{_TRANSFORMER_MODULE}.get_sequence_parallel_world_size", return_value=1),
        patch(f"{_TRANSFORMER_MODULE}.get_allgather_parallel_world_size", return_value=1),
        patch(f"{_TRANSFORMER_MODULE}.get_ulysses_parallel_world_size", return_value=1, create=True),
        patch(f"{_TRANSFORMER_MODULE}.get_sequence_parallel_rank", return_value=0),
        patch(f"{_TRANSFORMER_MODULE}.Attention", RecordingAttention),
    ]
    with ExitStack() as stack:
        for manager in patches:
            stack.enter_context(manager)
        yield


def _manager():
    with _patched_manager_environment():
        from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
            ImageKVCacheManager,
        )

        return ImageKVCacheManager(
            num_heads=NUM_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            scaling=1.0 / math.sqrt(HEAD_DIM),
            image_token_len=4,
        )


def _binding() -> DiffusionPageBinding:
    return DiffusionPageBinding(
        request_id="req",
        allocation_generation=11,
        sequences=(
            DiffusionSequenceBinding(
                sequence_id=0,
                seq_len=8,
                stable=DiffusionPageRange("primary", 0, 4, (1,), mutable=False),
                dynamic=DiffusionPageRange("primary", 4, 4, (3,), mutable=True),
                slot_mapping=build_slot_mapping(
                    block_ids=(1, 3),
                    block_size=BLOCK_SIZE,
                    token_start=0,
                    token_count=8,
                    device=torch.device("cpu"),
                ),
            ),
        ),
        page_states={
            1: PageState.RESERVED,
            3: PageState.RESERVED,
        },
        externally_required=frozenset(),
    )


def _run_step(
    manager,
    *,
    mode: DiffusionKVCacheMode,
    binding: DiffusionPageBinding | None,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    first_step: bool,
) -> torch.Tensor:
    query_len = query.shape[1]
    seq_len = 8
    context = ForwardContext(
        omni_diffusion_config=SimpleNamespace(diffusion_kv_mode=mode),
        diffusion_page_bindings=None if binding is None else {"req": binding},
    )
    with override_forward_context(context):
        return manager(
            query.flatten(0, 1),
            key.flatten(0, 1),
            value.flatten(0, 1),
            torch.zeros(1, 1, query_len, seq_len),
            query_lens=[query_len],
            seq_lens=[seq_len],
            first_step=first_step,
            uncond_cfg_prefill=False,
            num_image_tokens=4,
            gen_timestep_scatter_index=torch.tensor([[4]]),
            position_ids=torch.arange(seq_len - query_len, seq_len).reshape(1, query_len),
            request_ids=["req"],
            row_branches=[0],
        )


def test_paged_matches_dense_for_first_and_two_later_steps() -> None:
    dense = _manager()
    paged = _manager()
    paged.attn.paged_kv_cache = torch.zeros(
        2,
        6,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )
    binding = _binding()
    generator = torch.Generator().manual_seed(20260815)
    steps = [
        (
            torch.randn(1, 8, NUM_HEADS, HEAD_DIM, generator=generator),
            torch.randn(1, 8, NUM_KV_HEADS, HEAD_DIM, generator=generator),
            torch.randn(1, 8, NUM_KV_HEADS, HEAD_DIM, generator=generator),
            True,
        ),
        (
            torch.randn(1, 4, NUM_HEADS, HEAD_DIM, generator=generator),
            torch.randn(1, 4, NUM_KV_HEADS, HEAD_DIM, generator=generator),
            torch.randn(1, 4, NUM_KV_HEADS, HEAD_DIM, generator=generator),
            False,
        ),
        (
            torch.randn(1, 4, NUM_HEADS, HEAD_DIM, generator=generator),
            torch.randn(1, 4, NUM_KV_HEADS, HEAD_DIM, generator=generator),
            torch.randn(1, 4, NUM_KV_HEADS, HEAD_DIM, generator=generator),
            False,
        ),
    ]

    for query, key, value, first_step in steps:
        dense_output = _run_step(
            dense,
            mode=DiffusionKVCacheMode.DENSE_LEGACY,
            binding=None,
            query=query,
            key=key,
            value=value,
            first_step=first_step,
        )
        paged_output = _run_step(
            paged,
            mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            binding=binding,
            query=query,
            key=key,
            value=value,
            first_step=first_step,
        )
        torch.testing.assert_close(paged_output, dense_output)

    assert len(dense.attn.calls) == len(paged.attn.calls) == 3
    for dense_call, paged_call in zip(dense.attn.calls, paged.attn.calls):
        for dense_tensor, paged_tensor in zip(dense_call, paged_call):
            torch.testing.assert_close(paged_tensor, dense_tensor)
    assert paged.image_kv_cache_map is None
    assert paged.image_kv_cache_lens is None
    assert binding.locally_produced_pages == frozenset({1})
    assert binding.local_write_event is not None
    assert binding.local_write_event.query()
