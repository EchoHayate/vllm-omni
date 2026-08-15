# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.diffusion_kv.page import (
    DiffusionPageBinding,
    DiffusionPageRange,
    DiffusionSequenceBinding,
    PageState,
    build_slot_mapping,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_build_slot_mapping_uses_scheduler_block_ids() -> None:
    slots = build_slot_mapping(
        block_ids=(3, 9),
        block_size=4,
        token_start=1,
        token_count=6,
        device=torch.device("cpu"),
    )
    assert slots.tolist() == [13, 14, 15, 36, 37, 38]


def test_binding_rejects_dynamic_range_marked_immutable() -> None:
    stable = DiffusionPageRange("primary", 0, 4, (1,), mutable=False)
    dynamic = DiffusionPageRange("primary", 4, 4, (2,), mutable=False)
    with pytest.raises(ValueError, match="dynamic range must be mutable"):
        DiffusionSequenceBinding(
            sequence_id=0,
            seq_len=8,
            stable=stable,
            dynamic=dynamic,
            slot_mapping=torch.arange(8),
        )


def test_only_committed_external_pages_are_reusable() -> None:
    binding = DiffusionPageBinding(
        request_id="req",
        allocation_generation=7,
        sequences=(),
        page_states={4: PageState.INSTALLING_LOCAL},
        externally_required=frozenset({4}),
    )
    assert not binding.is_compute_ready
    binding.transition_page(4, PageState.COMMITTED)
    assert binding.is_compute_ready
