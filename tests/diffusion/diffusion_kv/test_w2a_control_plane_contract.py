# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from inspect import signature

import pytest

from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.sched.interface import NewRequestData
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_w2a_stacks_on_scheduler_owned_control_plane() -> None:
    assert "allocation_generation" in DiffusionKVMetadata.__dataclass_fields__
    assert "diffusion_kv_metadata" in NewRequestData.__dataclass_fields__
    assert "kv_cache_config" in signature(DiffusionKVCacheManager).parameters
    assert hasattr(DiffusionModelRunner, "set_kv_cache_config")
