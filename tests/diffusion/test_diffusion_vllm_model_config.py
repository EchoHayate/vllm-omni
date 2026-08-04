# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_omni.diffusion.worker.diffusion_worker import _DiffusionVllmModelConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_diffusion_model_config_provides_head_dtype_to_logits_processor():
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.model_config = _DiffusionVllmModelConfig(
        model="test-model",
        dtype=torch.bfloat16,
    )

    with set_current_vllm_config(vllm_config):
        logits_processor = LogitsProcessor(vocab_size=16)

    assert logits_processor.head_dtype is torch.bfloat16
