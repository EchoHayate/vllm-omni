# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor

from vllm_omni.diffusion.worker.diffusion_worker import (
    _DiffusionVllmModelConfig,
    _make_diffusion_vllm_model_config,
)
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_diffusion_vllm_model_config_supplies_dtype_for_quant_methods():
    od_config = SimpleNamespace(
        model="dummy",
        dtype=torch.bfloat16,
        quantization_config=build_quant_config(
            {
                "quant_method": "modelopt",
                "quant_algo": "FP8",
                "ignore": [],
            }
        ),
        tf_model_config=SimpleNamespace(),
        enforce_eager=True,
        is_moe=False,
    )

    model_config = _make_diffusion_vllm_model_config(od_config)

    assert model_config.dtype is torch.bfloat16
    assert model_config.quantization == "modelopt"
    assert model_config.quantization_config is od_config.quantization_config
    assert model_config.is_quantized()


def test_diffusion_model_config_provides_head_dtype_to_logits_processor():
    vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    vllm_config.model_config = _DiffusionVllmModelConfig(
        model="test-model",
        dtype=torch.bfloat16,
    )

    with set_current_vllm_config(vllm_config):
        logits_processor = LogitsProcessor(vocab_size=16)

    assert logits_processor.head_dtype is torch.bfloat16
