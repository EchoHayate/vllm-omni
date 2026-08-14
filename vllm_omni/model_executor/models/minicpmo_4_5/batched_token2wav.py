# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the shared CosyVoice2 batching backend."""

from vllm_omni.model_executor.models.common.cosyvoice2_batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
    PromptFeatures,
    state_shape_signature,
    tensor_signature,
)

__all__ = [
    "BatchedToken2Wav",
    "BatchedToken2WavState",
    "PromptFeatures",
    "state_shape_signature",
    "tensor_signature",
]
