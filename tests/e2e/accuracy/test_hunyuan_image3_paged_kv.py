# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import copy
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from tests.e2e.accuracy.helpers import (
    assert_images_pixel_close,
    assert_similarity,
    compute_image_ssim_psnr,
)
from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

MODEL_NAME = "tencent/HunyuanImage-3.0-Instruct"
MODEL_REVISION = "2ec2c78bee7d4b94157341fba86c4c2c7b1858b2"
PROMPT = "A brown and white dog is running on the grass."
SEED = 42
HEIGHT = 512
WIDTH = 512
NUM_INFERENCE_STEPS = 3
GUIDANCE_SCALES = (1.0, 5.0)
TENSOR_PARALLEL_SIZES = (1, 2)

MEAN_THRESHOLD = 3e-2
P99_THRESHOLD = 3e-1
SSIM_THRESHOLD = 0.97
PSNR_THRESHOLD = 30.0

_BASE_DEPLOY_CONFIG = {
    "pipeline": "hunyuan_image3_dit",
    "async_chunk": False,
    "trust_remote_code": True,
    "stages": [
        {
            "stage_id": 0,
            "max_num_seqs": 2,
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
            "trust_remote_code": True,
            "revision": MODEL_REVISION,
            "devices": "0",
            "distributed_executor_backend": "mp",
            "diffusion_kv_mode": "dense_legacy",
            "vae_use_slicing": False,
            "vae_use_tiling": False,
            "moe_backend": "flashinfer_cutlass",
            "parallel_config": {
                "pipeline_parallel_size": 1,
                "data_parallel_size": 1,
                "tensor_parallel_size": 1,
                "enable_expert_parallel": True,
                "sequence_parallel_size": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "allgather_degree": 1,
                "cfg_parallel_size": 1,
                "vae_patch_parallel_size": 1,
                "use_hsdp": False,
                "hsdp_shard_size": -1,
                "hsdp_replicate_size": 1,
            },
            "default_sampling_params": {
                "seed": SEED,
            },
        }
    ],
}


@dataclass(frozen=True)
class _RunResult:
    image: Image.Image
    peak_memory_mb: float
    snapshots: tuple[dict[str, Any], ...]


def _model_name() -> str:
    return os.environ.get("HUNYUAN_IMAGE3_MODEL", MODEL_NAME)


def _write_deploy_config(
    path: Path,
    *,
    diffusion_kv_mode: str,
    tensor_parallel_size: int,
    data_parallel_size: int = 1,
    enable_expert_parallel: bool = True,
) -> None:
    config = copy.deepcopy(_BASE_DEPLOY_CONFIG)
    stage = config["stages"][0]
    num_devices = tensor_parallel_size * data_parallel_size
    stage["devices"] = ",".join(str(index) for index in range(num_devices))
    stage["diffusion_kv_mode"] = diffusion_kv_mode
    stage["parallel_config"]["tensor_parallel_size"] = tensor_parallel_size
    stage["parallel_config"]["data_parallel_size"] = data_parallel_size
    stage["parallel_config"]["enable_expert_parallel"] = enable_expert_parallel
    path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False))


def _extract_image(outputs: list[Any]) -> Image.Image:
    for output in outputs:
        images = getattr(output, "images", None)
        if images:
            image = images[0].convert("RGB")
            image.load()
            return image
    raise AssertionError("HunyuanImage3 produced no image output")


def _extract_peak_memory_mb(outputs: list[Any]) -> float:
    return max((float(getattr(output, "peak_memory_mb", 0.0) or 0.0) for output in outputs), default=0.0)


def _collect_rank_snapshots(value: object) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"rank", "enabled", "metrics", "active_bindings"}.issubset(value):
            snapshots.append(value)
        else:
            for nested in value.values():
                snapshots.extend(_collect_rank_snapshots(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            snapshots.extend(_collect_rank_snapshots(nested))
    return snapshots


def _run_hunyuan(
    deploy_config: Path,
    *,
    guidance_scale: float,
) -> _RunResult:
    from tests.helpers.runtime import OmniRunner

    generator = torch.Generator(device=current_omni_platform.device_type or "cuda").manual_seed(SEED)
    params = OmniDiffusionSamplingParams(
        height=HEIGHT,
        width=WIDTH,
        seed=SEED,
        generator=generator,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=guidance_scale,
        guidance_scale_provided=True,
    )
    with OmniRunner(
        _model_name(),
        deploy_config=str(deploy_config),
        mode="text-to-image",
        trust_remote_code=True,
        revision=MODEL_REVISION,
    ) as runner:
        outputs = list(runner.omni.generate({"prompt": PROMPT}, params))
        rpc_results = runner.omni.engine.collective_rpc(
            method="get_diffusion_page_debug_snapshots",
            timeout=120,
            stage_ids=[0],
        )
        snapshots = tuple(_collect_rank_snapshots(rpc_results))
        return _RunResult(
            image=_extract_image(outputs),
            peak_memory_mb=_extract_peak_memory_mb(outputs),
            snapshots=snapshots,
        )


@pytest.fixture(scope="module")
def hunyuan_page_native_matrix(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[float, int], tuple[_RunResult, _RunResult]]:
    old_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
    root = tmp_path_factory.mktemp("hunyuan-page-native")
    dense_results: dict[float, _RunResult] = {}
    matrix: dict[tuple[float, int], tuple[_RunResult, _RunResult]] = {}
    try:
        dense_config = root / "dense-tp1.yaml"
        _write_deploy_config(
            dense_config,
            diffusion_kv_mode="dense_legacy",
            tensor_parallel_size=1,
        )
        for guidance_scale in GUIDANCE_SCALES:
            dense_results[guidance_scale] = _run_hunyuan(
                dense_config,
                guidance_scale=guidance_scale,
            )

        for tensor_parallel_size in TENSOR_PARALLEL_SIZES:
            paged_config = root / f"paged-tp{tensor_parallel_size}.yaml"
            _write_deploy_config(
                paged_config,
                diffusion_kv_mode="paged_scheduler",
                tensor_parallel_size=tensor_parallel_size,
            )
            for guidance_scale in GUIDANCE_SCALES:
                paged = _run_hunyuan(
                    paged_config,
                    guidance_scale=guidance_scale,
                )
                matrix[(guidance_scale, tensor_parallel_size)] = (
                    dense_results[guidance_scale],
                    paged,
                )
    finally:
        if old_backend is None:
            os.environ.pop("DIFFUSION_ATTENTION_BACKEND", None)
        else:
            os.environ["DIFFUSION_ATTENTION_BACKEND"] = old_backend
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
    return matrix


@pytest.fixture(scope="module")
def hunyuan_page_native_dp_idle(
    tmp_path_factory: pytest.TempPathFactory,
) -> _RunResult:
    old_backend = os.environ.get("DIFFUSION_ATTENTION_BACKEND")
    os.environ["DIFFUSION_ATTENTION_BACKEND"] = "TORCH_SDPA"
    root = tmp_path_factory.mktemp("hunyuan-page-native-dp-idle")
    config = root / "paged-dp2-tp2.yaml"
    _write_deploy_config(
        config,
        diffusion_kv_mode="paged_scheduler",
        tensor_parallel_size=2,
        data_parallel_size=2,
        enable_expert_parallel=False,
    )
    try:
        return _run_hunyuan(
            config,
            guidance_scale=1.0,
        )
    finally:
        if old_backend is None:
            os.environ.pop("DIFFUSION_ATTENTION_BACKEND", None)
        else:
            os.environ["DIFFUSION_ATTENTION_BACKEND"] = old_backend
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()


def _image_metrics(dense: Image.Image, paged: Image.Image) -> dict[str, float]:
    dense_array = np.asarray(dense, dtype=np.float32) / 255.0
    paged_array = np.asarray(paged, dtype=np.float32) / 255.0
    absolute_diff = np.abs(dense_array - paged_array)
    ssim, psnr = compute_image_ssim_psnr(
        prediction=paged,
        reference=dense,
    )
    return {
        "mean_abs_diff": float(np.mean(absolute_diff)),
        "p99_abs_diff": float(np.percentile(absolute_diff, 99)),
        "ssim": float(ssim),
        "psnr": float(psnr),
    }


def _assert_paged_rank_snapshots(
    snapshots: tuple[dict[str, Any], ...],
    *,
    tensor_parallel_size: int,
) -> None:
    assert len(snapshots) == tensor_parallel_size
    generations: set[int] = set()
    for snapshot in snapshots:
        assert snapshot["enabled"] is True
        assert snapshot["active_bindings"] == []
        metrics = snapshot["metrics"]
        assert metrics["stable_pages_requested"] > 0
        assert metrics["reference_gather_bytes"] > 0
        assert metrics["page_pool_pages_in_use"] == 0
        released = [terminal for terminal in metrics["terminal_snapshots"] if terminal["terminal_status"] == "released"]
        assert released
        generations.add(int(released[-1]["allocation_generation"]))
    assert len(generations) == 1


@pytest.mark.parametrize(
    ("guidance_scale", "tensor_parallel_size"),
    [
        (1.0, 1),
        (5.0, 1),
        (1.0, 2),
        (5.0, 2),
    ],
)
@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_hunyuan_image3_paged_kv_matches_dense(
    hunyuan_page_native_matrix: dict[tuple[float, int], tuple[_RunResult, _RunResult]],
    guidance_scale: float,
    tensor_parallel_size: int,
) -> None:
    dense, paged = hunyuan_page_native_matrix[(guidance_scale, tensor_parallel_size)]
    metrics = _image_metrics(dense.image, paged.image)
    rank_metrics = [snapshot["metrics"] for snapshot in paged.snapshots]
    report = {
        "guidance_scale": guidance_scale,
        "tensor_parallel_size": tensor_parallel_size,
        **metrics,
        "dense_peak_memory_mb": dense.peak_memory_mb,
        "paged_peak_memory_mb": paged.peak_memory_mb,
        "reference_gather_bytes": sum(item["reference_gather_bytes"] for item in rank_metrics),
    }
    print("HUNYUAN_PAGE_NATIVE_ACCURACY " + json.dumps(report, sort_keys=True))

    assert_images_pixel_close(
        model_name=f"{MODEL_NAME} dense vs paged gs={guidance_scale} tp={tensor_parallel_size}",
        vllm_image=paged.image,
        diffusers_image=dense.image,
        mean_threshold=MEAN_THRESHOLD,
        p99_threshold=P99_THRESHOLD,
    )
    assert_similarity(
        model_name=f"{MODEL_NAME} dense vs paged gs={guidance_scale} tp={tensor_parallel_size}",
        vllm_image=paged.image,
        diffusers_image=dense.image,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )
    _assert_paged_rank_snapshots(
        paged.snapshots,
        tensor_parallel_size=tensor_parallel_size,
    )
    assert all(snapshot["enabled"] is False for snapshot in dense.snapshots)


@hardware_test(res={"cuda": "H100"}, num_cards=4)
def test_hunyuan_image3_paged_kv_dp2_tp2_keeps_idle_replica_clean(
    hunyuan_page_native_matrix: dict[
        tuple[float, int],
        tuple[_RunResult, _RunResult],
    ],
    hunyuan_page_native_dp_idle: _RunResult,
) -> None:
    dense, _ = hunyuan_page_native_matrix[(1.0, 2)]
    paged = hunyuan_page_native_dp_idle
    metrics = _image_metrics(dense.image, paged.image)
    print(
        "HUNYUAN_PAGE_NATIVE_DP_IDLE "
        + json.dumps(
            {
                **metrics,
                "tensor_parallel_size": 2,
                "data_parallel_size": 2,
                "enable_expert_parallel": False,
                "dense_peak_memory_mb": dense.peak_memory_mb,
                "paged_peak_memory_mb": paged.peak_memory_mb,
            },
            sort_keys=True,
        )
    )
    assert_images_pixel_close(
        model_name=f"{MODEL_NAME} dense vs paged dp2 tp2",
        vllm_image=paged.image,
        diffusers_image=dense.image,
        mean_threshold=MEAN_THRESHOLD,
        p99_threshold=P99_THRESHOLD,
    )
    assert_similarity(
        model_name=f"{MODEL_NAME} dense vs paged dp2 tp2",
        vllm_image=paged.image,
        diffusers_image=dense.image,
        ssim_threshold=SSIM_THRESHOLD,
        psnr_threshold=PSNR_THRESHOLD,
    )

    assert len(paged.snapshots) == 4
    active = [snapshot for snapshot in paged.snapshots if snapshot["data_parallel_rank"] == 0]
    idle = [snapshot for snapshot in paged.snapshots if snapshot["data_parallel_rank"] == 1]
    assert {snapshot["tensor_parallel_rank"] for snapshot in active} == {0, 1}
    assert {snapshot["tensor_parallel_rank"] for snapshot in idle} == {0, 1}

    active_generations: set[int] = set()
    for snapshot in active:
        assert snapshot["active_bindings"] == []
        metrics = snapshot["metrics"]
        assert metrics["stable_pages_requested"] > 0
        released = [terminal for terminal in metrics["terminal_snapshots"] if terminal["terminal_status"] == "released"]
        assert released
        active_generations.add(int(released[-1]["allocation_generation"]))
    assert len(active_generations) == 1

    for snapshot in idle:
        assert snapshot["active_bindings"] == []
        metrics = snapshot["metrics"]
        assert metrics["stable_pages_requested"] == 0
        assert metrics["stable_pages_committed"] == 0
        assert metrics["page_pool_pages_in_use"] == 0
        assert metrics["terminal_snapshots"] == []
