# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MiniCPM-o 4.5 code2wav asset-root resolution (#5442).

In hub/CI deployments ``model_config.model`` is a repo id rather than a local
directory, so asset lookups must resolve to the downloaded snapshot. The
resolution must stay lazy: constructing the model with a fake path (as the
CPU unit tests do) must not touch the hub — only the first asset access may.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import (
    MiniCPMO45Code2Wav,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_model(model: str, extra: dict | None = None) -> MiniCPMO45Code2Wav:
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model=model,
            stage_connector_config={"extra": extra} if extra is not None else None,
        )
    )
    return MiniCPMO45Code2Wav(vllm_config=config)


def _no_hub(monkeypatch):
    def _fail(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("snapshot_download must not be called here")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fail)


def test_local_directory_is_used_unchanged(tmp_path, monkeypatch):
    _no_hub(monkeypatch)
    model = _make_model(str(tmp_path))
    assert model._asset_root() == tmp_path
    assert model._default_prompt_wav == str(tmp_path / "assets" / "HT_ref_audio.wav")


def test_repo_id_resolves_via_snapshot_download_and_caches(tmp_path, monkeypatch):
    calls = []

    def _fake_snapshot_download(model_ref, allow_patterns=None, **kwargs):
        calls.append({"model_ref": model_ref, "allow_patterns": allow_patterns})
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot_download)
    model = _make_model("openbmb/MiniCPM-o-4_5")
    assert model._default_prompt_wav == str(tmp_path / "snapshot" / "assets" / "HT_ref_audio.wav")
    assert calls[0]["model_ref"] == "openbmb/MiniCPM-o-4_5"
    assert "assets/*" in calls[0]["allow_patterns"]
    # The resolved root is cached: further asset accesses must not re-download.
    assert model._asset_root() == Path(tmp_path / "snapshot")
    assert len(calls) == 1


def test_snapshot_download_failure_propagates(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("offline and not cached")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _raise)
    model = _make_model("openbmb/MiniCPM-o-4_5")
    with pytest.raises(FileNotFoundError):
        model._asset_root()


def test_init_with_fake_path_does_not_resolve(monkeypatch):
    """Mirrors the CPU-test construction: constructing with a fake model path
    must not resolve anything — only asset access may."""
    _no_hub(monkeypatch)
    model = _make_model("/fake/model")
    assert model.model_path == "/fake/model"
    assert model._asset_root_cache is None


def test_prompt_wav_override_skips_resolution(monkeypatch):
    _no_hub(monkeypatch)
    model = _make_model("openbmb/MiniCPM-o-4_5", extra={"prompt_wav": "/custom/ref.wav"})
    assert model._default_prompt_wav == "/custom/ref.wav"
