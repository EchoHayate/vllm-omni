# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.helpers import runtime
from vllm_omni.entrypoints import omni as omni_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_omni_runner_cleans_up_when_omni_initialization_fails(monkeypatch):
    events: list[str] = []

    monkeypatch.setattr(runtime, "cleanup_dist_env_and_memory", lambda: events.append("dist"))
    monkeypatch.setattr(runtime, "run_pre_test_cleanup", lambda: events.append("pre"))
    monkeypatch.setattr(runtime, "run_post_test_cleanup", lambda: events.append("post"))
    monkeypatch.setattr(runtime.OmniRunner, "_cleanup_process", lambda self: events.append("process"))

    def fail_initialization(**kwargs):
        raise RuntimeError("stage initialization failed")

    monkeypatch.setattr(omni_module, "Omni", fail_initialization)

    with pytest.raises(RuntimeError, match="stage initialization failed"):
        runtime.OmniRunner("fake-model")

    assert events == ["dist", "pre", "post", "process", "pre", "post", "dist"]
