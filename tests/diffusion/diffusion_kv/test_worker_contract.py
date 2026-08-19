# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
import vllm_omni.diffusion.worker.diffusion_worker as worker_module
from vllm_omni.diffusion.data import DiffusionOutput, DiffusionPageMetrics
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.executor.multiproc_executor import MultiprocDiffusionExecutor
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.sched.interface import (
    CachedRequestData,
    DiffusionSchedulerOutput,
    NewRequestData,
    PageInstallRequest,
    PageReleaseRequest,
    WorkerKVUpdate,
)
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import (
    DiffusionWorker,
    WorkerProc,
    WorkerWrapperBase,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def make_metadata(request_id: str = "req-0") -> DiffusionKVMetadata:
    return DiffusionKVMetadata(
        request_id=request_id,
        allocation_generation=1,
        sequences=(
            DiffusionKVSequenceMetadata(
                sequence_id=0,
                prefix_len=4,
                target_len=2,
                seq_len=8,
                block_ids=([1, 2],),
            ),
        ),
    )


def make_runner(mode: DiffusionKVCacheMode) -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(diffusion_kv_mode=mode)
    runner.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_model_len=32,
        )
    )
    runner.device = torch.device("cpu")
    runner.kv_cache_config = None
    runner.page_registry = None
    runner._diffusion_page_bindings = {}
    return runner


def make_scheduler_output(
    *new_reqs: NewRequestData,
    cached: tuple[str, ...] = (),
    finished: set[str] | None = None,
    page_installs: tuple[PageInstallRequest, ...] = (),
    page_releases: tuple[PageReleaseRequest, ...] = (),
) -> DiffusionSchedulerOutput:
    return DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=CachedRequestData(request_ids=list(cached)),
        finished_req_ids=finished or set(),
        num_running_reqs=len(new_reqs) + len(cached),
        num_waiting_reqs=0,
        page_install_reqs=list(page_installs),
        page_release_reqs=list(page_releases),
    )


def make_new_request(
    request_id: str,
    *,
    generation: int = 1,
) -> NewRequestData:
    return NewRequestData(
        request_id=request_id,
        req=SimpleNamespace(request_id=request_id),
        diffusion_kv_metadata=DiffusionKVMetadata(
            request_id=request_id,
            allocation_generation=generation,
            sequences=(),
        ),
    )


def make_kv_cache_config():
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer0"],
            )
        ]
    )


def make_executor() -> tuple[MultiprocDiffusionExecutor, list[tuple]]:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
    )
    executor._ensure_open = lambda: None
    calls: list[tuple] = []

    def collective_rpc(method, *, args, unique_reply_rank, exec_all_ranks):
        calls.append((method, args, unique_reply_rank, exec_all_ranks))
        return DiffusionOutput(output=None)

    executor.collective_rpc = collective_rpc
    return executor, calls


def test_new_request_data_carries_scheduler_allocation_atomically() -> None:
    req = SimpleNamespace(request_id="req-0")
    state = SimpleNamespace(request_id="req-0", req=req)
    metadata = make_metadata()

    new_req = NewRequestData.from_state(state, diffusion_kv_metadata=metadata)

    assert new_req.req is req
    assert new_req.diffusion_kv_metadata is metadata


def test_paged_request_without_metadata_fails_before_request_forward() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()
    req = SimpleNamespace(request_id="req-0")

    with pytest.raises(ValueError, match="requires Diffusion KV metadata"):
        runner.execute_model(req)

    runner._execute_request_list.assert_not_called()


def test_dense_request_rejects_metadata_before_request_forward() -> None:
    runner = make_runner(DiffusionKVCacheMode.DENSE_LEGACY)
    runner._execute_request_list = Mock()

    with pytest.raises(ValueError, match="dense_legacy.*must not carry"):
        runner.execute_model(
            SimpleNamespace(request_id="req-0"),
            diffusion_kv_metadata=make_metadata(),
        )

    runner._execute_request_list.assert_not_called()


def test_paged_request_rejects_mismatched_metadata_identity() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()

    with pytest.raises(ValueError, match="request mismatch"):
        runner.execute_model(
            SimpleNamespace(request_id="req-0"),
            diffusion_kv_metadata=make_metadata("stale"),
        )

    runner._execute_request_list.assert_not_called()


def test_request_rpc_rejects_envelope_request_identity_mismatch() -> None:
    executor, calls = make_executor()
    req = SimpleNamespace(request_id="actual-request-id")
    new_req = NewRequestData(
        request_id="envelope-id",
        req=req,
        diffusion_kv_metadata=make_metadata("envelope-id"),
    )

    with pytest.raises(ValueError, match="request identity mismatch"):
        executor.execute_request(make_scheduler_output(new_req))

    assert calls == []


def test_executor_runs_page_control_on_all_ranks_without_model_request() -> None:
    executor = object.__new__(MultiprocDiffusionExecutor)
    executor.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        enable_distributed_layerwise_offload=False,
        dlo_use_allgather=True,
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            enable_expert_parallel=False,
        ),
    )
    executor._ensure_open = lambda: None
    update_rank0 = WorkerKVUpdate("req", 1, tp_rank=0, status="ready")
    update_rank1 = WorkerKVUpdate("req", 1, tp_rank=1, status="ready")
    calls = []

    def collective_rpc(
        method,
        *,
        args,
        unique_reply_rank,
        exec_all_ranks,
        timeout=None,
    ):
        calls.append((method, args, unique_reply_rank, exec_all_ranks))
        return [[update_rank0], [update_rank1]]

    executor.collective_rpc = collective_rpc
    metadata = make_metadata("req")
    scheduler_output = make_scheduler_output(
        page_installs=(
            PageInstallRequest(
                request_id="req",
                allocation_generation=1,
                metadata=metadata,
            ),
        )
    )

    output = executor.execute_request(scheduler_output)

    assert output.runner_outputs == []
    assert output.worker_kv_updates == [update_rank0, update_rank1]
    assert calls == [
        (
            "update_diffusion_kv_pages",
            (scheduler_output,),
            None,
            True,
        )
    ]


@pytest.mark.parametrize("status", ["ready", "released"])
def test_page_control_rpc_primary_returns_updates_from_all_tp_ranks(
    monkeypatch,
    status,
) -> None:
    update_rank0 = WorkerKVUpdate(
        "req",
        1,
        tp_rank=0,
        status=status,
        data_parallel_rank=1,
    )
    update_rank1 = WorkerKVUpdate(
        "req",
        1,
        tp_rank=1,
        status=status,
        data_parallel_rank=1,
    )
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = SimpleNamespace(
        _process_diffusion_page_control=Mock(return_value=[update_rank0]),
    )
    worker_wrapper = SimpleNamespace(
        execute_method=lambda method, *args, **kwargs: getattr(worker, method)(
            *args,
            **kwargs,
        )
    )
    worker_proc = object.__new__(WorkerProc)
    worker_proc.gpu_id = 0
    worker_proc.result_mq = object()
    worker_proc.worker = worker_wrapper
    tp_cpu_group = object()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group=tp_cpu_group),
    )

    def all_gather_object(outputs, local_value, *, group):
        assert group is tp_cpu_group
        outputs[:] = [
            local_value,
            (True, [update_rank1]),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_classifier_free_guidance_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_pipeline_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    scheduler_output = (
        make_scheduler_output(
            page_installs=(
                PageInstallRequest(
                    request_id="req",
                    allocation_generation=1,
                    metadata=make_metadata("req"),
                    data_parallel_rank=1,
                ),
            )
        )
        if status == "ready"
        else make_scheduler_output(
            page_releases=(
                PageReleaseRequest(
                    request_id="req",
                    allocation_generation=1,
                    data_parallel_rank=1,
                ),
            )
        )
    )
    rpc_request = {
        "method": "update_diffusion_kv_pages",
        "args": (scheduler_output,),
        "kwargs": {},
        "output_rank": None,
        "exec_all_ranks": True,
        "collect_rank_status": False,
        "wave_id": 7,
    }

    updates, should_reply = worker_proc._execute_rpc(rpc_request)

    assert should_reply is True
    assert updates == [update_rank0, update_rank1]

    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        lambda: 1,
    )

    _, should_reply = worker_proc._execute_rpc(rpc_request)

    assert should_reply is False


def test_page_control_rpc_gathers_tp_failure_before_raising(
    monkeypatch,
) -> None:
    update_rank1 = WorkerKVUpdate("req", 1, tp_rank=1, status="ready")
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = SimpleNamespace(
        _process_diffusion_page_control=Mock(
            side_effect=ValueError("rank-local failure"),
        ),
    )
    worker_proc = object.__new__(WorkerProc)
    worker_proc.gpu_id = 0
    worker_proc.result_mq = object()
    worker_proc.worker = SimpleNamespace(
        execute_method=lambda method, *args, **kwargs: getattr(worker, method)(
            *args,
            **kwargs,
        )
    )
    tp_cpu_group = object()
    gathered = Mock()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group=tp_cpu_group),
    )

    def all_gather_object(outputs, local_value, *, group):
        gathered(local_value, group)
        outputs[:] = [
            local_value,
            (True, [update_rank1]),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_sequence_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_classifier_free_guidance_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_pipeline_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        lambda: 0,
    )

    with pytest.raises(RuntimeError, match="TP rank 0.*rank-local failure"):
        worker_proc._execute_rpc(
            {
                "method": "update_diffusion_kv_pages",
                "args": (
                    make_scheduler_output(
                        page_installs=(
                            PageInstallRequest(
                                request_id="req",
                                allocation_generation=1,
                                metadata=make_metadata("req"),
                            ),
                        )
                    ),
                ),
                "kwargs": {},
                "output_rank": None,
                "exec_all_ranks": True,
                "collect_rank_status": False,
                "wave_id": 8,
            }
        )

    gathered.assert_called_once()
    local_value, group = gathered.call_args.args
    assert local_value[0] is False
    assert "rank-local failure" in local_value[1]
    assert group is tp_cpu_group


def test_batch_path_rejects_missing_metadata_before_forward() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()
    new_req = NewRequestData(
        request_id="req-0",
        req=SimpleNamespace(request_id="req-0"),
    )

    with pytest.raises(ValueError, match="requires Diffusion KV metadata"):
        runner.execute_model_batch(make_scheduler_output(new_req), runner.od_config)

    runner._execute_request_list.assert_not_called()


def test_batch_path_rejects_envelope_request_identity_mismatch() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner._execute_request_list = Mock()
    new_req = NewRequestData(
        request_id="envelope-id",
        req=SimpleNamespace(request_id="actual-request-id"),
        diffusion_kv_metadata=make_metadata("envelope-id"),
    )

    with pytest.raises(ValueError, match="request identity mismatch"):
        runner.execute_model_batch(make_scheduler_output(new_req), runner.od_config)

    runner._execute_request_list.assert_not_called()


def test_step_path_rejects_missing_metadata_before_step_support_check() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.pipeline = object()
    runner._supports_step_mode = Mock(return_value=False)
    new_req = NewRequestData(
        request_id="req-0",
        req=SimpleNamespace(request_id="req-0"),
    )

    with pytest.raises(ValueError, match="requires Diffusion KV metadata"):
        runner.execute_stepwise(make_scheduler_output(new_req))

    runner._supports_step_mode.assert_not_called()


def test_step_path_rejects_envelope_request_identity_mismatch() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.pipeline = object()
    runner._supports_step_mode = Mock()
    new_req = NewRequestData(
        request_id="envelope-id",
        req=SimpleNamespace(request_id="actual-request-id"),
        diffusion_kv_metadata=make_metadata("envelope-id"),
    )

    with pytest.raises(ValueError, match="request identity mismatch"):
        runner.execute_stepwise(make_scheduler_output(new_req))

    runner._supports_step_mode.assert_not_called()


def test_dense_request_rpc_keeps_legacy_positional_shape() -> None:
    executor, calls = make_executor()
    prepared_layout = object()
    req = SimpleNamespace(request_id="req-0", prepared_layout=prepared_layout)
    new_req = NewRequestData(request_id="req-0", req=req)

    result = executor.execute_request(make_scheduler_output(new_req))

    assert result.request_ids == ["req-0"]
    assert calls == [
        (
            "execute_model",
            (req, executor.od_config, None),
            0,
            True,
        )
    ]
    assert calls[0][1][0].prepared_layout is prepared_layout


def test_request_rpc_sends_attached_metadata_for_only_that_request() -> None:
    executor, calls = make_executor()
    req_0 = SimpleNamespace(request_id="req-0")
    req_1 = SimpleNamespace(request_id="req-1")
    metadata_0 = make_metadata("req-0")
    metadata_1 = make_metadata("req-1")
    new_reqs = (
        NewRequestData(request_id="req-0", req=req_0, diffusion_kv_metadata=metadata_0),
        NewRequestData(request_id="req-1", req=req_1, diffusion_kv_metadata=metadata_1),
    )

    result = executor.execute_request(make_scheduler_output(*new_reqs))

    assert result.request_ids == ["req-0", "req-1"]
    assert [call[1] for call in calls] == [
        (req_0, executor.od_config, None, metadata_0),
        (req_1, executor.od_config, None, metadata_1),
    ]


def test_worker_wrapper_only_extends_paged_request_rpc() -> None:
    calls: list[tuple] = []

    class Worker:
        def execute_model(self, req, od_config, **kwargs):
            calls.append((req, od_config, kwargs))
            return DiffusionOutput(output=None)

    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = Worker()
    req = SimpleNamespace(request_id="req-0")
    od_config = SimpleNamespace()
    metadata = make_metadata()

    wrapper.execute_model(req, od_config)
    wrapper.execute_model(req, od_config, diffusion_kv_metadata=metadata)

    assert calls == [
        (req, od_config, {"kv_prefetch_job": None}),
        (
            req,
            od_config,
            {
                "kv_prefetch_job": None,
                "diffusion_kv_metadata": metadata,
            },
        ),
    ]


def test_dense_worker_call_does_not_extend_model_runner_signature() -> None:
    calls: list[tuple] = []

    class DenseModelRunner:
        def execute_model(self, req, kv_prefetch_job=None):
            calls.append((req, kv_prefetch_job))
            return DiffusionOutput(output=None)

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = DenseModelRunner()
    worker.lora_manager = None
    worker._get_profiler = lambda: None
    req = SimpleNamespace(request_id="req-0")

    worker.execute_model(req, SimpleNamespace())

    assert calls == [(req, None)]


def test_dlo_worker_selects_request_and_metadata_from_same_envelope(monkeypatch) -> None:
    calls: list[tuple] = []

    class ModelRunner:
        def execute_model(self, req, **kwargs):
            calls.append((req, kwargs))
            return DiffusionOutput(output=None)

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = ModelRunner()
    worker.lora_manager = None
    worker._get_profiler = lambda: None
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_data_parallel_rank",
        lambda: 1,
    )
    req_0 = SimpleNamespace(request_id="req-0")
    req_1 = SimpleNamespace(request_id="req-1")
    metadata_0 = make_metadata("req-0")
    metadata_1 = make_metadata("req-1")
    envelopes = [
        NewRequestData(request_id="req-0", req=req_0, diffusion_kv_metadata=metadata_0),
        NewRequestData(request_id="req-1", req=req_1, diffusion_kv_metadata=metadata_1),
    ]

    result = worker.execute_model(envelopes, SimpleNamespace())

    assert calls == [
        (
            req_1,
            {
                "kv_prefetch_job": None,
                "diffusion_kv_metadata": metadata_1,
            },
        )
    ]
    assert result["dp_rank"] == 1


def test_paged_dp_worker_stays_idle_without_assigned_envelope(monkeypatch) -> None:
    calls: list[tuple] = []

    class ModelRunner:
        def execute_model(self, req, **kwargs):
            calls.append((req, kwargs))
            return DiffusionOutput(output=None)

    worker = object.__new__(DiffusionWorker)
    worker.model_runner = ModelRunner()
    worker.lora_manager = None
    worker._get_profiler = lambda: None
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_data_parallel_rank",
        lambda: 1,
    )
    req = SimpleNamespace(request_id="req-0")
    envelope = NewRequestData(
        request_id="req-0",
        req=req,
        diffusion_kv_metadata=make_metadata("req-0"),
        data_parallel_rank=0,
    )

    result = worker.execute_model([envelope], SimpleNamespace())

    assert calls == []
    assert result == {
        "dp_rank": 1,
        "output": None,
        "idle": True,
    }


def test_set_kv_cache_config_initializes_registry_only_in_paged_mode(monkeypatch) -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.get_kv_cache_spec = Mock(return_value={"layer0": object()})
    registry = object()
    registry_cls = Mock(return_value=registry)
    monkeypatch.setattr(model_runner_module, "WorkerPageRegistry", registry_cls)
    config = make_kv_cache_config()

    runner.set_kv_cache_config(config)

    registry_cls.assert_called_once_with(
        kv_cache_config=config,
        layer_specs={"layer0": runner.get_kv_cache_spec.return_value["layer0"]},
        device=torch.device("cpu"),
        max_num_reqs=4,
        max_model_len=32,
    )
    assert runner.page_registry is registry


def test_set_kv_cache_config_installs_registry_storage_on_attention_modules(
    monkeypatch,
) -> None:
    from vllm_omni.diffusion.attention.layer import Attention

    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    attention = object.__new__(Attention)
    torch.nn.Module.__init__(attention)
    attention.paged_kv_cache_role = "primary"
    attention.paged_kv_cache = None
    runner.pipeline = SimpleNamespace(
        named_modules=lambda: [("layer0", attention)],
    )
    layer_spec = object()
    layer_cache = torch.zeros(2, 4, 4, 2, 8)
    runner.get_kv_cache_spec = Mock(return_value={"layer0": layer_spec})
    registry = SimpleNamespace(get_layer_cache=Mock(return_value=layer_cache))
    monkeypatch.setattr(
        model_runner_module,
        "WorkerPageRegistry",
        Mock(return_value=registry),
    )
    runner.set_kv_cache_config(make_kv_cache_config())

    registry.get_layer_cache.assert_called_once_with("layer0")
    assert attention.paged_kv_cache is layer_cache


def test_dense_mode_rejects_page_data_plane_initialization(monkeypatch) -> None:
    runner = make_runner(DiffusionKVCacheMode.DENSE_LEGACY)
    registry_cls = Mock()
    monkeypatch.setattr(model_runner_module, "WorkerPageRegistry", registry_cls)

    with pytest.raises(ValueError, match="dense_legacy.*KVCacheConfig"):
        runner.set_kv_cache_config(make_kv_cache_config())

    assert runner.page_registry is None
    registry_cls.assert_not_called()


def test_runner_construction_does_not_initialize_page_registry_before_cache_sizing(
    monkeypatch,
) -> None:
    legacy_manager = SimpleNamespace(
        config=SimpleNamespace(
            enable_kv_async_prefetch=False,
            need_recv_cache=False,
        )
    )
    monkeypatch.setattr(
        model_runner_module.OmniKVTransferManager,
        "from_od_config",
        Mock(return_value=legacy_manager),
    )
    od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        cfg_kv_collect_func=None,
    )

    runner = DiffusionModelRunner(
        SimpleNamespace(),
        od_config,
        torch.device("cpu"),
    )
    runner._execute_request_list = Mock(return_value=object())
    synchronize = Mock()
    monkeypatch.setattr(
        model_runner_module.current_omni_platform,
        "synchronize",
        synchronize,
    )

    request = SimpleNamespace(request_id="profile")
    runner.profile_run([request])

    assert runner.kv_cache_config is None
    assert runner.page_registry is None
    assert runner._diffusion_page_bindings == {}
    runner._execute_request_list.assert_called_once()
    assert runner._execute_request_list.call_args.args[0] == [request]
    synchronize.assert_called_once()


def test_finished_ids_release_before_rebinding_reused_block_ids() -> None:
    calls: list[tuple] = []

    class Registry:
        def release_request(self, request_id, allocation_generation):
            calls.append(("release", request_id, allocation_generation))

        def bind_request(self, metadata):
            calls.append(("bind", metadata.request_id, metadata.allocation_generation))
            return SimpleNamespace(
                request_id=metadata.request_id,
                allocation_generation=metadata.allocation_generation,
            )

    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = Registry()
    runner._diffusion_page_bindings["old"] = SimpleNamespace(
        request_id="old",
        allocation_generation=3,
    )

    bindings = runner._prepare_diffusion_page_bindings(
        make_scheduler_output(
            make_new_request("new", generation=4),
            finished={"old"},
        )
    )

    assert calls.index(("release", "old", 3)) < calls.index(("bind", "new", 4))
    assert tuple(bindings) == ("new",)


def test_page_control_installs_ready_binding_and_reports_rank_update() -> None:
    binding = SimpleNamespace(
        request_id="req",
        allocation_generation=1,
        is_compute_ready=True,
    )
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = SimpleNamespace(
        bind_request=Mock(return_value=binding),
    )
    metadata = make_metadata("req")

    updates = runner._process_diffusion_page_control(
        make_scheduler_output(
            page_installs=(
                PageInstallRequest(
                    request_id="req",
                    allocation_generation=1,
                    metadata=metadata,
                ),
            )
        )
    )

    assert [(update.request_id, update.status, update.tp_rank) for update in updates] == [("req", "ready", 0)]
    assert runner._diffusion_page_bindings == {"req": binding}


def test_page_control_skips_binding_owned_by_another_dp_replica(
    monkeypatch,
) -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = SimpleNamespace(
        bind_request=Mock(),
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_data_parallel_rank",
        lambda: 1,
    )

    updates = runner._process_diffusion_page_control(
        make_scheduler_output(
            page_installs=(
                PageInstallRequest(
                    request_id="req",
                    allocation_generation=1,
                    metadata=make_metadata("req"),
                    data_parallel_rank=0,
                ),
            )
        )
    )

    assert updates == []
    assert runner._diffusion_page_bindings == {}
    runner.page_registry.bind_request.assert_not_called()


def test_page_control_reports_release_after_worker_binding_cleanup() -> None:
    calls = []
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = SimpleNamespace(
        release_request=lambda request_id, generation: calls.append(("release", request_id, generation))
    )
    runner._diffusion_page_bindings["req"] = SimpleNamespace(
        request_id="req",
        allocation_generation=3,
    )

    updates = runner._process_diffusion_page_control(
        make_scheduler_output(
            page_releases=(
                PageReleaseRequest(
                    request_id="req",
                    allocation_generation=3,
                ),
            )
        )
    )

    assert calls == [("release", "req", 3)]
    assert [(update.request_id, update.status, update.tp_rank) for update in updates] == [("req", "released", 0)]


def test_cached_step_reuses_the_original_page_binding() -> None:
    binding = SimpleNamespace(request_id="req", allocation_generation=7)
    registry = SimpleNamespace(bind_request=Mock(return_value=binding))
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = registry

    first = runner._prepare_diffusion_page_bindings(make_scheduler_output(make_new_request("req", generation=7)))
    cached = runner._prepare_diffusion_page_bindings(make_scheduler_output(cached=("req",)))

    assert cached["req"] is first["req"]
    registry.bind_request.assert_called_once()


def test_scheduled_new_request_reuses_control_installed_page_binding() -> None:
    binding = SimpleNamespace(request_id="req", allocation_generation=7)
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = SimpleNamespace(
        bind_request=Mock(),
        release_request=Mock(),
    )
    runner._diffusion_page_bindings["req"] = binding

    bindings = runner._prepare_diffusion_page_bindings(make_scheduler_output(make_new_request("req", generation=7)))

    assert bindings == {"req": binding}
    assert runner._diffusion_page_bindings == {"req": binding}
    runner.page_registry.bind_request.assert_not_called()
    runner.page_registry.release_request.assert_not_called()


def test_step_preflight_failure_does_not_install_page_binding() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.pipeline = object()
    runner._supports_step_mode = Mock(return_value=False)
    runner.page_registry = SimpleNamespace(bind_request=Mock())

    with pytest.raises(ValueError, match="does not support step execution"):
        runner.execute_stepwise(make_scheduler_output(make_new_request("req", generation=7)))

    runner.page_registry.bind_request.assert_not_called()
    assert runner._diffusion_page_bindings == {}


def test_step_setup_failure_releases_new_page_binding() -> None:
    binding = SimpleNamespace(request_id="req", allocation_generation=7)
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.od_config.cache_backend = "none"
    runner.od_config.parallel_config = SimpleNamespace(use_hsdp=False)
    runner.pipeline = object()
    runner.state_cache = {}
    runner._supports_step_mode = Mock(return_value=True)
    runner._update_states = Mock(side_effect=RuntimeError("step setup failed"))
    runner.page_registry = SimpleNamespace(
        bind_request=Mock(return_value=binding),
        release_request=Mock(),
    )

    with pytest.raises(RuntimeError, match="step setup failed"):
        runner.execute_stepwise(make_scheduler_output(make_new_request("req", generation=7)))

    runner.page_registry.release_request.assert_called_once_with("req", 7)
    assert runner._diffusion_page_bindings == {}


def test_binding_generation_mismatch_fails_before_forward() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = SimpleNamespace(
        bind_request=Mock(
            return_value=SimpleNamespace(
                request_id="req",
                allocation_generation=8,
            )
        ),
        release_request=Mock(),
    )

    with pytest.raises(ValueError, match="binding generation mismatch"):
        runner.install_diffusion_kv_metadata(
            "req",
            make_new_request("req", generation=7).diffusion_kv_metadata,
        )

    runner.page_registry.release_request.assert_called_once_with("req", 8)


def test_independent_tp_ranks_install_the_same_binding_identity() -> None:
    identities = []
    for _ in range(2):
        runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
        runner.page_registry = SimpleNamespace(
            bind_request=lambda metadata: SimpleNamespace(
                request_id=metadata.request_id,
                allocation_generation=metadata.allocation_generation,
            )
        )
        binding = runner.install_diffusion_kv_metadata(
            "req",
            make_new_request("req", generation=11).diffusion_kv_metadata,
        )
        identities.append((binding.request_id, binding.allocation_generation))

    assert identities == [("req", 11), ("req", 11)]


def test_shutdown_releases_pages_before_clearing_registry() -> None:
    calls: list[tuple] = []
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.page_registry = SimpleNamespace(
        release_request=lambda request_id, generation: calls.append(("release", request_id, generation))
    )
    runner._diffusion_page_bindings["req"] = SimpleNamespace(
        request_id="req",
        allocation_generation=5,
    )

    runner.shutdown_kv_cache_data_plane()

    assert calls == [("release", "req", 5)]
    assert runner.page_registry is None


def test_page_debug_snapshot_copies_metrics_and_active_binding_identity() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    metrics = DiffusionPageMetrics(
        stable_pages_requested=3,
        stable_pages_imported=2,
        stable_pages_committed=2,
        transferred_bytes=4096,
        reference_gather_bytes=8192,
        terminal_snapshots=[
            {
                "request_id": "done",
                "allocation_generation": 5,
                "page_count": 2,
                "transferred_bytes": 4096,
            }
        ],
    )
    runner.page_registry = SimpleNamespace(metrics=metrics)
    runner._diffusion_page_bindings["active"] = SimpleNamespace(
        request_id="active",
        allocation_generation=7,
    )

    snapshot = runner.get_diffusion_page_debug_snapshot()

    assert snapshot == {
        "enabled": True,
        "metrics": {
            "stable_pages_requested": 3,
            "stable_pages_imported": 2,
            "stable_pages_committed": 2,
            "transferred_bytes": 4096,
            "local_install_latency_s": 0.0,
            "local_kv_wait_s": 0.0,
            "stale_completions": 0,
            "duplicate_completions": 0,
            "cancellations": 0,
            "timeouts": 0,
            "page_pool_pages_in_use": 0,
            "page_pool_total_pages": 0,
            "page_pool_utilization": 0.0,
            "page_pool_utilization_high_water": 0.0,
            "staging_bytes": 0,
            "staging_bytes_high_water": 0,
            "in_flight_bytes": 0,
            "in_flight_bytes_high_water": 0,
            "reference_gather_bytes": 8192,
            "reference_gather_latency_s": 0.0,
            "terminal_snapshots": [
                {
                    "request_id": "done",
                    "allocation_generation": 5,
                    "page_count": 2,
                    "transferred_bytes": 4096,
                }
            ],
        },
        "active_bindings": [
            {
                "request_id": "active",
                "allocation_generation": 7,
            }
        ],
    }
    snapshot["metrics"]["terminal_snapshots"][0]["page_count"] = 99
    assert metrics.terminal_snapshots[0]["page_count"] == 2


def test_release_records_terminal_binding_snapshot_before_cleanup() -> None:
    runner = make_runner(DiffusionKVCacheMode.PAGED_SCHEDULER)
    metrics = DiffusionPageMetrics()
    runner.page_registry = SimpleNamespace(
        metrics=metrics,
        release_request=Mock(),
    )
    runner._diffusion_page_bindings["req"] = SimpleNamespace(
        request_id="req",
        allocation_generation=9,
        page_states={3: object(), 5: object()},
    )

    runner.release_diffusion_kv_requests({"req"})

    assert metrics.terminal_snapshots[-1] == {
        "request_id": "req",
        "allocation_generation": 9,
        "page_count": 2,
        "transferred_bytes": 0,
        "terminal_status": "released",
    }
    assert runner._diffusion_page_bindings == {}


def test_terminal_snapshot_history_is_bounded() -> None:
    metrics = DiffusionPageMetrics()

    for index in range(metrics.MAX_TERMINAL_SNAPSHOTS + 3):
        metrics.record_terminal_snapshot({"request_id": f"req-{index}"})

    assert len(metrics.terminal_snapshots) == metrics.MAX_TERMINAL_SNAPSHOTS
    assert metrics.terminal_snapshots[0] == {"request_id": "req-3"}
    assert metrics.terminal_snapshots[-1] == {"request_id": f"req-{metrics.MAX_TERMINAL_SNAPSHOTS + 2}"}


def test_dense_page_debug_snapshot_is_disabled_without_metrics() -> None:
    runner = make_runner(DiffusionKVCacheMode.DENSE_LEGACY)

    assert runner.get_diffusion_page_debug_snapshot() == {
        "enabled": False,
        "metrics": None,
        "active_bindings": [],
    }


def test_worker_page_debug_snapshot_gathers_every_rank(monkeypatch) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 3
    worker.model_runner = SimpleNamespace(
        get_diffusion_page_debug_snapshot=lambda: {
            "enabled": True,
            "metrics": {"stable_pages_committed": 2},
            "active_bindings": [],
        }
    )
    gathered = [
        {
            "rank": 3,
            "enabled": True,
            "metrics": {"stable_pages_committed": 2},
            "active_bindings": [],
        },
        {
            "rank": 4,
            "enabled": True,
            "metrics": {"stable_pages_committed": 2},
            "active_bindings": [],
        },
    ]
    monkeypatch.setattr(
        worker_module,
        "_run_and_gather_rank_values",
        lambda operation, func: gathered if operation == "diffusion page debug snapshot" else func(),
    )

    assert worker.get_diffusion_page_debug_snapshots() == gathered


def test_worker_page_debug_snapshot_reports_parallel_coordinates(
    monkeypatch,
) -> None:
    worker = object.__new__(DiffusionWorker)
    worker.rank = 3
    worker.model_runner = SimpleNamespace(
        get_diffusion_page_debug_snapshot=lambda: {
            "enabled": True,
            "metrics": {"stable_pages_committed": 2},
            "active_bindings": [],
        }
    )
    monkeypatch.setattr(
        worker_module,
        "_run_and_gather_rank_values",
        lambda operation, func: [func()],
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.distributed.parallel_state.get_data_parallel_rank",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tensor_model_parallel_rank",
        lambda: 1,
    )

    assert worker.get_diffusion_page_debug_snapshots() == [
        {
            "rank": 3,
            "data_parallel_rank": 1,
            "tensor_parallel_rank": 1,
            "enabled": True,
            "metrics": {"stable_pages_committed": 2},
            "active_bindings": [],
        }
    ]


def test_worker_shutdown_closes_page_data_plane_before_legacy_prefetch(
    monkeypatch,
) -> None:
    calls: list[str] = []
    worker = object.__new__(DiffusionWorker)
    worker.model_runner = SimpleNamespace(
        shutdown_kv_cache_data_plane=lambda: calls.append("page-data-plane"),
        kv_transfer_manager=SimpleNamespace(
            shutdown_prefetch=lambda: calls.append("legacy-prefetch"),
        ),
    )
    monkeypatch.setattr(
        worker_module,
        "destroy_distributed_env",
        lambda: calls.append("distributed-env"),
    )

    worker.shutdown()

    assert calls == [
        "page-data-plane",
        "legacy-prefetch",
        "distributed-env",
    ]


def test_forward_context_exposes_diffusion_page_bindings() -> None:
    binding = object()

    with set_forward_context(diffusion_page_bindings={"req": binding}):
        assert get_forward_context().diffusion_page_bindings == {"req": binding}


def test_forward_context_exposes_diffusion_page_metrics() -> None:
    metrics = DiffusionPageMetrics()

    with set_forward_context(diffusion_page_metrics=metrics):
        assert get_forward_context().diffusion_page_metrics is metrics
