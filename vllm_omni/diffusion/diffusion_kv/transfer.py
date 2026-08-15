# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import torch

from vllm_omni.diffusion.data import DiffusionPageMetrics
from vllm_omni.diffusion.diffusion_kv.page import DiffusionPageBinding, PageState

TransferIdentity = tuple[str, int, int, int]


@dataclass(frozen=True, slots=True)
class PageEndpoint:
    stage_id: str
    replica_id: int
    tp_rank: int
    cache_role: str
    layer_name: str
    kv_kind: Literal["key", "value"]
    block_id: int
    byte_offset: int
    num_bytes: int
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device_type: str

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id must be non-empty")
        if self.replica_id < 0 or self.tp_rank < 0:
            raise ValueError("replica_id and tp_rank must be non-negative")
        if not self.cache_role or not self.layer_name:
            raise ValueError("cache_role and layer_name must be non-empty")
        if self.kv_kind not in {"key", "value"}:
            raise ValueError(f"unsupported kv_kind {self.kv_kind!r}")
        if self.block_id < 0 or self.byte_offset < 0:
            raise ValueError("block_id and byte_offset must be non-negative")
        if self.num_bytes <= 0:
            raise ValueError("num_bytes must be positive")
        if not self.dtype or not self.shape or len(self.shape) != len(self.stride):
            raise ValueError("endpoint dtype, shape, and stride must describe one tensor span")
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("endpoint shape dimensions must be positive")
        if any(stride < 0 for stride in self.stride):
            raise ValueError("endpoint strides must be non-negative")
        if not self.device_type:
            raise ValueError("device_type must be non-empty")


@dataclass(frozen=True, slots=True)
class PageCopy:
    source: PageEndpoint
    destination: PageEndpoint


@dataclass(frozen=True, slots=True)
class PageTransferPlan:
    session_id: str
    request_id: str
    allocation_generation: int
    route_epoch: int
    op_id: int
    source_stage: str
    target_stage: str
    source_tp_rank: int
    target_tp_rank: int
    pages: tuple[PageCopy, ...]
    deadline_monotonic_s: float

    def __post_init__(self) -> None:
        if not self.session_id or not self.request_id:
            raise ValueError("session_id and request_id must be non-empty")
        if self.allocation_generation <= 0:
            raise ValueError("allocation_generation must be positive")
        if self.route_epoch < 0 or self.op_id < 0:
            raise ValueError("route_epoch and op_id must be non-negative")
        if not self.source_stage or not self.target_stage:
            raise ValueError("source_stage and target_stage must be non-empty")
        if self.source_tp_rank < 0 or self.target_tp_rank < 0:
            raise ValueError("source_tp_rank and target_tp_rank must be non-negative")
        if not self.pages:
            raise ValueError("page transfer plan must contain at least one page")
        if self.deadline_monotonic_s <= 0:
            raise ValueError("deadline_monotonic_s must be positive")

    @property
    def identity(self) -> TransferIdentity:
        return (
            self.session_id,
            self.allocation_generation,
            self.route_epoch,
            self.op_id,
        )


@dataclass(frozen=True, slots=True)
class PageTransferResult:
    identity: TransferIdentity
    status: str
    completed_pages: tuple[int, ...] = ()
    message: str = ""


@dataclass(frozen=True, slots=True)
class PageReceiveReservation:
    identity: TransferIdentity


class TransferState(str, Enum):
    ALLOCATED = "allocated"
    TARGET_READY = "target_ready"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    FAILED = "failed"
    STALE = "stale"


class _ImmediateEvent:
    def query(self) -> bool:
        return True


@dataclass(slots=True)
class _TransferSession:
    plan: PageTransferPlan
    state: TransferState
    created_monotonic_s: float
    target_ready_monotonic_s: float | None = None
    transfer_started_monotonic_s: float | None = None
    staging_bytes: int = 0
    in_flight_bytes: int = 0
    event: object | None = None
    result: PageTransferResult | None = None


class PageTransferSessionManager:
    def __init__(
        self,
        *,
        registry,
        timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
        event_factory: Callable[[], object] | None = None,
        copy_fn: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
        metrics: DiffusionPageMetrics | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.registry = registry
        self.timeout_s = timeout_s
        self._clock = clock
        self._event_factory = event_factory
        self._copy_fn = copy_fn
        self.metrics = metrics or getattr(registry, "metrics", None) or DiffusionPageMetrics()
        self._bindings: dict[str, DiffusionPageBinding] = {}
        self._endpoints: dict[PageEndpoint, torch.Tensor] = {}
        self._sessions: dict[TransferIdentity, _TransferSession] = {}
        self._session_identities: dict[str, set[TransferIdentity]] = {}
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}

    def register_binding(self, binding: DiffusionPageBinding) -> None:
        self._bindings[binding.request_id] = binding

    def unregister_binding(self, request_id: str, allocation_generation: int) -> None:
        binding = self._bindings.get(request_id)
        if binding is None:
            return
        if binding.allocation_generation != allocation_generation:
            raise ValueError(
                f"stale binding unregister for request {request_id!r}: "
                f"active={binding.allocation_generation}, requested={allocation_generation}"
            )
        self._bindings.pop(request_id, None)

    def register_endpoint(self, endpoint: PageEndpoint, tensor: torch.Tensor) -> None:
        self._validate_registered_tensor(endpoint, tensor)
        previous = self._endpoints.get(endpoint)
        if previous is not None and previous is not tensor:
            raise ValueError(f"endpoint is already registered to a different tensor: {endpoint!r}")
        self._endpoints[endpoint] = tensor

    def unregister_endpoint(self, endpoint: PageEndpoint) -> None:
        self._endpoints.pop(endpoint, None)

    def resolve_endpoint(self, endpoint: PageEndpoint) -> torch.Tensor:
        try:
            return self._endpoints[endpoint]
        except KeyError as exc:
            raise KeyError(f"missing endpoint registration: {endpoint!r}") from exc

    def open_session(self, plan: PageTransferPlan) -> TransferIdentity:
        existing = self._sessions.get(plan.identity)
        if existing is not None:
            if existing.plan != plan:
                raise ValueError(f"transfer identity {plan.identity!r} was reused with a different plan")
            return plan.identity
        self._validate_plan_route(plan)
        self._sessions[plan.identity] = _TransferSession(
            plan=plan,
            state=TransferState.ALLOCATED,
            created_monotonic_s=self._clock(),
        )
        self._session_identities.setdefault(plan.session_id, set()).add(plan.identity)
        return plan.identity

    def prepare_receive(self, plan: PageTransferPlan) -> PageReceiveReservation:
        identity = self.open_session(plan)
        session = self._sessions[identity]
        if session.result is not None:
            return PageReceiveReservation(identity)
        if session.state is TransferState.TRANSFERRING:
            return PageReceiveReservation(identity)
        if session.state is TransferState.TARGET_READY:
            return PageReceiveReservation(identity)
        for page_copy in plan.pages:
            self._validate_copy_geometry(page_copy)
            if page_copy.destination not in self._endpoints:
                raise KeyError(f"missing destination endpoint registration: {page_copy.destination!r}")
            source = self._endpoints.get(page_copy.source)
            destination = self._endpoints[page_copy.destination]
            if source is not None and self._tensors_overlap(source, destination):
                raise ValueError("source and destination page spans overlap")
        session.state = TransferState.TARGET_READY
        session.target_ready_monotonic_s = self._clock()
        session.staging_bytes = self._plan_num_bytes(plan)
        self.metrics.add_staging_bytes(session.staging_bytes)
        return PageReceiveReservation(identity)

    def send_pages(
        self,
        plan: PageTransferPlan,
        reservation: PageReceiveReservation,
    ) -> TransferIdentity:
        if reservation.identity != plan.identity:
            raise ValueError("receive reservation does not match transfer plan identity")
        session = self._sessions.get(plan.identity)
        if session is None:
            raise KeyError(f"transfer session {plan.identity!r} is not allocated")
        if session.result is not None:
            self.metrics.duplicate_completions += 1
            return plan.identity
        if session.state is not TransferState.TARGET_READY:
            raise ValueError(f"transfer session is not target-ready: {session.state.value}")

        transfer_started = self._clock()
        session.state = TransferState.TRANSFERRING
        session.transfer_started_monotonic_s = transfer_started
        if session.target_ready_monotonic_s is not None:
            self.metrics.local_kv_wait_s += max(
                0.0,
                transfer_started - session.target_ready_monotonic_s,
            )
        session.in_flight_bytes = self._plan_num_bytes(plan)
        self.metrics.add_in_flight_bytes(session.in_flight_bytes)
        copy_stream = None
        try:
            for page_copy in plan.pages:
                self._validate_copy_geometry(page_copy)
                try:
                    source = self._endpoints[page_copy.source]
                except KeyError as exc:
                    raise KeyError(f"missing source endpoint registration: {page_copy.source!r}") from exc
                try:
                    destination = self._endpoints[page_copy.destination]
                except KeyError as exc:
                    raise KeyError(f"missing destination endpoint registration: {page_copy.destination!r}") from exc
                if self._tensors_overlap(source, destination):
                    raise ValueError("source and destination page spans overlap")
                page_copy_stream = self._copy_tensor(destination, source)
                if page_copy_stream is not None:
                    if copy_stream is not None and page_copy_stream is not copy_stream:
                        raise ValueError("one page transfer plan must target a single CUDA device")
                    copy_stream = page_copy_stream
            session.event = self._record_completion_event(
                plan,
                copy_stream=copy_stream,
            )
        except Exception as exc:
            self._set_terminal(
                session,
                TransferState.FAILED,
                message=str(exc),
            )
        return plan.identity

    def poll_completion(
        self,
        identity: TransferIdentity | PageReceiveReservation,
    ) -> PageTransferResult | None:
        session = self._get_session(identity)
        if session.result is not None:
            self.metrics.duplicate_completions += 1
            return session.result
        if session.state is not TransferState.TRANSFERRING:
            return None
        deadline = min(
            session.plan.deadline_monotonic_s,
            session.created_monotonic_s + self.timeout_s,
        )
        if self._clock() >= deadline:
            return self._set_terminal(session, TransferState.TIMED_OUT)
        if session.event is None or not session.event.query():
            return None
        return self._complete_session(session)

    def complete_for_test(
        self,
        identity: TransferIdentity | PageReceiveReservation,
    ) -> PageTransferResult:
        session = self._get_session(identity)
        binding = self._bindings.get(session.plan.request_id)
        if binding is None or binding.allocation_generation != session.plan.allocation_generation:
            return self._set_terminal(session, TransferState.STALE)
        if session.result is not None:
            self.metrics.duplicate_completions += 1
            return session.result
        return self._complete_session(session)

    def complete_external(
        self,
        identity: TransferIdentity | PageReceiveReservation,
    ) -> PageTransferResult:
        session = self._get_session(identity)
        if session.result is not None:
            self.metrics.duplicate_completions += 1
            return session.result
        if session.state is not TransferState.TARGET_READY:
            raise ValueError(f"external page copy can complete only from target_ready, got {session.state.value}")
        return self._complete_session(session)

    def fail_external(
        self,
        identity: TransferIdentity | PageReceiveReservation,
        message: str,
    ) -> PageTransferResult:
        session = self._get_session(identity)
        if session.result is not None:
            self.metrics.duplicate_completions += 1
            return session.result
        return self._set_terminal(session, TransferState.FAILED, message=message)

    def cancel(
        self,
        identity: TransferIdentity | PageReceiveReservation,
    ) -> PageTransferResult:
        session = self._get_session(identity)
        if session.result is not None:
            self.metrics.duplicate_completions += 1
            return session.result
        self._wait_for_copy_completion(session)
        return self._set_terminal(session, TransferState.CANCELLED)

    def record_reference_gather(
        self,
        *,
        num_bytes: int,
        latency_s: float,
    ) -> None:
        self.metrics.record_reference_gather(
            num_bytes=num_bytes,
            latency_s=latency_s,
        )

    def close_session(self, session_id: str) -> None:
        for identity in tuple(self._session_identities.get(session_id, ())):
            session = self._sessions[identity]
            if session.result is None:
                self.cancel(identity)

    def close(self) -> None:
        for session_id in tuple(self._session_identities):
            self.close_session(session_id)
        self._endpoints.clear()
        self._bindings.clear()
        self._copy_streams.clear()

    def state(
        self,
        identity: TransferIdentity | PageReceiveReservation,
    ) -> TransferState:
        return self._get_session(identity).state

    def _complete_session(self, session: _TransferSession) -> PageTransferResult:
        binding = self._bindings.get(session.plan.request_id)
        if binding is None or binding.allocation_generation != session.plan.allocation_generation:
            return self._set_terminal(session, TransferState.STALE)

        completed_pages = tuple(dict.fromkeys(page_copy.destination.block_id for page_copy in session.plan.pages))
        for block_id in completed_pages:
            try:
                state = binding.page_states[block_id]
            except KeyError:
                return self._set_terminal(
                    session,
                    TransferState.FAILED,
                    message=f"destination block {block_id} is not bound to request {binding.request_id!r}",
                )
            if state is PageState.RESERVED:
                binding.transition_page(block_id, PageState.INSTALLING_LOCAL)
                state = PageState.INSTALLING_LOCAL
            if state in {PageState.INSTALLING_LOCAL, PageState.INSTALLING_REMOTE}:
                binding.transition_page(block_id, PageState.COMMITTED)
            elif state is not PageState.COMMITTED:
                return self._set_terminal(
                    session,
                    TransferState.FAILED,
                    message=f"destination block {block_id} cannot commit from state {state.value}",
                )
        return self._set_terminal(
            session,
            TransferState.COMPLETED,
            completed_pages=completed_pages,
        )

    def _set_terminal(
        self,
        session: _TransferSession,
        state: TransferState,
        *,
        completed_pages: tuple[int, ...] = (),
        message: str = "",
    ) -> PageTransferResult:
        completed_at = self._clock()
        if session.staging_bytes:
            self.metrics.remove_staging_bytes(session.staging_bytes)
            session.staging_bytes = 0
        if session.in_flight_bytes:
            self.metrics.remove_in_flight_bytes(session.in_flight_bytes)
            session.in_flight_bytes = 0

        page_count = len({page_copy.destination.block_id for page_copy in session.plan.pages})
        transferred_bytes = 0
        if state is TransferState.COMPLETED:
            transferred_bytes = self._plan_num_bytes(session.plan)
            self.metrics.stable_pages_imported += page_count
            self.metrics.stable_pages_committed += page_count
            self.metrics.transferred_bytes += transferred_bytes
            if session.transfer_started_monotonic_s is not None:
                self.metrics.local_install_latency_s += max(
                    0.0,
                    completed_at - session.transfer_started_monotonic_s,
                )
        elif state is TransferState.STALE:
            self.metrics.stale_completions += 1
        elif state is TransferState.CANCELLED:
            self.metrics.cancellations += 1
        elif state is TransferState.TIMED_OUT:
            self.metrics.timeouts += 1

        session.state = state
        session.result = PageTransferResult(
            identity=session.plan.identity,
            status=state.value,
            completed_pages=completed_pages,
            message=message,
        )
        self.metrics.terminal_snapshots.append(
            {
                "request_id": session.plan.request_id,
                "allocation_generation": session.plan.allocation_generation,
                "session_id": session.plan.session_id,
                "tp_rank": session.plan.target_tp_rank,
                "page_count": page_count,
                "transferred_bytes": transferred_bytes,
                "terminal_status": state.value,
            }
        )
        return session.result

    @staticmethod
    def _plan_num_bytes(plan: PageTransferPlan) -> int:
        return sum(page_copy.destination.num_bytes for page_copy in plan.pages)

    def _get_session(
        self,
        identity: TransferIdentity | PageReceiveReservation,
    ) -> _TransferSession:
        identity = self._normalize_identity(identity)
        try:
            return self._sessions[identity]
        except KeyError as exc:
            raise KeyError(f"unknown page transfer identity {identity!r}") from exc

    @staticmethod
    def _normalize_identity(
        identity: TransferIdentity | PageReceiveReservation,
    ) -> TransferIdentity:
        if isinstance(identity, PageReceiveReservation):
            return identity.identity
        return identity

    def _validate_plan_route(self, plan: PageTransferPlan) -> None:
        for page_copy in plan.pages:
            source = page_copy.source
            destination = page_copy.destination
            if source.stage_id != plan.source_stage or source.tp_rank != plan.source_tp_rank:
                raise ValueError("source endpoint does not match transfer plan route")
            if destination.stage_id != plan.target_stage or destination.tp_rank != plan.target_tp_rank:
                raise ValueError("destination endpoint does not match transfer plan route")

    def _validate_copy_geometry(self, page_copy: PageCopy) -> None:
        source = page_copy.source
        destination = page_copy.destination
        comparable = (
            "cache_role",
            "layer_name",
            "kv_kind",
            "num_bytes",
            "dtype",
            "shape",
            "stride",
            "device_type",
        )
        mismatches = [name for name in comparable if getattr(source, name) != getattr(destination, name)]
        if mismatches:
            raise ValueError(f"source/destination endpoint geometry mismatch: {mismatches}")

    def _validate_registered_tensor(self, endpoint: PageEndpoint, tensor: torch.Tensor) -> None:
        tensor_num_bytes = tensor.numel() * tensor.element_size()
        if endpoint.num_bytes != tensor_num_bytes:
            raise ValueError(
                f"endpoint byte count does not match tensor: endpoint={endpoint.num_bytes}, tensor={tensor_num_bytes}"
            )
        actual = {
            "dtype": str(tensor.dtype),
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "device_type": tensor.device.type,
        }
        expected = {
            "dtype": endpoint.dtype,
            "shape": endpoint.shape,
            "stride": endpoint.stride,
            "device_type": endpoint.device_type,
        }
        mismatches = [name for name in expected if expected[name] != actual[name]]
        if mismatches:
            raise ValueError(f"endpoint tensor geometry mismatch: {mismatches}")

    def _copy_tensor(
        self,
        destination: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.cuda.Stream | None:
        if self._copy_fn is not None:
            self._copy_fn(destination, source)
            return None
        if source.device.type == "cuda":
            device = torch.device(destination.device)
            stream = self._copy_streams.get(device)
            if stream is None:
                stream = torch.cuda.Stream(device=device)
                self._copy_streams[device] = stream
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                destination.copy_(source, non_blocking=True)
                source.record_stream(stream)
                destination.record_stream(stream)
            return stream
        destination.copy_(source)
        return None

    def _record_completion_event(
        self,
        plan: PageTransferPlan,
        *,
        copy_stream=None,
    ):
        if self._event_factory is not None:
            event = self._event_factory()
        elif any(page_copy.destination.device_type == "cuda" for page_copy in plan.pages):
            event = torch.cuda.Event()
        else:
            return _ImmediateEvent()
        if copy_stream is not None:
            copy_stream.record_event(event)
        elif any(page_copy.destination.device_type == "cuda" for page_copy in plan.pages):
            torch.cuda.current_stream().record_event(event)
        return event

    @staticmethod
    def _wait_for_copy_completion(session: _TransferSession) -> None:
        if session.state is not TransferState.TRANSFERRING or session.event is None:
            return
        synchronize = getattr(session.event, "synchronize", None)
        if callable(synchronize):
            synchronize()

    @staticmethod
    def _tensors_overlap(source: torch.Tensor, destination: torch.Tensor) -> bool:
        source_start, source_end = PageTransferSessionManager._tensor_byte_interval(source)
        destination_start, destination_end = PageTransferSessionManager._tensor_byte_interval(destination)
        return source_start < destination_end and destination_start < source_end

    @staticmethod
    def _tensor_byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
        base = tensor.untyped_storage().data_ptr()
        element_size = tensor.element_size()
        start = base + tensor.storage_offset() * element_size
        max_element_offset = sum((dimension - 1) * stride for dimension, stride in zip(tensor.shape, tensor.stride()))
        return start, start + (max_element_offset + 1) * element_size
