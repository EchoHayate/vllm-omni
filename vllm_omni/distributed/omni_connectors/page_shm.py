# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import struct
import tempfile
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory as shm_pkg

import torch

from vllm_omni.diffusion.diffusion_kv.transfer import (
    PageEndpoint,
    PageTransferPlan,
    PageTransferResult,
    PageTransferSessionManager,
    TransferIdentity,
)
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import (
    SharedMemoryConnector,
)

_HEADER_LENGTH = struct.Struct("!Q")


@dataclass(frozen=True, slots=True)
class PageShmRegistration:
    name: str
    size: int
    session_id: str
    op_id: int


@dataclass(frozen=True, slots=True)
class PageShmPayload:
    header_size: int
    total_size: int
    page_count: int


class SharedMemoryPageAdapter:
    def __init__(
        self,
        *,
        connector: SharedMemoryConnector,
        transfer_manager: PageTransferSessionManager,
    ) -> None:
        self.connector = connector
        self.transfer_manager = transfer_manager
        self._registrations: dict[str, PageShmRegistration] = {}
        self._payloads: dict[TransferIdentity, PageShmPayload] = {}
        self._results: dict[TransferIdentity, PageTransferResult] = {}

    def put_pages(
        self,
        plan: PageTransferPlan,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[PageShmRegistration, PageShmPayload]:
        if len(tensors) != len(plan.pages):
            raise ValueError(
                f"page tensor count does not match transfer plan: tensors={len(tensors)}, pages={len(plan.pages)}"
            )
        existing_payload = self._payloads.get(plan.identity)
        if existing_payload is not None:
            registration = next(
                registration
                for registration in self._registrations.values()
                if registration.session_id == plan.session_id and registration.op_id == plan.op_id
            )
            return registration, existing_payload

        for page_copy, tensor in zip(plan.pages, tensors):
            self._validate_tensor_geometry(page_copy.source, tensor)

        actual_total = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
        reserved_total = sum(page.source.num_bytes for page in plan.pages)
        if actual_total > reserved_total:
            raise ValueError(f"page payload size {actual_total} exceeds reserved source spans {reserved_total}")

        raw_spans: list[bytes] = []
        spans: list[dict[str, int]] = []
        payload_offset = 0
        for page_copy, tensor in zip(plan.pages, tensors):
            raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
            if len(raw) != page_copy.source.num_bytes:
                raise ValueError(
                    f"page payload byte count does not match reserved span: "
                    f"payload={len(raw)}, reserved={page_copy.source.num_bytes}"
                )
            spans.append({"offset": payload_offset, "num_bytes": len(raw)})
            raw_spans.append(raw)
            payload_offset += len(raw)

        header = json.dumps(
            {
                "identity": list(plan.identity),
                "page_count": len(raw_spans),
                "spans": spans,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        total_size = _HEADER_LENGTH.size + len(header) + payload_offset
        name = self._segment_name(plan.identity)
        registration = PageShmRegistration(
            name=name,
            size=total_size,
            session_id=plan.session_id,
            op_id=plan.op_id,
        )
        payload = PageShmPayload(
            header_size=len(header),
            total_size=total_size,
            page_count=len(raw_spans),
        )

        segment = None
        lock_file = self._lock_file_path(name)
        try:
            segment = shm_pkg.SharedMemory(name=name, create=True, size=total_size)
            with open(lock_file, "wb+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                segment.buf[: _HEADER_LENGTH.size] = _HEADER_LENGTH.pack(len(header))
                header_start = _HEADER_LENGTH.size
                segment.buf[header_start : header_start + len(header)] = header
                data_start = header_start + len(header)
                for span, raw in zip(spans, raw_spans):
                    start = data_start + span["offset"]
                    segment.buf[start : start + span["num_bytes"]] = raw
                fcntl.flock(lock, fcntl.LOCK_UN)
            self.connector._track_pending_key(name)
            self._registrations[name] = registration
            self._payloads[plan.identity] = payload
            return registration, payload
        except Exception:
            if segment is not None:
                segment.close()
                try:
                    segment.unlink()
                except FileNotFoundError:
                    pass
            self._remove_lock_file(lock_file)
            raise
        finally:
            if segment is not None:
                segment.close()

    def copy_into_pages(
        self,
        plan: PageTransferPlan,
        registration: PageShmRegistration,
    ) -> PageTransferResult:
        existing = self._results.get(plan.identity)
        if existing is not None:
            return existing
        self._validate_registration(plan, registration)
        reservation = self.transfer_manager.prepare_receive(plan)
        self.connector._track_pending_key(registration.name)
        self._registrations[registration.name] = registration

        segment = None
        lock_file = self._lock_file_path(registration.name)
        try:
            segment = shm_pkg.SharedMemory(name=registration.name)
            if segment.size < registration.size:
                raise ValueError(
                    f"SHM segment is smaller than registration: registration={registration.size}, actual={segment.size}"
                )
            with open(lock_file, "rb+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                packed = bytes(segment.buf[: registration.size])
                fcntl.flock(lock, fcntl.LOCK_UN)
            spans, data_start = self._parse_payload(plan, packed)
            for page_copy, span in zip(plan.pages, spans):
                destination = self.transfer_manager.resolve_endpoint(page_copy.destination)
                raw = bytearray(packed[data_start + span["offset"] : data_start + span["offset"] + span["num_bytes"]])
                source = torch.frombuffer(
                    raw,
                    dtype=destination.dtype,
                    count=destination.numel(),
                ).reshape(destination.shape)
                destination.copy_(source)
            result = self.transfer_manager.complete_external(reservation)
            self._results[plan.identity] = result
            return result
        except Exception as exc:
            self.transfer_manager.fail_external(reservation, str(exc))
            raise
        finally:
            if segment is not None:
                segment.close()
            self._cleanup_registration(registration)

    def cancel(
        self,
        plan: PageTransferPlan,
        registration: PageShmRegistration,
    ) -> PageTransferResult:
        identity = self.transfer_manager.open_session(plan)
        result = self.transfer_manager.cancel(identity)
        self._results[plan.identity] = result
        self._cleanup_registration(registration)
        return result

    def close(self) -> None:
        for registration in tuple(self._registrations.values()):
            self._cleanup_registration(registration)

    def _parse_payload(
        self,
        plan: PageTransferPlan,
        packed: bytes,
    ) -> tuple[list[dict[str, int]], int]:
        if len(packed) < _HEADER_LENGTH.size:
            raise ValueError("SHM page payload is smaller than the header prefix")
        (header_size,) = _HEADER_LENGTH.unpack(packed[: _HEADER_LENGTH.size])
        header_start = _HEADER_LENGTH.size
        data_start = header_start + header_size
        if data_start > len(packed):
            raise ValueError("SHM page payload header exceeds segment size")
        header = json.loads(packed[header_start:data_start])
        if tuple(header.get("identity", ())) != plan.identity:
            raise ValueError("SHM page payload transfer identity mismatch")
        spans = header.get("spans")
        if header.get("page_count") != len(plan.pages) or not isinstance(spans, list):
            raise ValueError("SHM page payload page count mismatch")
        if len(spans) != len(plan.pages):
            raise ValueError("SHM page payload span count mismatch")
        for page_copy, span in zip(plan.pages, spans):
            if set(span) != {"offset", "num_bytes"}:
                raise ValueError("SHM page payload span has unexpected fields")
            if span["offset"] < 0 or span["num_bytes"] != page_copy.destination.num_bytes:
                raise ValueError("SHM page payload span does not match destination reservation")
            if data_start + span["offset"] + span["num_bytes"] > len(packed):
                raise ValueError("SHM page payload span exceeds segment size")
        return spans, data_start

    @staticmethod
    def _validate_tensor_geometry(endpoint: PageEndpoint, tensor: torch.Tensor) -> None:
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
        mismatches = [name for name in expected if actual[name] != expected[name]]
        if mismatches:
            raise ValueError(f"page tensor geometry mismatch: {mismatches}")
        actual_num_bytes = tensor.numel() * tensor.element_size()
        if actual_num_bytes != endpoint.num_bytes:
            raise ValueError(
                f"page tensor byte count does not match reserved span: "
                f"tensor={actual_num_bytes}, reserved={endpoint.num_bytes}"
            )

    @staticmethod
    def _validate_registration(
        plan: PageTransferPlan,
        registration: PageShmRegistration,
    ) -> None:
        if registration.session_id != plan.session_id or registration.op_id != plan.op_id:
            raise ValueError("SHM registration does not match transfer plan")
        if registration.size <= _HEADER_LENGTH.size:
            raise ValueError("SHM registration size is too small")

    def _cleanup_registration(self, registration: PageShmRegistration) -> None:
        try:
            segment = shm_pkg.SharedMemory(name=registration.name)
            segment.close()
            segment.unlink()
        except FileNotFoundError:
            pass
        self._remove_lock_file(self._lock_file_path(registration.name))
        self.connector._discard_pending_key(registration.name)
        self._registrations.pop(registration.name, None)

    @staticmethod
    def _segment_name(identity: TransferIdentity) -> str:
        digest = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:8]
        return f"vp_{os.getpid():x}_{digest}_{uuid.uuid4().hex[:4]}"

    @staticmethod
    def _lock_file_path(name: str) -> str:
        root = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        return os.path.join(root, f"shm_{name}_lockfile.lock")

    @staticmethod
    def _remove_lock_file(path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
