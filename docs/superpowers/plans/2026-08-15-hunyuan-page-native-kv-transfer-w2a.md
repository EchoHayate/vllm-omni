# HunyuanImage3 Page-Native KV Transfer W2a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the W2a local page-native data plane for HunyuanImage3 so Scheduler-owned block IDs bind to real rank-local Worker KV pages, Hunyuan attention persists stable prefix K/V in those pages, local imported pages become visible only after completion, and dense mode remains unchanged.

**Architecture:** Stack on PR #6094 or its merged equivalent, then add a Worker-owned page registry, immutable page-transfer descriptors, a completion-aware local transfer session manager, and a Hunyuan paged reference adapter. Physical pages are the only persistent K/V storage in `paged_scheduler`; the initial CUDA reference backend may gather committed page spans into per-forward scratch tensors for Torch SDPA, but it may not create a second request-owned long-lived cache. Scheduler readiness is split from allocation so waiting transfers do not head-of-line block already-ready work.

**Tech Stack:** Python 3.12, PyTorch CUDA streams/events, native vLLM `KVCacheConfig`/`KVCacheSpec`/`BlockTable`, vLLM-Omni diffusion scheduler and step runner, POSIX shared memory, pytest, Buildkite H100 GPU jobs.

## Global Constraints

- Start implementation from the latest reviewed PR #6094 head or from `main` after #6094 merges; do not copy a stale subset of #6094 into a second allocator.
- Pilot only `HunyuanImage3`; other diffusion models must fail capability validation in `paged_scheduler`.
- The Scheduler is the sole owner of allocation, block identity, admission, refcounts, and terminal release.
- Each Worker owns rank-local physical tensors, page views, native BlockTables, slot mappings, copy streams, and completion events.
- Every operation carries `(request_id, allocation_generation)`; stale completion may clean resources but may never commit a new allocation.
- Externally installed pages are attention-visible only after all K/V copies complete and the Worker commits the page.
- Locally produced K/V may be written and consumed in the same ordered forward, but stable pages must be committed before a later denoising step, cache lookup, or transfer reuses them.
- Stable prefix pages may persist across steps; dynamic target slots are overwritten every step and are never published as reusable prefix pages.
- `dense_legacy` remains the default and must not initialize, call, or depend on any W2a component.
- `paged_scheduler` must fail at startup for unsupported model, backend, platform, connector, or sequence-parallel topology; it must never silently execute dense.
- CFG branches share pages only when complete leading blocks have identical semantic hash chains; W2a may conservatively allocate distinct pages when hashes are absent.
- Active TP ranks must agree on public request ID, allocation generation, sequence geometry, and readiness before compute.
- W2a supports same-process direct copy and local SharedMemory page installation only. Mooncake receiver-driven missing-only transfer belongs to W2b.
- Persistent K/V storage must be the Worker page pool. Per-forward dense scratch used by the reference attention adapter is allowed only as a bounded functional bridge and must be measured explicitly.
- No helper or routing microbenchmark may be presented as end-to-end evidence.
- Formal distributed acceptance requires the repository's H100 gate; an A100 or local smoke run is supplemental evidence only.
- Do not implement in `/Users/bytedance/Desktop/vllm-omni` while it is on `feat/llama-omni2-support`.

## Baseline and Branch Setup

The written design was reviewed against:

```text
origin/main:    555336e887da035aa473754d3f87ca7d8eb878a4
PR #6094 head:  7e2245e22c5d5df6497a0740873f261f20a7f7ca
design commit:  51934f90a854a4b21bc4d2b34f67c5d1663afaee
```

Before implementation, refresh the refs and create a new implementation worktree:

```bash
git fetch origin main pull/6094/head:refs/remotes/origin/pr/6094
git worktree add /private/tmp/vllm-omni-hunyuan-page-native-w2a \
  -b feat/hunyuan-page-native-kv-w2a origin/pr/6094
```

Expected: the new branch starts at the current #6094 head, not at the documentation-only branch and not at `feat/llama-omni2-support`.

If #6094 has merged, replace `origin/pr/6094` with the first `origin/main` commit containing:

- `vllm_omni/diffusion/diffusion_kv/initialization.py`
- `vllm_omni/diffusion/diffusion_kv/manager.py`
- `DiffusionModelRunner.set_kv_cache_config`
- `NewRequestData.diffusion_kv_metadata`

## File Structure

### New production files

- `vllm_omni/diffusion/diffusion_kv/page.py` — immutable page ranges/endpoints, Worker binding state, page-state transitions, and slot-mapping helpers.
- `vllm_omni/diffusion/diffusion_kv/worker_registry.py` — physical cache allocation, native BlockTable construction, request binding, commit, lookup, and release.
- `vllm_omni/diffusion/diffusion_kv/transfer.py` — transfer plans/results, session state machine, direct-copy implementation, cancellation, timeout, and stale-completion handling.
- `vllm_omni/distributed/omni_connectors/page_shm.py` — packed local page payload writer/reader that copies into registered final page spans without reconstructing model-owned K/V objects.
- `vllm_omni/diffusion/models/hunyuan_image3/paged_kv.py` — Hunyuan stable/dynamic range derivation and paged reference attention adapter.

### Modified production files

- `vllm_omni/diffusion/diffusion_kv/metadata.py` — serialize explicit stable/dynamic ranges and imported-page dependency.
- `vllm_omni/diffusion/diffusion_kv/manager.py` — emit those ranges while retaining native allocation ownership.
- `vllm_omni/diffusion/worker/diffusion_model_runner.py` — create registry/session manager, bind new requests, expose bindings through forward context, commit local writes, and release finished requests.
- `vllm_omni/diffusion/worker/diffusion_worker.py` — route cache initialization and shutdown to the runner without duplicating page ownership.
- `vllm_omni/diffusion/forward_context.py` — expose the current request-to-page binding map to model attention.
- `vllm_omni/diffusion/attention/backends/abstract.py` — add page-native capability fields to the backend contract.
- `vllm_omni/diffusion/attention/layer.py` — validate page-native capability and pass page bindings to Hunyuan attention.
- `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py` — select dense or paged cache path and remove model-owned cache allocation from paged mode.
- `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py` — stop snapshotting/restoring dense prompt K/V in paged mode and propagate request/branch identity.
- `vllm_omni/diffusion/sched/interface.py` — represent allocated-but-not-ready requests and Worker readiness updates.
- `vllm_omni/diffusion/sched/base_scheduler.py` — separate allocation from compute readiness, scan past transfer-waiting requests, and release only after Worker cleanup acknowledgement.
- `vllm_omni/diffusion/diffusion_engine.py` — route Worker readiness/cleanup updates back to the Scheduler.
- `vllm_omni/diffusion/data.py` — add bounded local-transfer and readiness configuration values plus metrics fields.
- `vllm_omni/diffusion/models/hunyuan_image3/request_layout.py` — validate stable/dynamic geometry and attach import intent without serializing dense tensors.

### New tests

- `tests/diffusion/diffusion_kv/test_page.py`
- `tests/diffusion/diffusion_kv/test_worker_registry.py`
- `tests/diffusion/diffusion_kv/test_transfer.py`
- `tests/distributed/omni_connectors/test_page_shm.py`
- `tests/diffusion/models/hunyuan_image3/test_paged_kv.py`
- `tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_paged_step_execution.py`
- `tests/e2e/accuracy/test_hunyuan_image3_paged_kv.py`
- `benchmarks/diffusion/hunyuan_image3_page_native.py`

---

### Task 1: Freeze the #6094 Integration Contract

**Files:**
- Modify: `docs/superpowers/specs/2026-08-15-hunyuan-page-native-kv-transfer-design.md`
- Create: `tests/diffusion/diffusion_kv/test_w2a_control_plane_contract.py`

**Interfaces:**
- Consumes: `DiffusionKVMetadata`, `KVCacheConfig`, `DiffusionModelRunner.set_kv_cache_config`, `NewRequestData.diffusion_kv_metadata`.
- Produces: an executable contract proving W2a is stacked on #6094 and not recreating Scheduler allocation.

- [ ] **Step 1: Write the failing control-plane contract test**

```python
from inspect import signature

from vllm_omni.diffusion.diffusion_kv.manager import DiffusionKVCacheManager
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.sched.interface import NewRequestData
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner


def test_w2a_stacks_on_scheduler_owned_control_plane() -> None:
    assert "allocation_generation" in DiffusionKVMetadata.__dataclass_fields__
    assert "diffusion_kv_metadata" in NewRequestData.__dataclass_fields__
    assert "kv_cache_config" in signature(DiffusionKVCacheManager).parameters
    assert hasattr(DiffusionModelRunner, "set_kv_cache_config")
```

- [ ] **Step 2: Run the test against the refreshed implementation branch**

Run:

```bash
pytest -q tests/diffusion/diffusion_kv/test_w2a_control_plane_contract.py
```

Expected: PASS. If any assertion fails, stop and rebase the implementation plan onto the merged/current #6094 interface before adding W2a code.

- [ ] **Step 3: Record the exact implementation base**

Append this block to the design document, replacing the hashes with command output:

```markdown
## Implementation Base

W2a implementation started from:

- `origin/main`: `555336e887da035aa473754d3f87ca7d8eb878a4`
- control-plane base: `7e2245e22c5d5df6497a0740873f261f20a7f7ca`
- `DiffusionKVMetadata` provider: record the exact hash printed by
  `git log -1 --format=%H -- vllm_omni/diffusion/diffusion_kv/metadata.py`

The implementation consumes the control-plane types directly and does not
duplicate Scheduler allocation.
```

Use:

```bash
git rev-parse origin/main
git rev-parse HEAD
git log -1 --format=%H -- vllm_omni/diffusion/diffusion_kv/metadata.py
```

- [ ] **Step 4: Commit**

```bash
git add \
  docs/superpowers/specs/2026-08-15-hunyuan-page-native-kv-transfer-design.md \
  tests/diffusion/diffusion_kv/test_w2a_control_plane_contract.py
git commit -m "test: freeze diffusion KV control-plane contract

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 2: Define Page Ranges, Bindings, and State Transitions

**Files:**
- Create: `vllm_omni/diffusion/diffusion_kv/page.py`
- Create: `tests/diffusion/diffusion_kv/test_page.py`

**Interfaces:**
- Consumes: Scheduler block IDs grouped by native cache group.
- Produces:
  - `DiffusionPageRange(cache_role, token_start, token_count, block_ids, mutable)`
  - `PageState`
  - `DiffusionSequenceBinding`
  - `DiffusionPageBinding`
  - `build_slot_mapping(block_ids, block_size, token_start, token_count, device)`

- [ ] **Step 1: Write failing value-object and slot-mapping tests**

```python
import pytest
import torch

from vllm_omni.diffusion.diffusion_kv.page import (
    DiffusionPageBinding,
    DiffusionPageRange,
    DiffusionSequenceBinding,
    PageState,
    build_slot_mapping,
)


def test_build_slot_mapping_uses_scheduler_block_ids() -> None:
    slots = build_slot_mapping(
        block_ids=(3, 9),
        block_size=4,
        token_start=1,
        token_count=6,
        device=torch.device("cpu"),
    )
    assert slots.tolist() == [13, 14, 15, 36, 37, 38]


def test_binding_rejects_dynamic_range_marked_immutable() -> None:
    stable = DiffusionPageRange("primary", 0, 4, (1,), mutable=False)
    dynamic = DiffusionPageRange("primary", 4, 4, (2,), mutable=False)
    with pytest.raises(ValueError, match="dynamic range must be mutable"):
        DiffusionSequenceBinding(
            sequence_id=0,
            seq_len=8,
            stable=stable,
            dynamic=dynamic,
            slot_mapping=torch.arange(8),
        )


def test_only_committed_external_pages_are_reusable() -> None:
    binding = DiffusionPageBinding(
        request_id="req",
        allocation_generation=7,
        sequences=(),
        page_states={4: PageState.INSTALLING_LOCAL},
        externally_required=frozenset({4}),
    )
    assert not binding.is_compute_ready
    binding.transition_page(4, PageState.COMMITTED)
    assert binding.is_compute_ready
```

- [ ] **Step 2: Verify the tests fail because the module is absent**

Run:

```bash
pytest -q tests/diffusion/diffusion_kv/test_page.py
```

Expected: collection FAIL with `ModuleNotFoundError: vllm_omni.diffusion.diffusion_kv.page`.

- [ ] **Step 3: Implement the complete page value model**

Create `vllm_omni/diffusion/diffusion_kv/page.py` with these public definitions:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class PageState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    INSTALLING_LOCAL = "installing_local"
    INSTALLING_REMOTE = "installing_remote"
    COMMITTED = "committed"
    RELEASING = "releasing"


_ALLOWED_PAGE_TRANSITIONS = {
    PageState.FREE: {PageState.RESERVED},
    PageState.RESERVED: {
        PageState.INSTALLING_LOCAL,
        PageState.INSTALLING_REMOTE,
        PageState.COMMITTED,
        PageState.RELEASING,
    },
    PageState.INSTALLING_LOCAL: {PageState.COMMITTED, PageState.RELEASING},
    PageState.INSTALLING_REMOTE: {PageState.COMMITTED, PageState.RELEASING},
    PageState.COMMITTED: {PageState.RELEASING},
    PageState.RELEASING: {PageState.FREE},
}


@dataclass(frozen=True, slots=True)
class DiffusionPageRange:
    cache_role: str
    token_start: int
    token_count: int
    block_ids: tuple[int, ...]
    mutable: bool

    def __post_init__(self) -> None:
        if not self.cache_role:
            raise ValueError("cache_role must be non-empty")
        if self.token_start < 0:
            raise ValueError("token_start must be non-negative")
        if self.token_count < 0:
            raise ValueError("token_count must be non-negative")
        if any(block_id < 0 for block_id in self.block_ids):
            raise ValueError("block_ids must be non-negative")


@dataclass(frozen=True, slots=True)
class DiffusionSequenceBinding:
    sequence_id: int
    seq_len: int
    stable: DiffusionPageRange
    dynamic: DiffusionPageRange
    slot_mapping: torch.Tensor

    def __post_init__(self) -> None:
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be non-negative")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.stable.mutable:
            raise ValueError("stable range must be immutable")
        if not self.dynamic.mutable:
            raise ValueError("dynamic range must be mutable")
        if self.stable.token_start != 0:
            raise ValueError("stable range must start at token zero")
        if self.dynamic.token_start != self.stable.token_count:
            raise ValueError("dynamic range must immediately follow stable range")
        if self.dynamic.token_start + self.dynamic.token_count > self.seq_len:
            raise ValueError("stable and dynamic ranges exceed seq_len")
        if self.slot_mapping.ndim != 1 or self.slot_mapping.numel() != self.seq_len:
            raise ValueError("slot_mapping must contain one slot per logical token")


@dataclass(slots=True)
class DiffusionPageBinding:
    request_id: str
    allocation_generation: int
    sequences: tuple[DiffusionSequenceBinding, ...]
    page_states: dict[int, PageState] = field(default_factory=dict)
    externally_required: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.allocation_generation <= 0:
            raise ValueError("allocation_generation must be positive")
        missing = self.externally_required.difference(self.page_states)
        if missing:
            raise ValueError(f"externally required pages lack state: {sorted(missing)}")

    @property
    def is_compute_ready(self) -> bool:
        return all(self.page_states[block_id] is PageState.COMMITTED for block_id in self.externally_required)

    def transition_page(self, block_id: int, target: PageState) -> None:
        current = self.page_states[block_id]
        if target not in _ALLOWED_PAGE_TRANSITIONS[current]:
            raise ValueError(f"invalid page transition {current.value} -> {target.value} for block {block_id}")
        self.page_states[block_id] = target


def build_slot_mapping(
    *,
    block_ids: tuple[int, ...],
    block_size: int,
    token_start: int,
    token_count: int,
    device: torch.device,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if token_start < 0 or token_count < 0:
        raise ValueError("token_start and token_count must be non-negative")
    logical_positions = torch.arange(token_start, token_start + token_count, dtype=torch.long, device=device)
    logical_blocks = torch.div(logical_positions, block_size, rounding_mode="floor")
    if logical_blocks.numel() and int(logical_blocks.max().item()) >= len(block_ids):
        raise ValueError("block_ids do not cover the requested token range")
    block_tensor = torch.tensor(block_ids, dtype=torch.long, device=device)
    return block_tensor[logical_blocks] * block_size + logical_positions.remainder(block_size)
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
pytest -q tests/diffusion/diffusion_kv/test_page.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_omni/diffusion/diffusion_kv/page.py \
  tests/diffusion/diffusion_kv/test_page.py
git commit -m "feat: define diffusion page bindings

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 3: Extend Scheduler Metadata with Explicit Stable and Dynamic Ranges

**Files:**
- Modify: `vllm_omni/diffusion/diffusion_kv/metadata.py`
- Modify: `vllm_omni/diffusion/diffusion_kv/manager.py`
- Modify: `vllm_omni/diffusion/diffusion_kv/request.py`
- Modify: `tests/diffusion/diffusion_kv/test_manager.py`
- Modify: `tests/diffusion/diffusion_kv/test_metadata.py`
- Modify: `tests/diffusion/diffusion_kv/test_request.py`

**Interfaces:**
- Consumes: `DiffusionKVRequest.prefix_len`, `target_len`, `seq_len`, native block IDs.
- Produces:
  - `DiffusionKVSequenceMetadata.page_ranges`
  - `DiffusionKVSequenceMetadata.imported_prefix_token_count`
  - one immutable stable range plus one mutable dynamic range per Hunyuan CFG row.

- [ ] **Step 1: Write failing metadata tests**

Add:

```python
def test_manager_emits_stable_and_dynamic_page_ranges() -> None:
    manager = _manager(8)
    metadata = manager.reserve_request("public", (_request("public", 0),))
    assert metadata is not None
    sequence = metadata.sequences[0]
    assert [(item.token_start, item.token_count, item.mutable) for item in sequence.page_ranges] == [
        (0, 4, False),
        (4, 4, True),
    ]
    assert sequence.page_ranges[0].block_ids == (sequence.block_ids[0][0],)
    assert sequence.page_ranges[1].block_ids == (sequence.block_ids[0][1],)
```

Add a boundary case where `prefix_len=6`, `target_len=2`, and `block_size=4`; expected stable blocks are the first two allocated blocks, while only the first complete block is publication-eligible. Store this distinction as `cacheable_prefix_block_count=1`.

Add:

```python
def test_imported_prefix_must_end_on_a_complete_page() -> None:
    manager = _manager(8)
    request = DiffusionKVRequest(
            "req/diffusion-kv/0",
            sequence_id=0,
            prefix_len=8,
            target_len=4,
            seq_len=12,
            imported_prefix_token_count=6,
        )
    with pytest.raises(DiffusionKVAdmissionError, match="complete cache blocks"):
        manager.reserve_request("req", (request,))
```

- [ ] **Step 2: Run the focused tests**

Run:

```bash
pytest -q \
  tests/diffusion/diffusion_kv/test_manager.py \
  tests/diffusion/diffusion_kv/test_metadata.py
```

Expected: FAIL because `page_ranges` and `cacheable_prefix_block_count` do not exist.

- [ ] **Step 3: Add metadata fields**

Add to `DiffusionKVSequenceMetadata`:

```python
page_ranges: tuple[DiffusionPageRange, ...] = ()
cacheable_prefix_block_count: int = 0
imported_prefix_token_count: int = 0
```

Import `DiffusionPageRange` from `page.py`. In `DiffusionKVCacheManager.reserve_request`, derive ranges from the first native cache group's block IDs:

```python
group_block_ids = tuple(blocks.get_block_ids()[0])
stable_block_count = (request.prefix_len + self.scheduler_block_size - 1) // self.scheduler_block_size
dynamic_end = request.prefix_len + request.target_len
dynamic_block_count = (dynamic_end + self.scheduler_block_size - 1) // self.scheduler_block_size
stable_ids = group_block_ids[:stable_block_count]
dynamic_ids = group_block_ids[request.prefix_len // self.scheduler_block_size : dynamic_block_count]
page_ranges = (
    DiffusionPageRange(
        cache_role="primary",
        token_start=0,
        token_count=request.prefix_len,
        block_ids=stable_ids,
        mutable=False,
    ),
    DiffusionPageRange(
        cache_role="primary",
        token_start=request.prefix_len,
        token_count=request.target_len,
        block_ids=dynamic_ids,
        mutable=True,
    ),
)
```

Store `self.scheduler_block_size` in the manager constructor. Preserve the full `block_ids` tuple for native compatibility.

Extend `DiffusionKVRequest.__init__` with:

```python
imported_prefix_token_count: int = 0,
```

Validate in `DiffusionKVRequest` that the value is non-negative and no larger
than `prefix_len`. In `DiffusionKVCacheManager.reserve_request`, require it to
be zero or aligned to `self.scheduler_block_size`, then copy it into
`DiffusionKVSequenceMetadata`. W2a imports complete stable pages only; any
remaining partial stable page is produced locally during the first forward.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/diffusion_kv/test_manager.py \
  tests/diffusion/diffusion_kv/test_metadata.py \
  tests/diffusion/diffusion_kv/test_request.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_omni/diffusion/diffusion_kv/metadata.py \
  vllm_omni/diffusion/diffusion_kv/manager.py \
  vllm_omni/diffusion/diffusion_kv/request.py \
  tests/diffusion/diffusion_kv/test_manager.py \
  tests/diffusion/diffusion_kv/test_metadata.py \
  tests/diffusion/diffusion_kv/test_request.py
git commit -m "feat: publish diffusion page ranges

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 4: Allocate Physical Worker Pages and Bind Scheduler IDs

**Files:**
- Create: `vllm_omni/diffusion/diffusion_kv/worker_registry.py`
- Create: `tests/diffusion/diffusion_kv/test_worker_registry.py`

**Interfaces:**
- Consumes: rank-local `KVCacheConfig`, discovered `dict[str, KVCacheSpec]`, `DiffusionKVMetadata`.
- Produces:
  - `WorkerPageRegistry`
  - `LayerPageStorage`
  - native `BlockTable` per cache group/request
  - `DiffusionPageBinding`
  - `get_layer_cache(layer_name) -> torch.Tensor`
  - `bind_request(metadata) -> DiffusionPageBinding`
  - `release_request(request_id, allocation_generation) -> None`

- [ ] **Step 1: Write failing CPU allocation and binding tests**

Use a fake `FullAttentionSpec` and monkeypatch the native BlockTable constructor behind a single helper. Required tests:

```python
def test_registry_allocates_native_shape_for_each_group() -> None:
    registry = _registry(num_blocks=8, block_size=4, num_heads=2, head_size=8)
    cache = registry.get_layer_cache("layer0")
    assert cache.shape == (2, 8, 4, 2, 8)
    assert cache.dtype is torch.bfloat16


def test_registry_binds_scheduler_ids_without_allocating_new_ids() -> None:
    registry = _registry(num_blocks=8)
    metadata = _metadata(block_ids=[3, 5])
    binding = registry.bind_request(metadata)
    assert binding.sequences[0].stable.block_ids == (3,)
    assert binding.sequences[0].dynamic.block_ids == (5,)
    assert binding.sequences[0].slot_mapping.tolist() == list(range(12, 16)) + list(range(20, 24))


def test_registry_rejects_generation_reuse_until_release() -> None:
    registry = _registry(num_blocks=8)
    registry.bind_request(_metadata(request_id="req", generation=1))
    with pytest.raises(ValueError, match="already has an active binding"):
        registry.bind_request(_metadata(request_id="req", generation=2))
```

Also cover:

- configured layer/spec mismatch;
- out-of-range block ID;
- duplicate block ID within one rank;
- overlapping active ownership across public requests;
- metadata sequence geometry mismatch;
- stale release;
- release transitions all owned pages through `RELEASING` to `FREE`.

- [ ] **Step 2: Verify failure**

Run:

```bash
pytest -q tests/diffusion/diffusion_kv/test_worker_registry.py
```

Expected: collection FAIL because `worker_registry.py` is absent.

- [ ] **Step 3: Implement physical allocation**

Use each native group's `kv_cache_spec.get_kv_cache_shape(kv_cache_config.num_blocks)` and dtype. Keep one physical tensor per native cache tensor/group and map all `group.layer_names` to the same storage when `shared_by` says they share:

```python
shape = group.kv_cache_spec.get_kv_cache_shape(kv_cache_config.num_blocks)
tensor = torch.empty(shape, dtype=group.kv_cache_spec.dtype, device=device)
```

Do not zero the entire pool. Zero or overwrite only pages when they enter `RESERVED`, before any read.

Construct the native BlockTable behind:

```python
def _make_native_block_table(
    *,
    block_size: int,
    max_num_reqs: int,
    max_num_blocks_per_req: int,
    device: torch.device,
) -> BlockTable:
    return BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
        pin_memory=False,
        device=device,
    )
```

Immediately adapt this helper if the refreshed vLLM constructor differs; no other W2a file may call the constructor directly.

For each sequence, append Scheduler-provided block IDs to the table and build the slot mapping from `page_ranges`. Page ownership is keyed by `(request_id, allocation_generation, sequence_id)`.

- [ ] **Step 4: Run registry tests**

Run:

```bash
pytest -q tests/diffusion/diffusion_kv/test_worker_registry.py
```

Expected: PASS on CPU.

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_omni/diffusion/diffusion_kv/worker_registry.py \
  tests/diffusion/diffusion_kv/test_worker_registry.py
git commit -m "feat: allocate and bind diffusion KV pages

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 5: Implement Transfer Descriptors and Direct-Copy Session Semantics

**Files:**
- Create: `vllm_omni/diffusion/diffusion_kv/transfer.py`
- Create: `tests/diffusion/diffusion_kv/test_transfer.py`

**Interfaces:**
- Consumes: registered source/destination page spans and `DiffusionPageBinding`.
- Produces:
  - `PageEndpoint`
  - `PageCopy`
  - `PageTransferPlan`
  - `PageTransferResult`
  - `TransferState`
  - `PageTransferSessionManager`
  - `prepare_receive`, `send_pages`, `poll_completion`, `cancel`, `close_session`

- [ ] **Step 1: Write failing state-machine tests**

Required tests:

```python
def test_direct_copy_commits_only_after_event_completion() -> None:
    event = FakeEvent(complete=False)
    manager, binding, plan = _direct_manager(event=event)
    handle = manager.send_pages(plan, manager.prepare_receive(plan))
    assert manager.poll_completion(handle) is None
    assert not binding.is_compute_ready
    event.complete = True
    result = manager.poll_completion(handle)
    assert result is not None
    assert result.status == "completed"
    assert binding.is_compute_ready


def test_stale_generation_never_commits_new_binding() -> None:
    manager, old_binding, plan = _direct_manager(generation=1)
    manager.close_session(plan.session_id)
    new_binding = _binding(generation=2)
    manager.register_binding(new_binding)
    result = manager.complete_for_test(plan.identity)
    assert result.status == "stale"
    assert not new_binding.is_compute_ready


def test_duplicate_terminal_completion_is_idempotent() -> None:
    manager, binding, plan = _direct_manager()
    handle = manager.send_pages(plan, manager.prepare_receive(plan))
    first = manager.poll_completion(handle)
    second = manager.poll_completion(handle)
    assert first == second
```

Also cover cancellation in `ALLOCATED`, `TARGET_READY`, and `TRANSFERRING`; monotonic timeout; dtype/shape/stride/byte-count mismatch; source and destination overlap; missing destination registration; and cleanup after copy failure.

- [ ] **Step 2: Verify failure**

Run:

```bash
pytest -q tests/diffusion/diffusion_kv/test_transfer.py
```

Expected: collection FAIL because `transfer.py` is absent.

- [ ] **Step 3: Implement immutable descriptors and identity**

Use exactly:

```python
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

    @property
    def identity(self) -> TransferIdentity:
        return (
            self.session_id,
            self.allocation_generation,
            self.route_epoch,
            self.op_id,
        )
```

The manager resolves endpoints through active registrations; it never accepts a raw device address from request metadata.

- [ ] **Step 4: Implement direct copy**

For each copy:

```python
with torch.cuda.stream(self._copy_stream):
    destination.copy_(source, non_blocking=True)
self._copy_stream.record_event(event)
```

`poll_completion` checks `event.query()`. Only after all events query true may it call `binding.transition_page(block_id, PageState.COMMITTED)`. Cancellation marks the session terminal before synchronizing memory-safety-critical operations.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q tests/diffusion/diffusion_kv/test_transfer.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  vllm_omni/diffusion/diffusion_kv/transfer.py \
  tests/diffusion/diffusion_kv/test_transfer.py
git commit -m "feat: add local page transfer sessions

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 6: Add a Page-Aware SharedMemory Adapter

**Files:**
- Create: `vllm_omni/distributed/omni_connectors/page_shm.py`
- Create: `tests/distributed/omni_connectors/test_page_shm.py`
- Modify: `vllm_omni/distributed/omni_connectors/connectors/shm_connector.py`

**Interfaces:**
- Consumes: `PageTransferPlan`, packed CPU payload, active destination registrations.
- Produces:
  - `PageShmRegistration`
  - `PageShmPayload`
  - `SharedMemoryPageAdapter.put_pages(plan, tensors)`
  - `SharedMemoryPageAdapter.copy_into_pages(plan, registration)`
  - deterministic cleanup of SHM object and lock file.

- [ ] **Step 1: Write failing SHM tests**

Required tests:

- pack two K/V page spans and copy them into preallocated destination tensors;
- verify no deserialized `KVCacheTransferData` or model-owned cache object is created;
- reject payload size larger than reserved `sum(endpoint.num_bytes)`;
- reject dtype/shape/stride mismatch;
- duplicate `copy_into_pages` returns the same terminal result;
- cancellation unlinks the segment;
- sender failure leaves destination pages uncommitted;
- receiver close cleans an unconsumed segment.

The success assertion must compare the destination page tensor directly with the source page tensor.

- [ ] **Step 2: Verify failure**

Run:

```bash
pytest -q tests/distributed/omni_connectors/test_page_shm.py
```

Expected: collection FAIL because `page_shm.py` is absent.

- [ ] **Step 3: Implement the packed payload**

Use a fixed JSON header followed by raw contiguous spans:

```python
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
```

The header includes only transfer identity and per-span byte offsets. Endpoint geometry remains authoritative from the receiver's active registration.

- [ ] **Step 4: Reuse SharedMemoryConnector cleanup primitives**

Add package-private helpers to `SharedMemoryConnector`:

```python
def _track_pending_key(self, key: str) -> None:
    self._pending_keys.add(key)


def _discard_pending_key(self, key: str) -> None:
    self._pending_keys.discard(key)
```

The page adapter calls these helpers but does not route page payloads through `serialize_obj` or `deserialize_obj`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q \
  tests/distributed/omni_connectors/test_page_shm.py \
  tests/distributed/omni_connectors/test_shm_connector.py \
  tests/distributed/omni_connectors/test_basic_connectors.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  vllm_omni/distributed/omni_connectors/page_shm.py \
  vllm_omni/distributed/omni_connectors/connectors/shm_connector.py \
  tests/distributed/omni_connectors/test_page_shm.py
git commit -m "feat: copy SHM payloads into diffusion pages

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 7: Install Registry and Transfer Lifecycle in the Diffusion Runner

**Files:**
- Modify: `vllm_omni/diffusion/worker/diffusion_model_runner.py`
- Modify: `vllm_omni/diffusion/worker/diffusion_worker.py`
- Modify: `vllm_omni/diffusion/forward_context.py`
- Modify: `vllm_omni/diffusion/sched/interface.py`
- Modify: `tests/diffusion/test_diffusion_model_runner.py`
- Modify: `tests/diffusion/diffusion_kv/test_worker_contract.py`

**Interfaces:**
- Consumes: `KVCacheConfig`, `DiffusionKVMetadata`, finished request IDs.
- Produces:
  - `DiffusionModelRunner.page_registry`
  - `DiffusionModelRunner.page_transfer_manager`
  - `DiffusionModelRunner.initialize_kv_cache_data_plane()`
  - `DiffusionModelRunner.install_diffusion_kv_metadata()`
  - `DiffusionModelRunner.release_diffusion_kv_requests()`
  - forward-context `diffusion_page_bindings`.

- [ ] **Step 1: Write failing runner lifecycle tests**

Required tests:

```python
def test_set_kv_cache_config_initializes_registry_only_in_paged_mode(mocker) -> None:
    runner = _runner(mode=DiffusionKVCacheMode.PAGED_SCHEDULER)
    registry_cls = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_model_runner.WorkerPageRegistry"
    )
    runner.set_kv_cache_config(_config())
    registry_cls.assert_called_once()


def test_dense_mode_never_initializes_page_registry(mocker) -> None:
    runner = _runner(mode=DiffusionKVCacheMode.DENSE_LEGACY)
    registry_cls = mocker.patch(
        "vllm_omni.diffusion.worker.diffusion_model_runner.WorkerPageRegistry"
    )
    assert runner.page_registry is None
    registry_cls.assert_not_called()


def test_finished_ids_release_before_rebinding_reused_block_ids() -> None:
    runner = _paged_runner()
    runner.execute_stepwise(_output(finished={"old"}, new=[_new("new")]))
    assert runner.page_registry.calls[:2] == [
        ("release", "old"),
        ("bind", "new"),
    ]
```

Also verify:

- metadata request/generation mismatch fails;
- the same request on cached steps reuses the same binding;
- shutdown cancels sessions before freeing physical tensors;
- all active TP ranks receive the same binding identity;
- no registry is initialized during memory profiling before `set_kv_cache_config`.

- [ ] **Step 2: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/diffusion/diffusion_kv/test_worker_contract.py
```

Expected: FAIL on missing lifecycle behavior.

- [ ] **Step 3: Initialize the data plane**

In `DiffusionModelRunner.__init__`:

```python
self.page_registry: WorkerPageRegistry | None = None
self.page_transfer_manager: PageTransferSessionManager | None = None
```

In `set_kv_cache_config`, after the existing layer-set validation:

```python
self.kv_cache_config = kv_cache_config
self.page_registry = WorkerPageRegistry(
    kv_cache_config=kv_cache_config,
    layer_specs=self.get_kv_cache_spec(),
    device=self.device,
    max_num_reqs=self.od_config.max_num_seqs,
)
self.page_transfer_manager = PageTransferSessionManager(
    registry=self.page_registry,
    timeout_s=self.od_config.diffusion_page_transfer_timeout_s,
)
```

Dense mode must reject an unexpected `set_kv_cache_config` call.

- [ ] **Step 4: Bind and release in scheduler order**

At the start of each execution cycle:

1. release `scheduler_output.finished_req_ids`;
2. validate every `scheduled_new_reqs` envelope;
3. bind its metadata;
4. install local imported pages;
5. expose bindings in `set_forward_context`;
6. after first-step forward, record and poll stable-page write completion;
7. on per-request exception, cancel the session and release the binding.

Extend `set_forward_context` with:

```python
diffusion_page_bindings: dict[str, DiffusionPageBinding] | None = None
```

Never place tensors or device pointers in `DiffusionSchedulerOutput`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/diffusion/diffusion_kv/test_worker_contract.py \
  tests/diffusion/test_diffusion_worker.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  vllm_omni/diffusion/worker/diffusion_model_runner.py \
  vllm_omni/diffusion/worker/diffusion_worker.py \
  vllm_omni/diffusion/forward_context.py \
  vllm_omni/diffusion/sched/interface.py \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/diffusion/diffusion_kv/test_worker_contract.py
git commit -m "feat: install diffusion pages in the worker

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 8: Implement the Hunyuan Stable/Dynamic Page Adapter

**Files:**
- Create: `vllm_omni/diffusion/models/hunyuan_image3/paged_kv.py`
- Create: `tests/diffusion/models/hunyuan_image3/test_paged_kv.py`

**Interfaces:**
- Consumes: Hunyuan `query_lens`, `seq_lens`, CFG row order, `DiffusionSequenceBinding`, layer page tensor.
- Produces:
  - `HunyuanPagedKVBatch`
  - `build_hunyuan_paged_kv_batch`
  - `write_hunyuan_kv`
  - `gather_hunyuan_kv_reference`
  - stable/dynamic publication masks.

- [ ] **Step 1: Write failing adapter tests**

Cover:

- one no-CFG row;
- two CFG rows with distinct block IDs;
- stable prefix and dynamic target spanning partial blocks;
- partial imported stable pages with local production of the remaining prefix;
- later step reuses stable slots and overwrites dynamic slots;
- dynamic page contents change after the second write;
- stable page contents do not change;
- target blocks are never returned by `cacheable_block_ids`;
- row geometry mismatch fails;
- position IDs and RoPE token order match dense logical order.

The central test:

```python
def test_second_step_reuses_stable_slots_and_overwrites_dynamic_slots() -> None:
    pages, binding = _pages_and_binding()
    first_key, first_value = _kv(seed=1)
    write_hunyuan_kv(pages, binding, first_key, first_value)
    stable_before = gather_stable(pages, binding).clone()
    dynamic_before = gather_dynamic(pages, binding).clone()

    second_key, second_value = _kv(seed=2)
    write_hunyuan_kv(
        pages,
        binding,
        second_key,
        second_value,
        stable_already_committed=True,
    )

    torch.testing.assert_close(gather_stable(pages, binding), stable_before)
    assert not torch.equal(gather_dynamic(pages, binding), dynamic_before)
```

- [ ] **Step 2: Verify failure**

Run:

```bash
pytest -q tests/diffusion/models/hunyuan_image3/test_paged_kv.py
```

Expected: collection FAIL because `paged_kv.py` is absent.

- [ ] **Step 3: Implement page writes**

`write_hunyuan_kv` flattens `[batch, tokens, kv_heads, head_dim]`, selects stable rows only on first local production, always selects dynamic rows, and uses `index_copy_` into a flattened `[num_blocks * block_size, kv_heads, head_dim]` view:

```python
flat_key_pages = layer_cache[0].flatten(0, 1)
flat_value_pages = layer_cache[1].flatten(0, 1)
flat_key_pages.index_copy_(0, slot_mapping, key_rows)
flat_value_pages.index_copy_(0, slot_mapping, value_rows)
```

For imported stable pages, skip writes for
`[0, imported_prefix_token_count)`, require those complete pages to be
committed, and locally write
`[imported_prefix_token_count, stable.token_count)` during the first forward.
Later steps reuse the union of imported and locally produced stable pages.

- [ ] **Step 4: Implement bounded reference gather**

`gather_hunyuan_kv_reference` uses `index_select` from page storage into per-forward scratch:

```python
key = flat_key_pages.index_select(0, binding.slot_mapping)
value = flat_value_pages.index_select(0, binding.slot_mapping)
return key.reshape(batch, seq_len, kv_heads, head_dim), value.reshape(
    batch,
    seq_len,
    kv_heads,
    head_dim,
)
```

This scratch is not stored in request state, `image_kv_cache_map`, or a connector payload. Add counters for gathered bytes and gather latency so the E2E report exposes the reference backend cost.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q tests/diffusion/models/hunyuan_image3/test_paged_kv.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  vllm_omni/diffusion/models/hunyuan_image3/paged_kv.py \
  tests/diffusion/models/hunyuan_image3/test_paged_kv.py
git commit -m "feat: map Hunyuan KV onto worker pages

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 9: Switch Hunyuan Attention Between Dense and Paged Modes

**Files:**
- Modify: `vllm_omni/diffusion/attention/backends/abstract.py`
- Modify: `vllm_omni/diffusion/attention/layer.py`
- Modify: `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py`
- Modify: `tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py`
- Create: `tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_paged_step_execution.py`

**Interfaces:**
- Consumes: current forward-context page bindings and layer cache.
- Produces:
  - `AttentionBackend.supports_paged_kv`
  - `AttentionBackend.supports_non_causal_paged_kv`
  - paged `ImageKVCacheManager.forward`
  - dense path byte-for-byte behavior preservation.

- [ ] **Step 1: Write failing mode-separation tests**

Required assertions:

```python
def test_paged_first_step_does_not_allocate_image_kv_cache_map() -> None:
    manager = _manager(mode=DiffusionKVCacheMode.PAGED_SCHEDULER)
    manager.forward(*_inputs(first_step=True), request_ids=["req"], row_branches=[0])
    assert manager.image_kv_cache_map is None
    assert manager.image_kv_cache_lens is None


def test_dense_first_step_still_populates_image_kv_cache_map() -> None:
    manager = _manager(mode=DiffusionKVCacheMode.DENSE_LEGACY)
    manager.forward(*_inputs(first_step=True))
    assert manager.image_kv_cache_map is not None
    assert manager.image_kv_cache_lens is not None
```

Add output parity for first step plus two subsequent steps using deterministic random tensors.

- [ ] **Step 2: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_paged_step_execution.py
```

Expected: FAIL because paged execution still calls `_cache_prompt_kv`.

- [ ] **Step 3: Add backend capability validation**

In `AttentionBackend`:

```python
supports_paged_kv: bool = False
supports_non_causal_paged_kv: bool = False
```

The W2a Hunyuan reference path is implemented by the model adapter and Torch SDPA, so the selected backend must be `TORCH_SDPA` for `paged_scheduler` until another backend explicitly implements both flags. Startup validation must include the model name, selected backend, and required capability in the error.

- [ ] **Step 4: Add paged branch without changing dense helpers**

At the top of `ImageKVCacheManager.forward`, resolve the cache mode. Keep `_cache_prompt_kv`, `_reuse_prompt_kv`, `_build_neg_ar_kv`, and dense call sites unchanged under `dense_legacy`.

In paged mode:

1. fetch request/branch bindings from forward context;
2. validate CFG row count and generation;
3. write local stable and dynamic K/V through `write_hunyuan_kv`;
4. gather committed stable plus current dynamic K/V through the reference adapter;
5. build the existing `AttentionMetadata`;
6. call `self.attn`;
7. record first-step stable-write completion for later commit.

Never assign `image_kv_cache_map`, `image_kv_cache_lens`, or `_injected_ar_kv` in paged mode.

- [ ] **Step 5: Run mode and parity tests**

Run:

```bash
pytest -q \
  tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_paged_step_execution.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  vllm_omni/diffusion/attention/backends/abstract.py \
  vllm_omni/diffusion/attention/layer.py \
  vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py \
  tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_paged_step_execution.py
git commit -m "feat: execute Hunyuan attention from KV pages

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 10: Remove Dense Prompt-KV Snapshots from Paged Step State

**Files:**
- Modify: `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`
- Modify: `vllm_omni/diffusion/models/hunyuan_image3/request_layout.py`
- Modify: `tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py`
- Modify: `tests/diffusion/models/hunyuan_image3/test_diffusion_kv_request.py`

**Interfaces:**
- Consumes: Runner-owned page binding identity.
- Produces: request/branch row mapping for paged attention without `_STEP_PROMPT_KV` or `_STEP_AR_KV` tensor snapshots.

- [ ] **Step 1: Write failing state tests**

Add tests proving:

- paged `prepare_encode` stores no `_STEP_AR_KV` tensor list;
- paged first step stores no `_STEP_PROMPT_KV`;
- later paged steps require a live Worker page binding instead;
- dense mode still snapshots/restores the existing fields;
- grouped request batching preserves `request_ids` and `row_branches` order;
- CFG rows map to the corresponding `DiffusionKVSequenceMetadata.sequence_id`.

- [ ] **Step 2: Verify failure**

Run:

```bash
pytest -q \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py \
  tests/diffusion/models/hunyuan_image3/test_diffusion_kv_request.py
```

Expected: FAIL because paged mode still snapshots dense cache tensors.

- [ ] **Step 3: Gate dense snapshot methods**

Call `_snapshot_injected_ar_kv`, `_restore_injected_ar_kv`, `_restore_prompt_kv_cache`, and `_capture_prompt_kv_cache` only when `diffusion_kv_mode is DENSE_LEGACY`.

For paged mode, add to model kwargs:

```python
model_kwargs["request_ids"] = [state.request_id for state in states for _ in range(cfg_factor)]
model_kwargs["row_branches"] = [
    branch
    for branch in range(cfg_factor)
    for _ in states
]
```

Use the existing `_step_row_order` output rather than rebuilding a different order.

- [ ] **Step 4: Validate prepared geometry**

In `build_hunyuan_diffusion_kv_requests`, enforce:

```python
if prefix_len + target_len > seq_len:
    raise ValueError(
        "Hunyuan stable prefix and dynamic target exceed valid sequence length: "
        f"prefix_len={prefix_len}, target_len={target_len}, seq_len={seq_len}"
    )
```

Read local page-import intent from:

```python
page_native_info = (request.kv_sender_info or {}).get("page_native") or {}
imported_prefix_token_count = int(
    page_native_info.get("available_prefix_token_count", 0)
)
```

Pass `imported_prefix_token_count` to `DiffusionKVRequest`. Do not attach dense
AR K/V tensors to `DiffusionKVRequest`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py \
  tests/diffusion/models/hunyuan_image3/test_diffusion_kv_request.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_paged_step_execution.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py \
  vllm_omni/diffusion/models/hunyuan_image3/request_layout.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py \
  tests/diffusion/models/hunyuan_image3/test_diffusion_kv_request.py
git commit -m "refactor: keep paged Hunyuan KV out of step state

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 11: Add Scheduler Readiness Without Head-of-Line Blocking

**Files:**
- Modify: `vllm_omni/diffusion/sched/interface.py`
- Modify: `vllm_omni/diffusion/sched/base_scheduler.py`
- Modify: `vllm_omni/diffusion/diffusion_engine.py`
- Modify: `tests/diffusion/test_diffusion_scheduler.py`
- Modify: `tests/diffusion/test_diffusion_engine_rpc_routing.py`

**Interfaces:**
- Consumes: Scheduler allocation and Worker readiness updates.
- Produces:
  - `DiffusionKVReadiness`
  - `WorkerKVUpdate`
  - allocated/waiting/ready request substates
  - queue scan that skips a transfer-waiting request and schedules a later ready request.

- [ ] **Step 1: Write failing scheduler tests**

Required cases:

```python
def test_waiting_for_local_kv_does_not_block_ready_request() -> None:
    scheduler = _paged_scheduler(max_num_seqs=1)
    scheduler.add_request(_request("waiting", imported_prefix_token_count=8))
    scheduler.add_request(_request("ready", imported_prefix_token_count=0))

    allocation = scheduler.schedule()
    assert [item.request_id for item in allocation.page_install_reqs] == [
        "waiting",
        "ready",
    ]
    ready_install = next(
        item for item in allocation.page_install_reqs if item.request_id == "ready"
    )
    scheduler.update_worker_kv(
        WorkerKVUpdate(
            "ready",
            ready_install.allocation_generation,
            tp_rank=0,
            status="ready",
        )
    )

    compute = scheduler.schedule()
    assert compute.scheduled_request_ids == ["ready"]


def test_request_runs_only_after_all_tp_ranks_report_same_generation() -> None:
    scheduler = _paged_scheduler(tp_size=2)
    scheduler.add_request(_request("req", imported_prefix_token_count=8))
    allocation = scheduler.schedule()
    generation = allocation.page_install_reqs[0].allocation_generation
    scheduler.update_worker_kv(
        WorkerKVUpdate("req", generation, tp_rank=0, status="ready")
    )
    assert scheduler.schedule().scheduled_request_ids == []
    scheduler.update_worker_kv(
        WorkerKVUpdate("req", generation, tp_rank=1, status="ready")
    )
    assert scheduler.schedule().scheduled_request_ids == ["req"]
```

Also cover:

- stale generation update ignored;
- one-rank failure terminally fails the public request;
- cancellation while installing releases Worker state before Scheduler blocks;
- impossible-fit versus temporary capacity remains correct;
- active and idle DP ranks agree on readiness;
- strict sampling-parameter compatibility remains enforced for ready requests.

- [ ] **Step 2: Verify failure**

Run:

```bash
pytest -q \
  tests/diffusion/test_diffusion_scheduler.py \
  tests/diffusion/test_diffusion_engine_rpc_routing.py
```

Expected: FAIL because allocation immediately transitions to `RUNNING` and the queue breaks on the first blocked request.

- [ ] **Step 3: Add readiness DTOs**

Add:

```python
class DiffusionKVReadiness(enum.Enum):
    UNALLOCATED = "unallocated"
    WAITING_FOR_LOCAL_KVS = "waiting_for_local_kvs"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class WorkerKVUpdate:
    request_id: str
    allocation_generation: int
    tp_rank: int
    status: Literal["ready", "failed", "released"]
    error: str | None = None


@dataclass(frozen=True)
class PageInstallRequest:
    request_id: str
    allocation_generation: int
    metadata: DiffusionKVMetadata
```

Add Scheduler-owned generation and rank-status fields to `SchedulerRequestState`; do not add tensor fields.
Add `page_install_reqs: list[PageInstallRequest]` to
`DiffusionSchedulerOutput`, and include it in `is_empty` so allocation/install
work is not dropped merely because no model request runs in the same cycle.

- [ ] **Step 4: Split allocation from compute admission**

Refactor `BaseScheduler.schedule` into these passes:

1. include already-running requests;
2. allocate unallocated waiting requests while capacity permits;
3. emit page-install work for newly allocated requests;
4. scan the complete waiting deque for `READY` requests compatible with the active batch;
5. leave transfer-waiting requests in place without breaking the scan;
6. move only ready requests to `_running`.

Keep stable FIFO order among requests with equal readiness and compatibility.

- [ ] **Step 5: Route Worker updates**

`DiffusionEngine` must apply Worker updates under the existing scheduler/RPC lock before the next `schedule()` call. A release update is required before the Scheduler returns blocks to the native pool after cancellation or failure.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/test_diffusion_scheduler.py \
  tests/diffusion/test_diffusion_engine_rpc_routing.py \
  tests/diffusion/test_diffusion_engine_cleanup.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add \
  vllm_omni/diffusion/sched/interface.py \
  vllm_omni/diffusion/sched/base_scheduler.py \
  vllm_omni/diffusion/diffusion_engine.py \
  tests/diffusion/test_diffusion_scheduler.py \
  tests/diffusion/test_diffusion_engine_rpc_routing.py
git commit -m "feat: schedule diffusion requests after page commit

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 12: Add Startup Capability Checks and Bounded Configuration

**Files:**
- Modify: `vllm_omni/diffusion/data.py`
- Modify: `vllm_omni/diffusion/worker/diffusion_model_runner.py`
- Modify: `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`
- Modify: `tests/diffusion/test_diffusion_config_propagation.py`
- Modify: `tests/diffusion/test_diffusion_model_runner.py`

**Interfaces:**
- Consumes: selected model/backend/platform/connector/topology.
- Produces:
  - `diffusion_page_transfer_timeout_s`
  - `diffusion_page_max_in_flight_bytes`
  - `diffusion_page_max_sessions`
  - actionable startup validation.

- [ ] **Step 1: Write failing configuration tests**

Verify:

- timeout is finite and positive;
- max bytes and sessions are positive;
- dense mode ignores W2a values;
- paged mode rejects non-Hunyuan model;
- paged mode rejects non-CUDA in W2a;
- paged mode rejects backend other than Torch SDPA reference path;
- paged local import rejects Mooncake Store and accepts direct/SharedMemory page adapters;
- unsupported SP/AG layout fails before the first request.

- [ ] **Step 2: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/test_diffusion_config_propagation.py \
  tests/diffusion/test_diffusion_model_runner.py
```

Expected: FAIL on missing fields and checks.

- [ ] **Step 3: Add bounded defaults**

Add config fields:

```python
diffusion_page_transfer_timeout_s: float = 30.0
diffusion_page_max_in_flight_bytes: int = 8 * 1024**3
diffusion_page_max_sessions: int = 8
```

Validate them in `OmniDiffusionConfig.__post_init__`. Error text must identify the invalid field and value.

- [ ] **Step 4: Implement one startup validator**

Add `DiffusionModelRunner._validate_page_native_capability()` and call it after model load and before cache spec discovery. It must report:

```python
raise ValueError(
    "paged_scheduler is unsupported for "
    f"model={model_name}, backend={backend_name}, platform={platform_name}, "
    f"connector={connector_name}, topology={topology_name}: {reason}"
)
```

No caller may catch this error and switch to dense mode.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/test_diffusion_config_propagation.py \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/diffusion/diffusion_kv/test_initialization.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add \
  vllm_omni/diffusion/data.py \
  vllm_omni/diffusion/worker/diffusion_model_runner.py \
  vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py \
  tests/diffusion/test_diffusion_config_propagation.py \
  tests/diffusion/test_diffusion_model_runner.py
git commit -m "feat: validate Hunyuan page-native capability

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 13: Add Observability and Cleanup Accounting

**Files:**
- Modify: `vllm_omni/diffusion/diffusion_kv/worker_registry.py`
- Modify: `vllm_omni/diffusion/diffusion_kv/transfer.py`
- Modify: `vllm_omni/diffusion/data.py`
- Modify: `tests/diffusion/diffusion_kv/test_worker_registry.py`
- Modify: `tests/diffusion/diffusion_kv/test_transfer.py`

**Interfaces:**
- Consumes: page/session state transitions.
- Produces:
  - `DiffusionPageMetrics`
  - per-request terminal snapshot
  - aggregate counters and high-water marks.

- [ ] **Step 1: Write failing metric tests**

Assert counters for:

- stable pages requested, imported, committed;
- transferred bytes;
- local install latency;
- time waiting for local K/V;
- stale completion;
- duplicate completion;
- cancellation;
- timeout;
- page-pool utilization;
- staging and in-flight byte high-water marks;
- reference-gather bytes and latency.

The terminal snapshot must include:

```python
{
    "request_id": "req",
    "allocation_generation": 3,
    "session_id": "session-3",
    "tp_rank": 0,
    "page_count": 4,
    "transferred_bytes": 8192,
    "terminal_status": "completed",
}
```

- [ ] **Step 2: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/diffusion_kv/test_worker_registry.py \
  tests/diffusion/diffusion_kv/test_transfer.py
```

Expected: FAIL on missing metrics.

- [ ] **Step 3: Implement metrics without device synchronization**

Use Python integers and `time.monotonic()` around host-side state changes. Do not call `.item()` on CUDA tensors in the hot path. Page counts and byte sizes come from metadata and registered geometry.

- [ ] **Step 4: Run tests**

Run:

```bash
pytest -q \
  tests/diffusion/diffusion_kv/test_worker_registry.py \
  tests/diffusion/diffusion_kv/test_transfer.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  vllm_omni/diffusion/diffusion_kv/worker_registry.py \
  vllm_omni/diffusion/diffusion_kv/transfer.py \
  vllm_omni/diffusion/data.py \
  tests/diffusion/diffusion_kv/test_worker_registry.py \
  tests/diffusion/diffusion_kv/test_transfer.py
git commit -m "feat: report diffusion page lifecycle metrics

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 14: Run the CPU and Static Regression Gate

**Files:**
- Modify only files required to fix failures introduced by Tasks 1–13.

**Interfaces:**
- Consumes: all W2a unit and existing dense regression tests.
- Produces: a clean static/unit gate before GPU work.

- [ ] **Step 1: Run the focused W2a suite**

```bash
pytest -q \
  tests/diffusion/diffusion_kv \
  tests/diffusion/models/hunyuan_image3/test_paged_kv.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_paged_step_execution.py \
  tests/distributed/omni_connectors/test_page_shm.py
```

Expected: PASS.

- [ ] **Step 2: Run dense Hunyuan regression tests**

```bash
pytest -q \
  tests/diffusion/models/hunyuan_image3/test_image_kv_cache_manager.py \
  tests/diffusion/models/hunyuan_image3/test_hunyuan_image3_step_execution.py \
  tests/diffusion/models/hunyuan_image3/test_diffusion_kv_request.py
```

Expected: PASS with both explicit dense and paged assertions.

- [ ] **Step 3: Run scheduler/runner/connector regressions**

```bash
pytest -q \
  tests/diffusion/test_diffusion_scheduler.py \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/diffusion/test_diffusion_engine_cleanup.py \
  tests/distributed/omni_connectors/test_shm_connector.py \
  tests/distributed/omni_connectors/test_basic_connectors.py
```

Expected: PASS.

- [ ] **Step 4: Run formatting and repository hooks**

```bash
git diff --check
uvx --from pre-commit==4.0.1 pre-commit run --all-files
```

Expected: all applicable hooks PASS.

- [ ] **Step 5: Commit any gate-only corrections**

```bash
git add -u
git commit -m "test: close W2a CPU regression gate

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Skip this commit only when `git status --short` is empty.

### Task 15: Add Deterministic GPU Functional Coverage

**Files:**
- Create: `tests/e2e/accuracy/test_hunyuan_image3_paged_kv.py`
- Modify: `.buildkite/test_areas/diffusion.yaml`
- Modify: `tests/e2e/accuracy/test_hunyuan_image3.py`

**Interfaces:**
- Consumes: real HunyuanImage3 checkpoint and CUDA page path.
- Produces: dense-versus-paged parity evidence across first and later steps.

- [ ] **Step 1: Add the H100 test matrix**

Parameterize:

```python
@pytest.mark.parametrize(
    ("guidance_scale", "tensor_parallel_size"),
    [
        (1.0, 1),
        (5.0, 1),
        (1.0, 2),
        (5.0, 2),
    ],
)
```

Each case uses:

- exact model revision;
- fixed prompt and seed;
- first step plus at least two subsequent denoising steps;
- identical dense and paged inputs;
- local imported-page case;
- partial imported-prefix case where only complete leading pages arrive and
  the remaining stable prefix is projected locally;
- cancellation-during-copy case;
- explicit non-paged model startup regression.

- [ ] **Step 2: Report correctness metrics**

Compute and print:

```text
mean_abs_diff
p99_abs_diff
ssim
psnr
dense_peak_memory_mb
paged_peak_memory_mb
reference_gather_bytes
```

Use the existing Hunyuan accuracy thresholds as the starting gate. Any threshold change requires a separate accuracy justification and may not be weakened only for W2a.

- [ ] **Step 3: Add active/idle DP and supported EP-padding cases**

Use the repository's distributed test harness so one DP rank has work while another is idle. Assert all participating TP ranks report the same generation and that idle ranks do not retain a binding after completion.

- [ ] **Step 4: Run a supplemental remote smoke**

On the approved server:

```bash
ssh sitian@10.232.195.203
```

Run only the focused test file on available GPUs. Record exact hardware with `nvidia-smi -L`. If the server does not provide H100s or the required topology, label this result supplemental and do not treat it as the formal gate.

- [ ] **Step 5: Trigger the formal Buildkite H100 gate**

Required command:

```bash
pytest -sv tests/e2e/accuracy/test_hunyuan_image3_paged_kv.py \
  -m 'full_model and cuda and H100' \
  --run-level full_model
```

Required hardware for formal distributed acceptance: four H100 GPUs when exercising active/idle DP plus TP>1 topology.

Expected: all required matrix cases PASS. Save Buildkite URL, commit SHA, worker hardware, and raw test log.

- [ ] **Step 6: Commit**

```bash
git add \
  tests/e2e/accuracy/test_hunyuan_image3_paged_kv.py \
  tests/e2e/accuracy/test_hunyuan_image3.py \
  .buildkite/test_areas/diffusion.yaml
git commit -m "test: cover Hunyuan page-native KV on GPU

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 16: Add Controlled End-to-End Performance Evidence

**Files:**
- Create: `benchmarks/diffusion/hunyuan_image3_page_native.py`
- Create: `docs/performance/hunyuan_image3_page_native_w2a.md`

**Interfaces:**
- Consumes: dense and paged feature toggles on the same implementation commit.
- Produces: reproducible cold/repeated/mixed/concurrency A/B with median and dispersion.

- [ ] **Step 1: Implement the benchmark CLI**

Required arguments:

```text
--model
--model-revision
--mode dense_legacy|paged_scheduler
--workload cold|repeated|mixed
--concurrency 1|4|8
--warmup
--repetitions
--seed
--height
--width
--steps
--output-json
```

Reject fewer than five measured repetitions.

- [ ] **Step 2: Emit one JSON record per request and one summary**

Per-request fields:

```text
commit
model_revision
mode
workload
concurrency
request_id
latency_ms
transferred_bytes
transfer_wait_ms
scheduler_ready_ms
scheduler_waiting_ms
reference_gather_ms
peak_memory_mb
```

Summary fields:

```text
throughput_qps
latency_p50_ms
latency_p95_ms
latency_p99_ms
latency_median_ms
latency_mad_ms
gpu_idle_percent
hbm_high_water_mb
```

- [ ] **Step 3: Run controlled A/B**

Use the same commit except for `diffusion_kv_mode`, same checkpoint revision, prompts, seeds, image resolution, steps, dtype, topology, warmup, and request count.

Run:

```bash
python benchmarks/diffusion/hunyuan_image3_page_native.py \
  --model tencent/HunyuanImage-3.0-Instruct \
  --model-revision 2ec2c78bee7d4b94157341fba86c4c2c7b1858b2 \
  --mode dense_legacy \
  --workload cold \
  --concurrency 1 \
  --warmup 2 \
  --repetitions 5 \
  --seed 0 \
  --height 1024 \
  --width 1024 \
  --steps 4 \
  --output-json /tmp/hunyuan-w2a-dense-cold-c1.json
```

Repeat for both modes, all three workloads, and concurrency `1`, `4`, and `8`.

- [ ] **Step 4: Write the performance report**

The report must:

- state that W2a uses a reference page gather if that remains true;
- separate local-install time from attention gather time;
- report median and MAD plus P50/P95/P99;
- report all repetitions or link the raw JSON artifact;
- identify cold-path regression, if any;
- make no performance-win claim unless repeated or mixed context improves repeatably without a statistically meaningful cold regression;
- state that helper timings are supporting evidence only.

- [ ] **Step 5: Commit**

```bash
git add \
  benchmarks/diffusion/hunyuan_image3_page_native.py \
  docs/performance/hunyuan_image3_page_native_w2a.md
git commit -m "perf: benchmark Hunyuan page-native KV end to end

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 17: Final W2a Acceptance Audit

**Files:**
- Create: `docs/performance/hunyuan_image3_page_native_w2a_acceptance.md`
- Modify only files needed to close an identified acceptance gap.

**Interfaces:**
- Consumes: code, tests, Buildkite results, raw benchmark JSON, and design acceptance criteria.
- Produces: a requirement-to-evidence matrix and an explicit `GO` or `NO_GO`.

- [ ] **Step 1: Build the requirement-to-evidence matrix**

The acceptance document must contain one row for every W2a criterion:

| Requirement | Code Evidence | Test Evidence | External Gate | Status |
|---|---|---|---|---|
| Real Worker `kv_caches` use Scheduler block IDs | `worker_registry.py` | `test_worker_registry.py` | H100 log | |
| No `image_kv_cache_map` in paged mode | transformer paged branch | mode-separation test | H100 log | |
| Stable pages persist; dynamic slots overwrite | `paged_kv.py` | multi-step tests | parity log | |
| Uncommitted external pages are invisible | transfer manager | fake-event and cancellation tests | local-import GPU case | |
| Local direct/SHM transfer uses terminal completion | transfer and page SHM adapter | transfer/SHM tests | GPU local-import case | |
| Dense mode unchanged | dense branch | dense regression suite | dense accuracy case | |
| Required topology parity passes | runner/scheduler | distributed unit tests | four-H100 Buildkite | |
| End-to-end results and limits reported | benchmark/report | JSON verifier | benchmark artifacts | |

Mark missing evidence `NO_GO`; do not infer coverage from an unrelated green suite.

- [ ] **Step 2: Verify persistent-cache ownership**

Run:

```bash
rg -n "image_kv_cache_map|_STEP_PROMPT_KV|_STEP_AR_KV" \
  vllm_omni/diffusion/models/hunyuan_image3
```

Inspect every hit. Dense-only references are allowed. Any paged assignment or paged request-state tensor snapshot is a release blocker.

- [ ] **Step 3: Verify no silent fallback**

Run:

```bash
rg -n "paged_scheduler|dense_legacy" \
  vllm_omni/diffusion \
  tests/diffusion
```

Inspect exception handling around capability validation. Any catch-and-fallback behavior is a release blocker.

- [ ] **Step 4: Verify commits and repository state**

```bash
git status --short
git log --format='%h %s%n%b' origin/main..HEAD
git diff --check origin/main...HEAD
```

Expected:

- clean worktree;
- every commit ends with exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>`;
- no whitespace errors.

- [ ] **Step 5: Re-run final gates**

```bash
pytest -q \
  tests/diffusion/diffusion_kv \
  tests/diffusion/models/hunyuan_image3 \
  tests/diffusion/test_diffusion_scheduler.py \
  tests/diffusion/test_diffusion_model_runner.py \
  tests/distributed/omni_connectors/test_page_shm.py \
  tests/distributed/omni_connectors/test_shm_connector.py
uvx --from pre-commit==4.0.1 pre-commit run --all-files
```

Then verify the exact Buildkite H100 run and benchmark artifacts listed in the acceptance matrix.

- [ ] **Step 6: Set the final status**

Set `GO` only if every matrix row is covered by direct evidence. In particular:

- CPU tests do not replace H100 topology coverage;
- a local GPU smoke does not replace the formal four-H100 gate;
- a helper microbenchmark does not replace controlled E2E A/B;
- output parity does not prove cleanup or stale-generation safety;
- a performance improvement does not excuse correctness or dense regression.

Otherwise set `NO_GO` and list each missing gate with the exact command or owner needed to close it.

- [ ] **Step 7: Commit**

```bash
git add docs/performance/hunyuan_image3_page_native_w2a_acceptance.md
git commit -m "docs: record Hunyuan W2a acceptance evidence

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

## Execution Order and Review Gates

1. Tasks 1–4 establish control-plane compatibility and physical page ownership.
2. Tasks 5–7 establish completion-aware local installation and Runner lifecycle.
3. Tasks 8–10 migrate Hunyuan execution without changing dense behavior.
4. Tasks 11–13 add Scheduler readiness, capability checks, and observability.
5. Task 14 is the CPU/static review gate.
6. Task 15 is the correctness/topology GPU gate.
7. Task 16 is the end-to-end performance evidence gate.
8. Task 17 is the explicit acceptance audit.

Do not begin W2b Mooncake sender-push work from this branch until Task 17 is `GO`. W2b must reuse `PageEndpoint`, `PageTransferPlan`, `PageTransferResult`, the session identity, Worker registrations, and commit-before-visibility semantics created here.
