# HunyuanImage3 Page-Native KV Transfer Design

**Status:** Approved design, awaiting written-spec review

**Date:** 2026-08-15

**Pilot model:** HunyuanImage3

**Primary RFC:** vllm-project/vllm-omni#5244

**Control-plane dependency:** vllm-project/vllm-omni#6094, inspected at `7e2245e22c5d5df6497a0740873f261f20a7f7ca`

**Design base:** `origin/main` at `555336e8`

## 1. Summary

This design replaces HunyuanImage3's model-owned contiguous prompt-KV tensors
with Scheduler-owned, Worker-installed pages and adds a page-native AR-to-DiT
transfer lifecycle.

The work is split into two independently reviewable phases:

- **W2a: local page-native data plane.** Install real rank-local `kv_caches`,
  bind Scheduler block IDs to Worker BlockTables and slot mappings, execute
  Hunyuan attention against those pages, and integrate a completion-aware
  SharedMemory/direct-copy path.
- **W2b: remote missing-only transfer.** Add receiver-driven, sender-push
  Mooncake transfer into preallocated destination pages, source leases,
  destination completion, and missing-only page selection.

W2a is the first implementation target. W2b must reuse W2a's page descriptors,
session state machine, and readiness contract rather than introducing a second
cache representation.

The design deliberately does not extend PR #5640's dense event-driven H2D
path. That path demonstrated a valid local-copy microbenchmark improvement but
only approximately `+0.469%` HunyuanImage3 end-to-end throughput, within run
variance. The new path changes ownership and execution granularity so that
cache-aware scheduling and page-level transfer can produce measurable
end-to-end effects.

## 2. Current State and Proven Gap

### 2.1 Current `main`

Current HunyuanImage3 execution stores stable prompt K/V in
`ImageKVCacheManager.image_kv_cache_map`:

- first step allocates contiguous tensors and copies prompt K/V into them;
- later denoising steps concatenate or otherwise combine those tensors with
  current image-token K/V;
- `_injected_ar_kv` imports AR K/V as model-owned tensors;
- Worker scheduling cannot reason about the physical residency or readiness of
  those tensors.

This representation prevents:

- destination-page preallocation before AR transfer;
- missing-only transfer based on receiver-local page hits;
- page refcounting and delayed free;
- cache-aware admission independent of FIFO request order;
- direct connector writes into the final attention cache;
- a single ownership model for local reuse and disaggregated transfer.

### 2.2 What PR #6094 provides

PR #6094 establishes the control plane:

- Workers expose native `KVCacheSpec` objects from cache-enabled attention
  modules.
- The Engine builds rank-local Worker `KVCacheConfig` objects and one Scheduler
  `KVCacheConfig`.
- `DiffusionKVCacheManager` wraps native `KVCacheManager` for atomic public
  request allocation.
- The Scheduler sends `DiffusionKVMetadata` containing sequence block IDs.
- `DiffusionModelRunner` retains its rank-local `KVCacheConfig`.

PR #6094 explicitly does **not** provide:

- physical Worker `kv_caches`;
- Worker BlockTables or slot mappings for diffusion;
- installation of Scheduler block IDs into model execution;
- a paged Hunyuan attention read/write path;
- transfer directly into destination pages;
- connector completion tied to page visibility;
- delayed source-page release after receiver acknowledgement.

W2 begins at this boundary.

### 2.3 Compatibility rule

Implementation must either:

1. stack on the latest reviewed #6094 head, or
2. start from `main` after #6094 merges.

It must not copy a stale subset of #6094 into a separate allocator. If #6094's
public types or initialization flow change before implementation, W2 must adapt
to the merged/current interfaces while preserving the invariants in this
document.

## 3. Goals

### 3.1 W2a goals

1. Allocate rank-local physical KV page tensors from the Worker
   `KVCacheConfig`.
2. Convert `DiffusionKVMetadata.block_ids` into rank-local BlockTables and slot
   mappings.
3. Execute HunyuanImage3 prompt reuse through paged K/V instead of
   `image_kv_cache_map`.
4. Keep stable prefix pages allocated across denoising steps while overwriting
   dynamic target slots every step.
5. Support local imported AR pages through an explicit transfer session and
   completion event.
6. Prevent pages from becoming visible to attention before installation or
   transfer completes.
7. Preserve `dense_legacy` behavior and fail fast when `paged_scheduler` lacks
   a required page-native capability.
8. Produce controlled end-to-end evidence, not only helper microbenchmarks.

### 3.2 W2b goals

1. Let the DiT receiver perform local lookup and destination allocation first.
2. Transfer only missing stable pages from AR.
3. Use receiver-driven rendezvous followed by sender-push RDMA writes.
4. Retain source-page leases until terminal receiver acknowledgement.
5. Admit DiT compute only after every required destination page is committed.
6. Handle cancellation, timeout, duplicate messages, late completion, and
   Worker failure without exposing stale data or leaking pages.

## 4. Non-Goals

This design does not:

- migrate every diffusion model in the first phase;
- unify AR and DiT into one model runner;
- replace native vLLM `BlockPool`, `KVCacheManager`, `KVCacheConfig`, or
  `BlockTable`;
- add a second Omni-owned physical page allocator;
- prefix-cache dynamic latent K/V across denoising steps;
- silently fall back from `paged_scheduler` to `dense_legacy`;
- claim a performance win from transfer-helper timings alone;
- require Mooncake Store support for W2b;
- redesign embedding, VAE, audio-stream, or generic payload transfer.

HunyuanImage3 is the only required model for W2a and W2b. Other models may
adopt the interfaces later after defining their own stable/dynamic cache
semantics.

## 5. Core Invariants

### INV-1: Scheduler owns allocation

The Scheduler is the sole owner of request allocation, block identity,
admission, refcounts, and terminal release. Workers may materialize and operate
on pages but may not invent request block IDs.

### INV-2: Worker owns physical tensors

Each Worker owns its rank-local physical `kv_caches`, page views, BlockTables,
slot mappings, CUDA streams, and completion events. Scheduler metadata does not
contain tensor pointers.

### INV-3: Completion precedes cross-operation visibility

An externally installed destination page may be referenced by attention only
after every required K/V copy for the current allocation generation has
completed and the Worker has committed the page. A locally produced page may
be written and consumed within the same ordered model forward; it must be
committed before a later denoising step, cache lookup, or transfer can reuse
it.

### INV-4: Source lease survives the read

A source page may not be freed, evicted, overwritten, or reused until the
sender observes a terminal acknowledgement for the transfer operation.

### INV-5: Stable and dynamic ranges are structurally separate

Stable prefix pages may survive across denoising steps. Dynamic target slots
must be overwritten for each step and must never be published as reusable
prefix pages.

### INV-6: Allocation generation rejects stale work

All page-install and transfer operations carry
`(request_id, allocation_generation)`. A completion for an older generation is
ignored and cleaned up; it can never make a new allocation ready.

### INV-7: Dense mode is unchanged

`dense_legacy` continues using the existing contiguous model-owned cache path.
No W2 component is initialized or invoked in dense mode.

### INV-8: Unsupported paged mode fails at startup

If a selected model, attention backend, platform, or connector cannot satisfy
the page-native contract, `paged_scheduler` startup fails with a specific
error. It never silently executes dense.

### INV-9: CFG branches share only semantically identical pages

Conditional and unconditional branches do not share pages merely because
their lengths match. Sharing requires identical chained semantic hashes for
complete leading blocks. Otherwise each branch receives distinct blocks.

### INV-10: Rank-local metadata must agree

Every active TP rank must receive metadata for the same public request and
allocation generation. A rank mismatch is a terminal request error, not a
partial execution.

## 6. Architecture

```text
Hunyuan preprocessing
  |
  +-- model inputs -------------------------------------------> Worker
  |
  +-- DiffusionKVRequest(s) ---------------------------------> Scheduler
                                                                 |
                                                   DiffusionKVCacheManager
                                                                 |
                                            native KVCacheManager / BlockPool
                                                                 |
                                              DiffusionKVMetadata(block_ids)
                                                                 |
                                                                 v
                                                     Worker Page Registry
                                             +-------------------+------------------+
                                             |                                      |
                                   Local page installation                  Transfer session
                                   BlockTable + slot map              SHM/direct or Mooncake
                                             |                                      |
                                             +-------------------+------------------+
                                                                 |
                                                         commit generation
                                                                 |
                                                                 v
                                                Hunyuan paged attention execution
```

The implementation adds four bounded units:

1. **Worker page registry** — allocates and indexes rank-local native page
   tensors and creates request bindings from Scheduler block IDs.
2. **Hunyuan paged cache adapter** — translates Hunyuan sequence layout into
   stable-prefix and dynamic-target slot mappings consumed by attention.
3. **Page transfer session manager** — tracks target readiness, in-flight
   operations, completion, cancellation, and source leases.
4. **Scheduler readiness integration** — distinguishes allocated requests from
   page-ready requests and schedules only the latter.

These units share descriptors and state transitions but not ownership.

## 7. Data Model

### 7.1 `DiffusionPageRange`

One contiguous logical token range mapped to one cache role.

```python
@dataclass(frozen=True)
class DiffusionPageRange:
    cache_role: str
    token_start: int
    token_count: int
    block_ids: tuple[int, ...]
    mutable: bool
```

Rules:

- `cache_role` resolves to one native cache group/layer set.
- `token_start` and `token_count` describe the request sequence axis.
- `block_ids` are Scheduler-owned physical IDs for the target Worker rank.
- `mutable=False` identifies stable context eligible for cross-step retention.
- `mutable=True` identifies per-step target slots that must be rewritten.

The final code may derive this view from `DiffusionKVSequenceMetadata` rather
than serializing a duplicate object. The semantics must remain explicit.

### 7.2 `PageEndpoint`

One rank-local transfer endpoint.

```python
@dataclass(frozen=True)
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
```

`PageEndpoint` describes an already allocated source or destination span. It
does not own the tensor and is invalid after its page lease or reservation
ends.

### 7.3 `PageTransferPlan`

One immutable transfer intent.

```python
@dataclass(frozen=True)
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
```

Each `PageCopy` pairs a source `PageEndpoint` with a destination
`PageEndpoint`. The plan includes only missing stable pages. Dynamic target
pages are local allocations and are not transferred from AR.

### 7.4 `PageTransferResult`

```python
@dataclass(frozen=True)
class PageTransferResult:
    session_id: str
    request_id: str
    allocation_generation: int
    route_epoch: int
    op_id: int
    status: Literal["completed", "failed", "cancelled", "timed_out", "stale"]
    completed_page_count: int
    transferred_bytes: int
    error: str | None
```

Completion is idempotent by
`(session_id, allocation_generation, route_epoch, op_id)`.

## 8. Worker Page Registry

### 8.1 Initialization

After #6094 installs a rank-local `KVCacheConfig`, each Worker:

1. validates that configured layer names exactly match cache-enabled attention
   modules;
2. allocates native-layout physical K/V tensors for each cache group;
3. builds block-indexed views without copying;
4. registers page geometry and device addresses;
5. creates one copy stream and event pool for local installation;
6. reports capability metadata to the Engine.

The registry is created once per loaded model and destroyed during Worker
shutdown.

### 8.2 Request binding

For each new `DiffusionKVMetadata`:

1. validate `request_id`, `allocation_generation`, sequence count, and rank
   count;
2. build a rank-local BlockTable from Scheduler block IDs;
3. derive stable-prefix and dynamic-target slot mappings;
4. create a binding in state `ALLOCATED`;
5. start required local or remote page installation;
6. commit the binding only after all required events complete.

The registry rejects:

- duplicate active generations;
- block IDs outside the configured pool;
- overlapping page ownership within one rank;
- metadata whose sequence geometry differs from preprocessed model inputs;
- an allocation with no corresponding Scheduler request.

### 8.3 Page states

Each target page has one of:

```text
FREE
RESERVED
INSTALLING_LOCAL
INSTALLING_REMOTE
COMMITTED
RELEASING
```

Only `COMMITTED` externally installed pages may be consumed as preexisting
attention K/V. Locally materialized pages may be present as writable slots in
the first-step BlockTable and become reusable only after commit. Transitioning
from `COMMITTED` to `RELEASING` removes the page from Worker request bindings
before the Scheduler releases the corresponding block to its pool.

## 9. HunyuanImage3 Execution Mapping

### 9.1 Logical sequence

For each CFG row, Hunyuan uses one primary sequence:

```text
[ stable multimodal prompt/reference prefix | mutable image target ]
```

There is one `DiffusionKVRequest` per CFG row. Hunyuan does not require an
independent `DiffusionKVContext` for its current joint self-attention layout.

### 9.2 First denoising step

The first step:

1. projects K/V for the complete logical sequence;
2. writes stable prefix K/V into stable slots;
3. writes current target K/V into mutable slots;
4. runs attention using the request BlockTable in the same ordered forward;
5. records completion for the stable-prefix writes;
6. publishes only complete stable prefix blocks as cacheable when semantic
   hashes are available.

No `image_kv_cache_map` tensor is allocated in paged mode.

### 9.3 Later denoising steps

Each later step:

1. reuses committed stable prefix slots;
2. recomputes current target K/V;
3. overwrites the mutable target slots for the same request binding;
4. executes attention over stable plus current target slots;
5. never exposes target slots through prefix-cache publication.

The target BlockTable may remain structurally stable across steps, but its
contents are step-local.

### 9.4 Imported AR K/V

When AR has already produced stable prefix pages:

- DiT allocates destination pages before transfer;
- imported pages populate only the stable prefix range;
- local first-step prompt projection is skipped only for ranges proven ready;
- the dynamic target path remains local and executes every denoising step.

If AR provides only part of the stable prefix, DiT transfers the missing page
set and computes or retrieves the remainder according to the request plan.

### 9.5 Attention backend contract

The selected backend must support:

- native page tensor layout from `KVCacheSpec`;
- block-table or slot-mapping based K/V writes;
- non-causal Hunyuan attention over stable and dynamic ranges;
- CFG batch rows;
- Hunyuan RoPE and position mapping;
- TP-local heads and rank-local pages;
- SP/AG layouts used by supported Hunyuan deployments.

Backend support is capability-checked at startup. A backend that can export a
spec but cannot execute the paged path is rejected.

## 10. Transfer Session Lifecycle

### 10.1 State machine

```text
ALLOCATED
    |
    v
TARGET_READY
    |
    +----------------------+
    | no imported pages    |
    v                      v
COMPLETED             TRANSFERRING
                           |
                           +----------------+----------------+----------------+
                           v                v                v                v
                       COMPLETED          FAILED          CANCELLED        TIMED_OUT
```

Meanings:

- `ALLOCATED`: Scheduler blocks exist; Worker binding is not yet usable.
- `TARGET_READY`: all destination endpoints are allocated and registered.
- `TRANSFERRING`: at least one page copy is in flight.
- `COMPLETED`: every externally required page is committed, or the request has
  no imported-page dependency and its writable local binding is ready.
- terminal failure states: destination pages are not exposed, target
  reservations are released, and source leases are acknowledged for cleanup.

### 10.2 Required connector operations

The Omni-facing interface exposes:

```python
prepare_receive(plan) -> ReceiveRegistration
send_pages(plan, registration) -> TransferHandle
poll_completion(handle) -> PageTransferResult | None
cancel(handle, reason) -> None
close_session(session_id) -> None
```

For W2a, SharedMemory/direct-copy implementations satisfy this interface.
For W2b, the Mooncake implementation reuses upstream paged-KV connector
capabilities beneath the Omni session interface.

### 10.3 Readiness

A public diffusion request is compute-ready only when:

- Scheduler allocation is active;
- all required Worker ranks installed the same allocation generation;
- every externally required stable page is committed;
- writable slots for locally produced stable and dynamic K/V are bound;
- all required model inputs are ready;
- no transfer dependency is failed or timed out.

Readiness is an AND across dependencies and ranks.

## 11. W2a Local Data Path

### 11.1 SharedMemory path

The existing SharedMemory connector serializes complete Python objects. W2a
adds a page-aware adapter:

1. receiver allocates final destination pages;
2. receiver publishes page descriptors and SHM registration metadata;
3. sender writes page payloads keyed by session and operation;
4. receiver copies directly from SHM into final page spans on the copy stream;
5. receiver records CUDA events and commits pages after event completion;
6. receiver returns a terminal result;
7. both sides clean SHM segments and session metadata.

The adapter may use a packed SHM payload for transport, but it must not
deserialize into a second long-lived model-owned KV object.

### 11.2 Same-process/direct-copy path

For colocated or test deployments:

1. source and destination endpoint descriptors are validated;
2. `copy_` runs on the target copy stream;
3. an event records copy completion;
4. page commit occurs only after the event reports complete.

This path is the deterministic functional reference for session semantics and
failure injection.

### 11.3 Scheduler behavior

W2a introduces explicit readiness without requiring remote transport:

- requests with no imported pages may become ready after local page binding;
  their stable pages become reusable only after the first forward commits
  locally produced K/V;
- requests with local imported pages wait for copy completion;
- a waiting request does not block a later ready request;
- capacity remains reserved while installation is active;
- cancellation releases both the request binding and Scheduler allocation.

## 12. W2b Mooncake Missing-Only Data Path

### 12.1 Rendezvous

The receiver:

1. performs local stable-prefix lookup;
2. allocates destination blocks for misses;
3. registers destination addresses;
4. sends the missing semantic page IDs, source keys, destination descriptors,
   and session identity to AR.

The sender:

1. validates the request generation and route epoch;
2. acquires source-page leases;
3. maps requested semantic pages to source blocks;
4. executes sender-push RDMA WRITE into registered destination spans;
5. reports send completion but retains leases until receiver terminal ACK.

The receiver:

1. verifies transport completion;
2. validates operation identity and expected page count;
3. commits destination pages;
4. sends terminal ACK;
5. marks the dependency complete.

### 12.2 Missing-only rule

The receiver is authoritative for the missing set. It sends only complete
stable pages absent from its local cache for the same semantic hash chain.

The sender must reject:

- a page not owned by the source request;
- a semantic page whose hash does not match the request;
- a source page outside the active lease;
- a destination span with incompatible dtype, shape, stride, or byte count.

### 12.3 Transport scope

W2b's required transport is Mooncake Transfer Engine. Mooncake Store remains a
legacy object-transfer path and is not accepted as evidence for page-native
completion, delayed free, or direct destination writes.

## 13. Scheduler Integration

### 13.1 Request states

Scheduler-visible readiness is extended conceptually to:

```text
WAITING_FOR_CAPACITY
WAITING_FOR_LOCAL_KVS
WAITING_FOR_REMOTE_KVS
READY
RUNNING
FINISHED
FINISHED_ERROR
CANCELLED
```

The existing public status enum may remain smaller if these substates are
represented in request state, but scheduling decisions and metrics must
distinguish them.

### 13.2 Admission algorithm

For each waiting request:

1. reserve Scheduler blocks atomically for all CFG rows;
2. build the stable-page lookup result;
3. create local and remote transfer plans;
4. reserve in-flight transfer bytes and destination pages;
5. start installation without entering model execution;
6. continue scanning for other ready work;
7. move the request to `READY` after all dependencies complete.

The scheduler must not preserve strict head-of-line blocking when the first
request waits on remote K/V and a later request is already ready.

### 13.3 Budgeting

Admission accounts for:

- Scheduler KV blocks;
- destination HBM bytes;
- pinned/SHM staging bytes;
- in-flight transfer bytes;
- maximum concurrent transfer sessions;
- model execution concurrency.

If a request cannot fit when the pool is empty, it fails with an actionable
admission error. If it cannot fit only because other requests hold capacity,
it remains waiting.

## 14. Failure and Cleanup Semantics

### 14.1 Cancellation

Cancellation:

1. marks the session terminal;
2. prevents future completion from committing pages;
3. asks the connector to cancel best-effort;
4. waits only for operations required to make memory reuse safe;
5. releases destination reservations;
6. acknowledges source leases for release;
7. frees Scheduler request allocation.

### 14.2 Timeout

Timeout is evaluated using monotonic deadlines. It produces a terminal request
error that includes session ID, operation ID, completed page count, expected
page count, elapsed time, and transport.

### 14.3 Late or duplicate completion

- duplicate terminal completion is idempotent;
- late completion for a closed session is classified `stale`;
- stale completion cannot commit pages;
- any transport resource associated with stale work is cleaned up.

### 14.4 Worker failure

If any participating rank fails:

- the public request fails on all ranks;
- no surviving rank may execute the partial binding;
- target reservations are released after safe stream synchronization;
- source leases are eventually released through timeout/failure
  acknowledgement;
- the Engine records the failed allocation generation.

### 14.5 Shutdown

Worker shutdown drains or cancels sessions, synchronizes outstanding copy
events required for memory safety, closes connectors, destroys page bindings,
and then releases physical cache tensors.

## 15. Observability

The implementation exposes per-request and aggregate metrics:

- stable pages requested, hit, missed, transferred, and committed;
- transferred bytes;
- local-install and remote-transfer latency;
- time in each readiness state;
- source lease duration;
- destination reservation duration;
- stale and duplicate completion counts;
- transfer timeout and cancellation counts;
- ready-queue depth;
- GPU idle time while ready queue is empty;
- Scheduler block utilization;
- HBM and staging-memory high-water marks.

Logs include:

`request_id`, `allocation_generation`, `session_id`, `route_epoch`, `op_id`,
source/target stage, TP rank, page count, bytes, state transition, and terminal
reason.

## 16. Validation Strategy

### 16.1 Unit tests

Worker page registry:

- allocates physical caches matching `KVCacheConfig`;
- rejects layer/spec mismatch;
- maps valid block IDs;
- rejects out-of-range and overlapping block IDs;
- commits only after event completion;
- rejects stale allocation generations;
- releases pages on cancellation and failure.

Hunyuan adapter:

- derives stable and dynamic ranges for one and two CFG rows;
- maps position/RoPE data to the same logical tokens as dense mode;
- reuses stable slots across steps;
- overwrites dynamic slots each step;
- never publishes target blocks;
- rejects mismatched sequence geometry.

Session manager:

- valid transition sequences;
- invalid transition rejection;
- duplicate completion idempotence;
- stale completion rejection;
- timeout cleanup;
- cancellation during each non-terminal state;
- source lease retained until terminal ACK.

Scheduler:

- active and idle DP ranks agree on readiness;
- a remote-waiting request does not head-of-line block a ready request;
- atomic CFG allocation rollback;
- impossible-fit versus temporary-capacity behavior;
- terminal cleanup frees Scheduler and Worker state.

### 16.2 Functional GPU tests

Required W2a GPU coverage:

- HunyuanImage3 dense versus paged output parity;
- first step and at least two subsequent denoising steps;
- CFG enabled and disabled;
- TP=1 and TP>1;
- active and idle DP ranks;
- EP padding where supported by the deployment;
- local imported-page transfer;
- cancellation during copy;
- non-paged model regression.

Correctness comparisons report:

- mean absolute difference;
- p99 absolute difference;
- SSIM;
- PSNR;
- deterministic seed and exact model revision;
- dense and paged memory high-water marks.

Thresholds must be defined from the repository's Hunyuan accuracy tests and
must not be weakened solely to make the paged path pass.

### 16.3 W2b distributed tests

Required W2b coverage:

- four H100 GPUs for the target deployment topology;
- cold request with all stable pages missing;
- repeated context with all stable pages hitting;
- partial-prefix hit;
- concurrent requests at concurrency 1, 4, and 8;
- active and idle DP ranks;
- TP rank-aware page routing;
- cancellation before target-ready, during transfer, and after send completion
  but before receiver ACK;
- timeout and sender/receiver process failure;
- duplicate and stale control messages.

### 16.4 End-to-end performance gate

Performance claims require controlled A/B against current dense behavior:

- same commit except the feature toggle;
- same model revision, prompts, seeds, image resolution, steps, dtype,
  parallel topology, warmup, and request count;
- at least five measured repetitions after warmup;
- report median and dispersion, not only the best run.

Required workloads:

1. **Cold context:** no local stable-page hit.
2. **Repeated context:** receiver-local stable-page hit.
3. **Mixed context:** partial hit and miss.
4. **Concurrency:** 1, 4, and 8.

Required metrics:

- end-to-end latency and throughput;
- P50/P95/P99 request latency;
- transferred bytes per request;
- transfer wait time;
- GPU idle percentage;
- Scheduler ready/waiting time;
- HBM high-water mark.

A mergeable performance claim must show a repeatable end-to-end benefit on at
least the repeated-context or mixed-context workload without a statistically
meaningful regression on cold context. Helper or routing microbenchmarks are
supporting evidence only.

## 17. Rollout

### 17.1 W2a feature gate

W2a is selected only by `diffusion_kv_mode=paged_scheduler` plus a
page-native-capable Hunyuan backend. `dense_legacy` remains the default until
functional and end-to-end gates pass.

### 17.2 W2b connector gate

Remote page transfer requires an explicit page-native connector configuration.
Selecting a legacy object connector with remote `paged_scheduler` fails at
startup with a capability error.

### 17.3 Removal of dense Hunyuan cache

`image_kv_cache_map` is retained while dense mode remains supported. It is not
removed in W2a or W2b. Removal is a separate decision after paged mode becomes
the default and migration evidence exists.

## 18. Security and Robustness

- Descriptors are validated before using offsets, shapes, strides, or byte
  counts.
- Transfer metadata cannot select arbitrary device addresses; endpoints are
  resolved through active page registrations.
- Session identity includes generation and route epoch to prevent cross-request
  reuse.
- Transport payload sizes are bounded by the reserved plan.
- Errors redact environment secrets and connector credentials.

## 19. Implementation Boundaries

Likely W2a code areas after #6094 integration:

- `vllm_omni/diffusion/diffusion_kv/` — page descriptors, registry-facing
  metadata, and lifecycle types;
- `vllm_omni/diffusion/worker/diffusion_model_runner.py` — physical cache
  initialization and request binding;
- `vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py` —
  paged attention adapter while preserving dense mode;
- `vllm_omni/diffusion/sched/` — readiness-aware admission and cleanup;
- `vllm_omni/distributed/omni_connectors/` — page-aware SHM/direct session
  adapter;
- focused unit, GPU functional, and Hunyuan E2E tests.

Likely W2b additions:

- Mooncake page endpoint registration and sender-push operations;
- source lease and receiver ACK integration;
- missing-only semantic page lookup;
- multi-node distributed correctness and performance tests.

Exact file changes belong in the implementation plan after this spec is
reviewed. No implementation should be added to the existing
`feat/llama-omni2-support` checkout.

## 20. Acceptance Criteria

W2a is complete only when:

1. Hunyuan paged mode uses real Worker `kv_caches` and Scheduler block IDs.
2. No `image_kv_cache_map` is allocated in paged mode.
3. Stable prefix pages persist across steps and dynamic slots are overwritten.
4. Attention cannot observe uncommitted pages.
5. Local page transfer follows the session/completion lifecycle.
6. Dense mode is unchanged.
7. Unit and GPU parity gates pass for the required topologies.
8. End-to-end results and validation limits are reported.

W2b is complete only when:

1. DiT performs local lookup and destination allocation before transfer.
2. Mooncake transfers only missing stable pages into final destination pages.
3. Source pages remain leased until receiver terminal ACK.
4. Scheduler admission waits for page commit without head-of-line blocking
   ready work.
5. Cancellation, timeout, duplicate, stale, and Worker-failure tests pass.
6. Four-H100 distributed correctness passes.
7. Controlled end-to-end A/B demonstrates the claimed benefit and reports
   cold-path tradeoffs.

## 21. Rejected Alternatives

### 21.1 Continue optimizing dense H2D copies

Rejected as the primary direction because it leaves model-owned contiguous
cache, FIFO transfer waits, full-cache movement, and no receiver-local lookup.
It cannot deliver missing-only page transfer or Scheduler-owned readiness.

### 21.2 Add another Worker-local allocator

Rejected because it creates conflicting Scheduler and Worker block identities,
duplicates native vLLM allocation policy, and makes cancellation and refcount
ownership ambiguous.

### 21.3 Serialize full KV objects through every connector

Rejected because it requires gather/flatten/temporary object reconstruction,
prevents direct destination writes, and cannot provide page-level
completion/visibility.

### 21.4 Implement Mooncake before local page execution

Rejected because transport cannot be validated correctly until the receiver
has final destination pages, BlockTables, and a commit-before-visibility
contract.

### 21.5 Generalize all diffusion models in W2a

Rejected because model attention topologies differ. Hunyuan provides a
concrete joint-sequence pilot; generalization should follow proven interfaces
and per-model semantic mappings.
