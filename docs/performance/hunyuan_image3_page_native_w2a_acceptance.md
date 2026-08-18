# HunyuanImage3 Page-Native KV W2a Acceptance Audit

## Audit Identity

- Audit date: August 18, 2026
- Restack verification head: `b97cd67624d82190c4e7210d587dafe551ea3e0a`
- Restack base: `4cd667875f983819414ca6a19cf612b58e6c138e`
- Pull request: `vllm-project/vllm-omni#6219`
- Former control-plane dependency: `vllm-project/vllm-omni#6094`, merged as
  `0a1846aa458283a13c1a7b5a96ab585937f9726c`
- Model: `tencent/HunyuanImage-3.0-Instruct`
- Model revision: `2ec2c78bee7d4b94157341fba86c4c2c7b1858b2`
- Overall status: **NO_GO**

`NO_GO` means the branch is not yet accepted as complete W2a evidence. It does
not mean the local CPU implementation tests failed. The release blockers are
the missing formal four-H100 run and missing controlled dense/paged benchmark
artifacts. The former stacked dependency has merged and its two commits are no
longer part of the PR-exclusive history.

## Requirement-to-Evidence Matrix

| Requirement | Code Evidence | Test Evidence | External Gate | Status |
|---|---|---|---|---|
| Real Worker `kv_caches` use Scheduler block IDs | `vllm_omni/diffusion/diffusion_kv/worker_registry.py` validates `DiffusionKVMetadata` block IDs, builds native BlockTables, maps owned blocks to rank-local cache-group storage, and installs that storage on attention layers | `test_build_slot_mapping_uses_scheduler_block_ids`, `test_diffusion_kv_metadata_uses_native_cache_group_block_ids`, Worker registry tests | Required H100 full-model log is absent | **NO_GO: external gate missing** |
| No `image_kv_cache_map` or dense AR snapshot in paged mode | `hunyuan_image3_transformer.py::_forward_paged` requires live bindings and installed pages; `pipeline_hunyuan_image3.py` snapshots/restores `_STEP_AR_KV` and `_STEP_PROMPT_KV` only outside paged mode | `test_paged_first_step_does_not_allocate_image_kv_cache_map`, `test_paged_prepare_encode_stores_no_dense_ar_kv_snapshot`, `test_paged_first_step_uses_live_binding_without_prompt_kv_snapshot` | Required dense/paged H100 parity log is absent | **NO_GO: external gate missing** |
| Stable pages persist and dynamic slots overwrite | `paged_kv.py` builds stable/dynamic masks and writes Scheduler-selected slots; Worker binding persists across cached steps | `test_second_step_reuses_stable_slots_and_overwrites_dynamic_slots`, `test_paged_matches_dense_for_first_and_two_later_steps`, `test_cached_step_reuses_the_original_page_binding` | Required full-model multi-step parity log is absent | **NO_GO: external gate missing** |
| Uncommitted external pages are invisible | `page.py` exposes only committed external pages as reusable; `transfer.py` commits after terminal copy completion and rejects stale generations | `test_only_committed_external_pages_are_reusable`, `test_imported_page_must_be_committed_before_batch_is_compute_ready`, `test_direct_copy_commits_only_after_event_completion`, stale/cancel/timeout tests | CUDA local-import/cancel test is defined but has no H100 execution log | **NO_GO: external gate missing** |
| Local direct/SHM transfer uses terminal completion | `transfer.py` waits for the copy event before commit/cancel/close; `page_shm.py` copies into preallocated destinations and returns terminal transfer results | `test_pack_two_page_spans_and_copy_into_preallocated_pages`, duplicate/cancellation/sender-failure SHM tests, `test_close_waits_for_in_flight_copy_before_cancelling` | CUDA partial-prefix/cancel test is defined but has no H100 execution log | **NO_GO: external gate missing** |
| Dense mode remains unchanged and does not initialize W2a | Dense branches retain `image_kv_cache_map`; runner capability and data-plane initialization are gated on `paged_scheduler` | `test_dense_first_step_still_populates_image_kv_cache_map`, `test_dense_mode_rejects_page_data_plane_initialization`, `test_dense_mode_skips_worker_kv_initialization`, `test_dense_mode_ignores_page_native_capability` | Required dense H100 accuracy case is absent | **NO_GO: external gate missing** |
| Required topology parity passes | Scheduler owns DP rank until release ACK; executor dispatches only the selected non-EP DP replica; idle workers skip model execution | Scheduler DP ownership tests, paged DP worker idle tests, multiprocess dispatch tests; formal test includes TP1, TP2, and DP2×TP2 active/idle | No four-H100 Buildkite context exists on PR #6219 | **NO_GO: formal topology gate missing** |
| End-to-end results and limits are reported | `benchmarks/diffusion/hunyuan_image3_page_native.py` emits request records, wave deltas, percentiles, median/MAD, sampled GPU idle, and HBM high water; report documents reference gather and attribution limits | `tests/benchmarks/test_hunyuan_image3_page_native.py` validates CLI rejection, statistics, and JSON schema | No 18-cell dense/paged raw JSON matrix is attached | **NO_GO: benchmark artifacts missing** |

## Persistent-Cache Ownership Audit

Command:

```bash
rg -n "image_kv_cache_map|_STEP_PROMPT_KV|_STEP_AR_KV" \
  vllm_omni/diffusion/models/hunyuan_image3
```

Findings:

1. `image_kv_cache_map` remains in the transformer because `dense_legacy` is
   still supported.
2. `_restore_prompt_kv_cache` and `_capture_prompt_kv_cache` are dense-only
   step-state helpers.
3. `_STEP_AR_KV` is populated only when
   `HunyuanImage3Pipeline._uses_paged_kv()` is false.
4. Paged validation requires live Worker page bindings and rejects dense
   injected AR K/V.
5. Paged attention reads `Attention.paged_kv_cache`, which is installed from
   the Worker registry's rank-local `kv_caches`.

No paged assignment to `image_kv_cache_map` or paged request-state tensor
snapshot was found. This satisfies the code-review portion only; the H100
execution evidence remains missing.

## No-Silent-Fallback Audit

Command:

```bash
rg -n "paged_scheduler|dense_legacy" \
  vllm_omni/diffusion \
  tests/diffusion
```

Findings:

- startup validation permits only HunyuanImage3, CUDA, Torch SDPA, direct/SHM
  page adapters, and supported topology;
- unsupported combinations raise `ValueError` with model, backend, platform,
  connector, and topology details;
- missing native cache-enabled attention layers raise `RuntimeError`;
- paged requests missing metadata, bindings, page storage, row identity, or a
  supported layout fail instead of selecting `dense_legacy`;
- dense requests carrying Scheduler-only metadata are rejected.

The AR reuse log message that says "fallback to full recompute" belongs to the
legacy dense AR reuse path. The paged path explicitly rejects injected dense AR
K/V and does not catch the capability errors to change modes.

## Local Verification Evidence

### Benchmark red/green contract

Initial RED:

```text
3 failed
missing benchmark module:
benchmarks/diffusion/hunyuan_image3_page_native.py
```

GREEN:

```bash
pytest -q tests/benchmarks/test_hunyuan_image3_page_native.py
```

```text
3 passed
```

### Focused W2a regression set

```bash
pytest -q \
  tests/benchmarks/test_hunyuan_image3_page_native.py \
  tests/diffusion/diffusion_kv \
  tests/diffusion/test_diffusion_scheduler.py \
  tests/diffusion/test_multiproc_engine_concurrency.py
```

```text
305 passed
```

### Full local acceptance set on vLLM 0.27

The existing virtual environment initially resolved the editable vLLM source
from `/private/tmp/vllm-v0.26.0`. Collection then failed because vLLM-Omni
commit `2ce15ab7` rebased main to vLLM 0.27 and imports
`FusedMoEFactory`, which vLLM 0.26 does not provide.

An independent vLLM worktree was created without modifying the retained 0.26
checkout:

```text
/private/tmp/vllm-v0.27.0
vLLM tag v0.27.0
commit 4bdc8a788d2e2ce9165d552b3d4d8b72604626bf
```

The two files that failed collection under 0.26 passed under 0.27:

```text
19 passed
```

The first complete 0.27 run then exposed a real W2a regression in five
pre-existing batch-runner tests: lightweight test runners do not initialize
`page_registry`, while `_execute_request_list` directly read that optional
attribute. Commit `eb3d509c` changed the read to `getattr` without changing a
fully initialized production runner. The five-test RED/GREEN result was:

```text
before: 5 failed
after:  5 passed
```

Final command:

```bash
PYTHONPATH=/private/tmp/vllm-v0.27.0 \
  pytest -q \
    tests/diffusion/diffusion_kv \
    tests/diffusion/models/hunyuan_image3 \
    tests/diffusion/test_diffusion_scheduler.py \
    tests/diffusion/test_diffusion_model_runner.py \
    tests/distributed/omni_connectors/test_page_shm.py \
    tests/distributed/omni_connectors/test_shm_connector.py \
    tests/benchmarks/test_hunyuan_image3_page_native.py
```

```text
543 passed, 7 skipped
```

After restacking on current main, two additional tests failed because their
fixtures still assumed pre-batching/pre-readiness contracts:

- the memory-profile test passed one request directly to the now-batched
  `profile_run(requests)` API;
- the capacity-under-load test expected a newly allocated request to enter
  compute before the Worker reported its page installation as `ready`.

Commit `b97cd676` updates only those fixtures. It preserves the production
contract that allocation emits `page_install_reqs`, Worker readiness is
reported through `WorkerKVUpdate(status="ready")`, and only a later schedule
dispatches compute. Fresh restack verification produced:

```text
305 passed
543 passed, 7 skipped
```

### Changed-file hooks

```bash
uvx --from pre-commit==4.0.1 pre-commit run --files \
  benchmarks/diffusion/hunyuan_image3_page_native.py \
  tests/benchmarks/test_hunyuan_image3_page_native.py \
  docs/performance/hunyuan_image3_page_native_w2a.md
```

All changed-file hooks passed.

These local results do not replace the hardware gates in the matrix.

## Live External State

State refreshed on August 18, 2026 before publishing the restacked head:

### PR #6219

- state: open, ready for review;
- remote head: `9c5be4dd31a2efc60565f12744be4b6c7989f0fe`
  at the time of the live query;
- local restack verification head: `b97cd67624d82190c4e7210d587dafe551ea3e0a`;
- DCO: success;
- Read the Docs: success on the old remote head;
- review: required;
- merge state: conflicting on the old remote head; the local branch has been
  rebuilt on current main and removes the merged #6094 commits from the
  PR-exclusive range;
- Buildkite H100 context: absent;
- maintainer label request for `cuda-test` and `nightly-test`: posted, no
  response yet.

Publishing the restacked head will refresh lightweight CI and mergeability. It
does not add hardware evidence.

### Dependency PR #6094

- state: merged on August 18, 2026 at `07:00:24Z`;
- merge commit: `0a1846aa458283a13c1a7b5a96ab585937f9726c`;
- the W2a branch has been restacked onto main after that merge;
- the old #6094 commits are absent from the PR-exclusive history.

## Missing Evidence and Exact Closure Actions

### 1. Formal four-H100 correctness/topology gate

Owner needed: vLLM-Omni maintainer with label/Buildkite permission.

Trigger `cuda-test` and `nightly-test` on PR #6219, then run:

```bash
pytest -s -v tests/e2e/accuracy/test_hunyuan_image3_paged_kv.py \
  -m 'full_model and cuda and H100' \
  --run-level full_model
```

Required saved evidence:

- Buildkite URL;
- exact tested commit;
- `h100_4` worker identity;
- raw log for all six formal tests;
- dense/paged accuracy metrics;
- active/idle DP rank snapshots;
- partial-prefix import and cancellation result.

### 2. Controlled end-to-end dense/paged matrix

Owner needed: four-H100 benchmark worker or Buildkite step with artifact
upload.

Run the 18 commands documented in
`docs/performance/hunyuan_image3_page_native_w2a.md`:

```text
2 modes × 3 workloads × 3 concurrency levels
```

Required saved evidence:

- 18 raw JSON artifacts;
- complete stdout/stderr;
- same commit/model revision/topology for every A/B pair;
- at least five measured waves per cell;
- populated result table;
- cold-path regression analysis;
- no speedup claim based only on install/gather helpers.

The repeated and mixed labels currently describe prompt distributions. A
receiver-local cross-request page hit must not be claimed without a directly
observed content-key hit signal.

### 3. Control-plane dependency — complete

#6094 merged and #6219 was rebuilt on current main. The fresh local restack
sets are green. This item is no longer an acceptance blocker.

### 4. Exact Scheduler timing, if required for a performance claim

The current benchmark reports:

- an exact internal Scheduler metric if `scheduler_waiting_ms` becomes
  available;
- otherwise an orchestrator `queue_wait_ms` proxy;
- a derived `scheduler_ready_ms` E2E residency proxy.

If exact Scheduler state residency is required, add request-scoped transition
timestamps to `WAITING_FOR_LOCAL_KVS`, `READY`, and `RUNNING`, propagate them to
the request output, and update the JSON artifact. Do not relabel the current
proxy as exact instrumentation.

## Final Decision

**NO_GO**

Direct local code and CPU/static evidence cover the W2a ownership, lifecycle,
fail-fast, dense separation, and benchmark-schema contracts. Acceptance is
still blocked because:

1. PR #6219 has no formal four-H100 Buildkite result;
2. the controlled 18-cell end-to-end A/B artifacts do not exist;
3. no cross-request receiver-local hit signal exists for a repeated-context
   performance claim.

Set this audit to `GO` only after every matrix row has direct external evidence
on the rebased implementation commit. CPU tests, helper timings, an A100 smoke,
or DCO success are not substitutes.
