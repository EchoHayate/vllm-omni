# HunyuanImage3 W2a Page-Native KV End-to-End Performance

## Status

- Implementation commit: record the exact commit embedded in each JSON artifact.
- Model: `tencent/HunyuanImage-3.0-Instruct`
- Model revision: `2ec2c78bee7d4b94157341fba86c4c2c7b1858b2`
- Benchmark harness: `benchmarks/diffusion/hunyuan_image3_page_native.py`
- Formal hardware: four H100 GPUs.
- Measurement status: **pending formal H100 execution**.
- Performance claim: **none**.

The benchmark harness is committed so that dense and paged runs can be
executed from the same source revision. This document must not be changed to
claim a win until the raw JSON artifacts from the full controlled matrix are
available and reviewed.

## What W2a Measures Today

`paged_scheduler` changes persistent KV ownership:

- the Scheduler owns admission, block IDs, allocation generations, and release;
- the Worker owns physical page tensors, BlockTables, slot mappings, and copy
  completion;
- stable pages remain resident across the denoising steps of one request;
- dynamic image-token slots are overwritten on each denoising step.

The current W2a attention path still uses
`gather_hunyuan_kv_reference`. Each attention forward gathers committed page
slots into contiguous scratch key/value tensors before Torch SDPA. Therefore:

- page installation and transfer wait are distinct from attention gather;
- `reference_gather_ms` is expected to be non-zero in paged mode;
- a page-management helper improvement is not an end-to-end performance win;
- removal or fusion of the reference gather is a possible later optimization,
  not something this report assumes.

## Important Workload Boundary

The benchmark defines:

- `cold`: every request receives a distinct prompt;
- `repeated`: every request receives the same prompt;
- `mixed`: alternating repeated and distinct prompts;
- `concurrency`: one wave contains `1`, `4`, or `8` independently submitted
  requests.

For W2a, repeated prompt text does **not by itself prove a receiver-local
cross-request stable-page hit**. The current implementation demonstrates
stable-page persistence within a request, but it does not yet expose a
cross-request content-key lookup result in the benchmark output. Until such a
lookup is implemented and directly observed, `repeated` and `mixed` are input
distributions, not cache-hit claims.

## Controlled A/B Rules

Dense and paged runs must use:

1. the same repository commit;
2. the exact model revision above;
3. the same visible GPUs and tensor-parallel size;
4. the same prompts, seeds, resolution, steps, guidance scale, and dtype;
5. the same warmup count, measured repetition count, and concurrency;
6. no unrelated server or GPU workload;
7. at least five measured waves per matrix cell.

The only intended A/B difference is:

```text
diffusion_kv_mode=dense_legacy
diffusion_kv_mode=paged_scheduler
```

The formal settings are:

```text
warmup=2
repetitions=5
seed=0
height=1024
width=1024
steps=4
concurrency=1|4|8
workload=cold|repeated|mixed
mode=dense_legacy|paged_scheduler
```

## Commands

Run the full matrix from the repository root on a four-H100 worker:

```bash
set -euo pipefail

MODEL=tencent/HunyuanImage-3.0-Instruct
REVISION=2ec2c78bee7d4b94157341fba86c4c2c7b1858b2
OUT_DIR=/tmp/hunyuan-image3-w2a-$(git rev-parse --short HEAD)
mkdir -p "${OUT_DIR}"

for mode in dense_legacy paged_scheduler; do
  for workload in cold repeated mixed; do
    for concurrency in 1 4 8; do
      python benchmarks/diffusion/hunyuan_image3_page_native.py \
        --model "${MODEL}" \
        --model-revision "${REVISION}" \
        --mode "${mode}" \
        --workload "${workload}" \
        --concurrency "${concurrency}" \
        --warmup 2 \
        --repetitions 5 \
        --seed 0 \
        --height 1024 \
        --width 1024 \
        --steps 4 \
        --tensor-parallel-size 4 \
        --output-json \
          "${OUT_DIR}/${mode}-${workload}-c${concurrency}.json"
    done
  done
done
```

This produces 18 raw JSON artifacts. Preserve all of them with:

- the Buildkite URL;
- worker GPU model;
- driver and CUDA versions;
- repository commit;
- model revision;
- complete stdout/stderr.

## JSON Contract

Each measured request has:

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

Each artifact summary has:

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

The artifact also retains per-wave page-metric deltas. Worker page counters are
cumulative and rank-local:

- at concurrency `1`, their delta can be attributed to the only request;
- at concurrency `4` or `8`, the script does not divide aggregate values and
  pretend they are per-request measurements;
- in those concurrent cells, request-level transfer/gather fields are `null`
  and the exact aggregate delta remains under `waves[].page_metric_delta`.

## Metric Semantics and Limitations

| Metric | Source | Boundary |
|---|---|---|
| Request latency | Common wave submission timestamp to each final image output | True client-observed E2E latency inside one in-process `Omni` lifetime |
| Throughput | Completed measured requests divided by summed measured wave wall time | Excludes model startup and warmup |
| Transferred bytes | Delta of Worker `DiffusionPageMetrics.transferred_bytes` | Exact per request only at concurrency 1 |
| Transfer wait | Delta of Worker `local_kv_wait_s` | Exact per request only at concurrency 1 |
| Local install | Wave-level delta of Worker `local_install_latency_s` | Kept separate from transfer wait and gather |
| Reference gather | Delta of Worker `reference_gather_latency_s` | Supporting attribution for the current reference adapter |
| Scheduler waiting | Exact `scheduler_waiting_ms` if later exposed; otherwise `queue_wait_ms` | Current fallback is an orchestrator queue proxy, not exact Scheduler state residency |
| Scheduler ready | `latency - scheduler_waiting - transfer_wait` | Derived E2E residency proxy, not an internal Scheduler timestamp |
| GPU idle | Fraction of `nvidia-smi` samples whose mean visible-GPU utilization is at most 5% | Sampling-based and interval-dependent |
| HBM high water | Maximum of Worker-reported request peak and sampled summed `memory.used` | Summed across visible GPUs |

The Scheduler timing proxies must not be presented as exact internal
ready/waiting instrumentation. If exact state residency becomes required for a
performance claim, add request-scoped timestamps at the Scheduler state
transitions and propagate them to the output instead of inferring them in this
benchmark.

## Result Table

Populate this table only from the raw artifacts:

| Workload | Concurrency | Mode | QPS | Median ms | MAD ms | P50 ms | P95 ms | P99 ms | GPU idle % | HBM MB | Artifact |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cold | 1 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| cold | 1 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| cold | 4 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| cold | 4 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| cold | 8 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| cold | 8 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| repeated | 1 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| repeated | 1 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| repeated | 4 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| repeated | 4 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| repeated | 8 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| repeated | 8 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| mixed | 1 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| mixed | 1 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| mixed | 4 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| mixed | 4 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| mixed | 8 | dense | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| mixed | 8 | paged | pending | pending | pending | pending | pending | pending | pending | pending | pending |

## Claim Gate

Do not claim a performance win unless:

1. repeated or mixed context improves repeatably across the raw repetitions;
2. the improvement exceeds run-to-run dispersion;
3. cold context has no statistically meaningful regression;
4. output parity and four-H100 topology tests pass on the same commit;
5. the result is end-to-end, not inferred from local install or gather helpers.

With no formal H100 artifacts attached, the current conclusion is:

> The W2a benchmark protocol and machine-readable evidence path exist, but
> end-to-end performance acceptance remains `NO_GO` and no speedup is claimed.
