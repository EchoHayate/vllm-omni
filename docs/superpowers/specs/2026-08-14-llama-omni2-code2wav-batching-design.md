# LLaMA-Omni2 Code2Wav Batching Design

**Date:** 2026-08-14

**Status:** Approved for implementation

## 1. Goal

Rebuild PR #5556 on the latest `origin/main`, preserve its native
LLaMA-Omni2 Thinker → Talker → Code2Wav model support, and replace the
Code2Wav stage's cumulative single-request synthesis with request-owned
incremental state and true GPU batching.

The change is successful only when the online three-stage deployment shows a
measurable end-to-end concurrency gain. A helper-only or isolated
microbenchmark improvement is not sufficient.

## 2. Success Criteria

### 2.1 Performance

- At concurrency 1, median TTFP and median RTF must not regress by more than
  5% relative to the rebuilt pre-batching implementation.
- At concurrency 4 or 8, either audio throughput or median RTF must improve by
  at least 10%.
- Each reported point must use at least three controlled runs with identical
  prompts, sampling parameters, codec limits, deployment topology, and GPU
  assignment.
- The report must include the median and dispersion for each metric rather
  than a single best run.
- A profiler trace or operator summary must show that the gain comes from
  Code2Wav batching: encoder, CFM estimator, and HiFT calls must execute with
  batch size greater than one for concurrent requests in the same shape
  bucket.

### 2.2 Correctness

- Output sample rate remains 24 kHz.
- Every emitted audio tensor is finite.
- Request finish reason and terminal-chunk behavior remain unchanged.
- Codec units are consumed exactly once and in order.
- Streaming output remains continuous across chunk boundaries.
- A final chunk flushes retained overlap exactly once.
- Cancellation and failed batches release request-owned state.
- Concurrent requests cannot observe or mutate one another's Flow or HiFT
  caches.
- The batched and sequential reference paths produce matching output lengths
  and numerically close waveforms under deterministic settings.

## 3. Current Problem

The implementation in the existing #5556 branch keeps all codec units for a
request and calls a monkey-patched `flow.inference()` on the complete prefix
for every emitted chunk. It then removes already-consumed mel frames with a
token offset before running HiFT.

For a request whose emitted codec windows have lengths `C, 2C, 3C, ...`, Flow
therefore processes `C + 2C + 3C + ...` units instead of processing each unit
once. The implementation also loops over requests in Python, runs Flow and
HiFT separately for every request, and copies each audio result to CPU inside
the request loop. This prevents the Code2Wav stage from benefiting from
vLLM-Omni's multi-request scheduling.

The old loader is part of the problem. It directly instantiates
`flashcosyvoice.modules.flow.CausalMaskedDiffWithXvec` and monkey-patches an
`inference()` method around its full `forward()` method. That bypasses the
incremental API exposed by the actual checkpoint-configured CosyVoice2 Flow.

## 4. Verified Runtime Contracts

The Step-Audio2/CosyVoice2 runtime used by the decoder exposes:

```python
flow.setup_cache(
    token: Tensor,  # [batch, prompt_tokens + lookahead]
    mel: Tensor,    # [batch, prompt_mel_frames, mel_channels]
    spk: Tensor,    # [batch, 192]
    n_timesteps: int = 10,
) -> dict[str, Tensor]
```

and:

```python
flow.inference_chunk(
    token: Tensor,  # [batch, chunk_tokens]
    spk: Tensor,    # [batch, 192]
    cache: dict[str, Tensor],
    last_chunk: bool = False,
    n_timesteps: int = 10,
) -> tuple[Tensor, dict[str, Tensor]]
```

The four Flow cache tensors use different batch axes:

| Cache | Logical shape | Batch axis |
|---|---|---|
| `conformer_cnn_cache` | `[batch, channels, width]` | 0 |
| `conformer_att_cache` | `[depth, batch, heads, time, width]` | 1 |
| `estimator_cnn_cache` | `[steps, depth, 2*batch, channels, width]` | 2 |
| `estimator_att_cache` | `[steps, depth, 2*batch, heads, time, width]` | 2 |

The estimator's `2*batch` axis is classifier-free guidance layout:

```text
[conditional request 0 ... conditional request B-1,
 unconditional request 0 ... unconditional request B-1]
```

It must not be stacked as adjacent per-request pairs.

The upstream CFM implementation contains fixed internal cache buffers sized
for its original single-request path. Calling upstream `inference_chunk()`
with a larger batch would therefore not be a safe true-batching
implementation. The backend must use the same dynamic-buffer CFM stepping
pattern already proven by the MiniCPM-o 4.5 Code2Wav implementation on current
main.

HiFT accepts batched mel and source cache tensors:

```python
hift(
    speech_feat: Tensor,   # [batch, mel_channels, mel_frames]
    cache_source: Tensor,  # [batch, 1, source_frames]
) -> tuple[Tensor, Tensor]
```

The streaming overlap contract is:

- retain 8 mel frames;
- retain `8 * 480 = 3840` generated source samples;
- retain the same number of waveform samples for cross-fade;
- withhold the retained waveform tail for non-final chunks;
- emit the complete waveform on the final chunk.

## 5. Chosen Architecture

### 5.1 Rebuild Instead of Merge

Create the replacement branch from current `origin/main`. Port the two useful
old commits—the model-support design and implementation—without carrying the
old merge commit. Resolve each conflict against current runtime contracts
rather than preserving obsolete code mechanically.

This keeps #5556 reviewable and avoids retaining a branch that is already far
behind main.

### 5.2 Load the Checkpoint-Configured Flow

The decoder loader will:

1. resolve the local or Hugging Face decoder directory;
2. validate the YAML, `flow.pt`, and `hift.pt` assets;
3. load the YAML with the runtime's supported HyperPyYAML loader;
4. obtain the configured `flow` object, including its real encoder, decoder,
   lookahead width, upsample ratio, and incremental methods;
5. load `flow.pt` and `hift.pt` strictly;
6. move both modules to the stage device and switch them to evaluation mode.

The implementation will not monkey-patch full-prefix `flow.forward()` as an
incremental interface.

The loader will accept the decoder's actual YAML filename from configuration
and will retain compatibility with the filename used by the existing #5556
recipe.

### 5.3 Shared CosyVoice2 Batched Backend

Current main already contains the required dynamic batching machinery in the
MiniCPM-o 4.5 Code2Wav backend:

- dynamic CFM buffers;
- true batched classifier-free guidance;
- request-state stack and split;
- batched HiFT;
- request-owned Flow and HiFT cache tensors;
- deterministic cache-shape signatures.

The generic machinery will move to a model-neutral CosyVoice2 backend module.
The existing MiniCPM-o import path will remain as a compatibility re-export so
that this PR does not force unrelated call-site churn.

The shared backend will expose model-neutral errors and accept:

- a loaded token-to-wave asset object or explicit Flow/HiFT modules;
- prompt features or an empty-prompt feature bundle;
- `n_timesteps`;
- mel/source overlap lengths;
- optional CUDA Graph wrappers.

MiniCPM-o behavior must remain unchanged and its existing batching suite must
continue to pass.

LLaMA-Omni2 will use an empty prompt token/mel sequence plus the packaged
default English speaker embedding. It will initialize the Flow cache once per
new request and then process only newly arrived codec units.

### 5.4 Request-Owned State

Each active request owns:

```python
@dataclass(frozen=True)
class LlamaOmni2Code2WavState:
    chunk_seq: int
    pending_tokens: Tensor
    token2wav: BatchedToken2WavState
```

`BatchedToken2WavState` contains:

```python
flow_cache: dict[str, Tensor]
hift_cache: dict[str, Tensor]
```

Every per-request cache tensor has logical batch size one. A forward pass
stacks selected states into a temporary batch, executes the model once, and
splits the resulting cache back into independent cloned request states.

The state map is keyed by the runtime request ID, not by output position.

State mutation is transactional:

1. parse and validate all work items;
2. build temporary batched states;
3. run Flow and HiFT;
4. split and validate outputs;
5. commit all new request states together.

If model execution or output validation fails, the previous state map remains
unchanged. Newly created temporary state is discarded.

### 5.5 Exact-Shape Bucketing

Flow and HiFT are batched only when requests share all execution-relevant
properties:

```text
(
  codec_chunk_length,
  last_chunk,
  flow_cache_shape_signature,
  hift_cache_shape_signature,
  device,
  dtype,
)
```

The implementation does not pad arbitrary codec lengths into one batch.
Padding would alter encoder lookahead behavior and CFM cache lengths, and it
would make waveform trimming dependent on model internals.

Non-final chunks shorter than or equal to the Flow lookahead width remain in
`pending_tokens`. They are combined with the next producer chunk. A final
chunk may flush a shorter pending suffix using the encoder's final-chunk
padding behavior.

Requests in a singleton bucket still use the same incremental backend. There
is no separate cumulative-prefix fallback.

### 5.6 Runtime Payload Contract

The Code2Wav model will consume the current generation-runner inputs:

- flattened `input_ids`;
- `seq_token_counts` to recover one codec segment per scheduled sequence;
- `runtime_additional_information`;
- runtime `request_ids` when available.

For each segment it will normalize:

- `request_id`;
- `chunk_seq`;
- `last_chunk` or the existing #5556 terminal flag;
- codec tensor from `codes.audio`, falling back to the segment only when the
  producer payload does not provide explicit codec data.

The output preserves one slot per input sequence, including buffered chunks
that do not yet produce audio. Each produced slot contains:

- audio tensor;
- sample rate 24000;
- terminal flag;
- sequence index;
- consumed codec-unit count;
- codec delta required by the downstream output path.

Audio remains on the accelerator through Flow and HiFT. CPU transfer occurs
only at the existing output serialization boundary, not inside the
per-request synthesis loop.

### 5.7 Lifecycle

- **First chunk:** create Flow and HiFT state, append codec units, decode if
  enough lookahead is available.
- **Middle chunk:** validate monotonic `chunk_seq`, append only new units,
  decode an exact-shape batch, and commit split caches.
- **Final chunk:** decode all pending units with final encoder behavior, emit
  retained overlap, mark the slot terminal, and remove request state.
- **Empty final chunk:** flush pending lookahead if present; otherwise emit an
  empty finite audio tensor and remove state.
- **Cancellation:** remove request state and any request-owned prompt feature
  entry.
- **Duplicate/reordered chunk:** raise a structured error without mutating
  state.
- **Batch failure:** roll back the entire bucket.

## 6. Deployment Changes

The rebuilt deployment will raise Code2Wav stage admission above one sequence
so the model runner can present concurrent requests to one forward call.
Thinker and Talker limits will be changed only where required to admit the
chosen end-to-end benchmark concurrency without changing model semantics.

The deployment YAML will expose:

- Code2Wav maximum sequences;
- codec chunk size and lookahead contract;
- minimum batching threshold if the scheduler supports delayed admission;
- optional Flow/HiFT CUDA Graph controls, disabled by default for the first
  correctness and E2E comparison.

The first performance claim will come from eager dynamic batching. CUDA Graph
capture is a separate optimization and must not be required to demonstrate
the batching benefit.

## 7. Benchmark Design

### 7.1 Harness

Extend the repository's TTS benchmark for LLaMA-Omni2's
`/v1/chat/completions` audio response path using the `openai-chat-omni`
request format. Do not route this model through `/v1/audio/speech`.

The harness will record:

- request latency;
- TTFP or time to first audio bytes;
- generated audio duration;
- RTF;
- aggregate audio seconds per wall-clock second;
- request success/failure;
- finish reason;
- sample rate;
- non-finite sample count;
- audio underrun count when available.

### 7.2 Controlled Matrix

Run the rebuilt sequential reference and batched implementation with:

- the same current-main commit plus the same ported model-support code;
- the same model and decoder snapshots;
- fixed prompts and fixed random seed;
- identical Thinker and Talker sampling settings;
- concurrency 1, 4, and 8;
- at least three measured runs after warmup;
- the same GPU placement and no overlapping benchmark jobs.

The report will include GPU model, driver, CUDA, PyTorch, vLLM, vLLM-Omni,
model revision, decoder revision, command lines, raw result files, and
profiler commands.

### 7.3 Attribution

Use PyTorch profiler or Nsight Systems around the Code2Wav stage. The evidence
must include:

- Flow encoder call count and observed batch sizes;
- CFM estimator call count and observed CFG batch sizes;
- HiFT call count and observed batch sizes;
- host-side per-request loop removal;
- CPU transfer location;
- GPU utilization or kernel occupancy comparison where available.

## 8. Test Strategy

### 8.1 Shared Backend Tests

- true batch executes one encoder call with batch `B`;
- CFM executes with CFG batch `2B`;
- HiFT executes once with batch `B`;
- cache stack/split preserves conditional and unconditional row ordering;
- split states do not alias one another;
- short non-final chunks are rejected by the backend boundary;
- final short chunks flush correctly;
- mixed cache shapes are not combined;
- sequential and batched deterministic outputs are close.

### 8.2 LLaMA-Omni2 Model Tests

- loader constructs Flow from YAML and does not monkey-patch full `forward()`;
- multiple request payloads produce one true batch;
- request IDs remain isolated when output slots are reordered;
- pending lookahead survives across producer chunks;
- duplicate and reordered chunks fail without state mutation;
- mixed final/non-final requests form separate buckets;
- empty final chunks flush or close correctly;
- cancellation removes state;
- a failed bucket rolls back all members;
- output slots, codec deltas, consumed counts, sample rate, and terminal flags
  remain aligned.

### 8.3 Regression Tests

- all existing MiniCPM-o Code2Wav batching tests;
- all ported #5556 LLaMA-Omni2 unit tests;
- focused scheduler, connector, and output-accumulation tests touched by the
  model-support port;
- repository formatting and static checks for changed files.

### 8.4 Real-Model Validation

- decoder-only CUDA smoke with the real Flow/HiFT assets;
- sequential-versus-batched deterministic waveform comparison;
- online Thinker → Talker → Code2Wav streaming smoke;
- concurrent online E2E benchmark;
- profiler capture proving batched Code2Wav execution.

## 9. Alternatives Rejected

### 9.1 Keep Full-Prefix Recompute and Only Batch Requests

Rejected because it preserves quadratic work across streaming chunks and
requires padding requests with different cumulative lengths. It would improve
kernel occupancy while retaining the dominant redundant computation.

### 9.2 Call Upstream `flow.inference_chunk()` Directly With Batch > 1

Rejected because the verified upstream CFM uses fixed internal cache buffers
for its original path. A larger batch would either exceed those buffers or
retain incorrect CFG cache ordering. Dynamic CFM stepping is required.

### 9.3 Pad All Active Requests to the Longest Codec Chunk

Rejected because padding changes lookahead and cache-length semantics and
complicates exact audio trimming. Exact-shape buckets are simpler and preserve
the model contract.

### 9.4 Optimize Only a Local Helper Microbenchmark

Rejected because the merge decision depends on real three-stage model risk and
real user-visible throughput. Microbenchmarks may diagnose a hotspot but
cannot satisfy the performance gate.

## 10. Scope Boundaries

This work includes:

- rebuilding #5556 on current main;
- incremental and batched LLaMA-Omni2 Code2Wav;
- shared CosyVoice2 batching extraction where needed;
- focused tests;
- deployment settings;
- a reproducible LLaMA-Omni2 E2E benchmark and profiler evidence.

This work does not include:

- changing LLaMA-Omni2 model quality or sampling defaults;
- quantizing Flow or HiFT;
- making CUDA Graph capture mandatory;
- unrelated scheduler refactoring;
- claiming a performance improvement before controlled real-model results
  meet the stated gate.
