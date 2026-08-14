# LLaMA-Omni2 Code2Wav Batching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild PR #5556 on current main and deliver request-owned incremental LLaMA-Omni2 Code2Wav synthesis with true exact-shape GPU batching and controlled end-to-end evidence.

**Architecture:** Port the existing native three-stage model support onto current main, replace the full-prefix decoder wrapper with the checkpoint-configured incremental CosyVoice2 Flow, and share current main's dynamic CFM/HiFT batching machinery through a model-neutral backend. LLaMA-Omni2 request state is parsed transactionally, grouped by exact execution shape, stacked for one GPU call, split into independent caches, and validated with local tests plus real 8×A100 serving runs.

**Tech Stack:** Python 3.10+, PyTorch, vLLM, vLLM-Omni multi-stage runtime, HyperPyYAML, CosyVoice2 Flow/HiFT, pytest, Ruff, `vllm bench serve --omni`, PyTorch profiler/Nsight Systems.

## Global Constraints

- Base all implementation commits on `origin/main` commit `728640f4d9bb7c1b68d646e2a4c59ce1ce45de9c` or a newer explicitly refreshed main.
- Preserve the existing #5556 Thinker → Talker → Code2Wav model semantics and 24 kHz audio contract.
- Do not use cumulative full-prefix Flow recomputation as a fallback.
- Batch only exact-shape work; do not pad arbitrary codec lengths.
- Keep request-owned Flow and HiFT caches isolated and transactionally committed.
- At concurrency 1, median TTFP and median RTF may not regress by more than 5%.
- At concurrency 4 or 8, audio throughput or median RTF must improve by at least 10%.
- Run at least three controlled measurements per reported point and include dispersion.
- Correctness parity and profiler attribution are required before making an E2E performance claim.
- Use `sitian@10.232.195.203` through `/tmp/ssh-sitian-10.232.195.203`; do not create another remote environment or duplicate model downloads.
- Every commit message must end with exactly one `Co-authored-by: TRAE CLI <noreply@bytedance.com>` trailer.

---

### Task 1: Port Native LLaMA-Omni2 Support Onto Current Main

**Files:**
- Port: commit `ad539adf096d0e809b0892aea585640c7f4d6feb`
- Create: `vllm_omni/model_executor/models/llama_omni2/`
- Create: `vllm_omni/model_executor/stage_input_processors/llama_omni2.py`
- Create: `vllm_omni/transformers_utils/configs/llama_omni2.py`
- Create: `vllm_omni/deploy/llama_omni2.yaml`
- Create: `tests/model_executor/models/llama_omni2/`
- Modify: current registry, scheduler, connector, runner, output, docs, and recipe files touched by the old commit

**Interfaces:**
- Consumes: current-main `PipelineConfig`, `StagePipelineConfig`, async chunk connector, AR runner, generation runner, and multimodal output accumulation interfaces.
- Produces: registered `Omni2Speech2SQwen2ForCausalLM` pipeline with native Thinker, Talker, and `LlamaOmni2Code2Wav` stages on current main.

- [ ] **Step 1: Apply the old model-support commit without committing**

Run:

```bash
git cherry-pick --no-commit ad539adf096d0e809b0892aea585640c7f4d6feb
```

Expected: newly created LLaMA-Omni2 files apply; current-main integration files report conflicts.

- [ ] **Step 2: Resolve current-main integration conflicts**

Use current main as the structural authority, then add only the
LLaMA-Omni2-specific cases. Resolve the known conflict groups as follows:

```text
docs/models/supported_models.md
recipes/README.md
    Add one LLaMA-Omni2 row without removing newer model rows.

tests/engine/test_arg_utils.py
vllm_omni/engine/arg_utils.py
    Add the LLaMA-Omni2 architecture mapping to the current mapping.

vllm_omni/config/pipeline_registry.py
vllm_omni/model_executor/models/registry.py
    Import and register the new pipeline/model classes alongside current models.

vllm_omni/core/sched/omni_ar_scheduler.py
vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py
vllm_omni/worker/gpu_ar_model_runner.py
vllm_omni/worker/omni_connector_model_runner_mixin.py
    Preserve newer scheduling and connector behavior; add only the explicit
    LLaMA-Omni2 stage contract required by its async chunk path.

vllm_omni/outputs/multimodal_accumulation.py
    Merge LLaMA-Omni2 metadata into the current metadata-key set.

tests/test_config_factory.py
    Move the old test cases into `tests/config/test_config_factory.py`, which
    is the current-main config-factory test module.
```

Remove the old repository-level validation scaffolding if it duplicates
current CI conventions:

```bash
git rm -f Dockerfile.validation run_validation.sh
```

Keep the focused model tests and E2E harnesses.

- [ ] **Step 3: Confirm that no conflict markers or accidental deletions remain**

Run:

```bash
git diff --check
rg -n '^(<<<<<<<|=======|>>>>>>>)' . --glob '!*.lock'
git status --short
```

Expected: no conflict markers; all paths are staged or modified intentionally.

- [ ] **Step 4: Run import and config collection tests**

Run:

```bash
python -m pytest \
  tests/model_executor/models/llama_omni2/test_config.py \
  tests/model_executor/models/llama_omni2/test_pipeline.py \
  tests/model_executor/models/llama_omni2/test_model_wrapper.py \
  -q
```

Expected: tests collect on current main. Failures caused by the old
Code2Wav implementation are allowed only in `test_code2wav.py`; registry or
config failures must be resolved before continuing.

- [ ] **Step 5: Commit the current-main model-support port**

Run:

```bash
git add -A
git commit -m "feat(models): port native LLaMA-Omni2 support" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

Expected: one port commit with no merge commit and no unresolved files.

### Task 2: Extract a Model-Neutral CosyVoice2 Batched Backend

**Files:**
- Create: `vllm_omni/model_executor/models/common/cosyvoice2_batched_token2wav.py`
- Modify: `vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py`
- Modify: `vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py`
- Create: `tests/model_executor/models/common/test_cosyvoice2_batched_token2wav.py`
- Modify: `tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py`

**Interfaces:**
- Consumes: loaded Flow and HiFT modules exposing encoder, decoder, speaker projection, input embedding, and HiFT inference.
- Produces:
  - `CosyVoice2PromptFeatures`
  - `CosyVoice2BatchedState`
  - `CosyVoice2BatchedToken2Wav.setup_batch(features, batch_size)`
  - `CosyVoice2BatchedToken2Wav.decode_batch(tokens, features, states, *, last_chunk, flush_encoder=False)`
  - `cosyvoice2_state_shape_signature(state)`
- Compatibility: old MiniCPM-o names remain importable aliases.

- [ ] **Step 1: Add failing shared-backend import and batching tests**

Create a CPU fake Flow/HiFT fixture and assert the model-neutral API:

```python
from vllm_omni.model_executor.models.common.cosyvoice2_batched_token2wav import (
    CosyVoice2BatchedState,
    CosyVoice2BatchedToken2Wav,
)


def test_shared_backend_runs_true_batch_and_splits_cfg_caches():
    token2wav = _FakeToken2Wav()
    backend = CosyVoice2BatchedToken2Wav(token2wav)
    prompt = backend.prepare_prompt("shared", "/fake/prompt.wav")
    states = backend.setup_batch(prompt, 2)

    audios, states = backend.decode_batch(
        torch.tensor([[10, 11], [20, 21]]),
        prompt,
        states,
        last_chunk=False,
    )

    assert token2wav.flow.encoder.calls == [2, 2]
    assert token2wav.flow.decoder.estimator.cfg_batches == [4, 4, 4, 4]
    assert token2wav.hift.calls == [2]
    assert len(audios) == len(states) == 2
    assert isinstance(states[0], CosyVoice2BatchedState)
    assert states[0].flow_cache["estimator_cnn_cache"].data_ptr() != (
        states[1].flow_cache["estimator_cnn_cache"].data_ptr()
    )
```

- [ ] **Step 2: Verify the new shared test fails**

Run:

```bash
python -m pytest \
  tests/model_executor/models/common/test_cosyvoice2_batched_token2wav.py \
  -q
```

Expected: FAIL with `ModuleNotFoundError` for
`common.cosyvoice2_batched_token2wav`.

- [ ] **Step 3: Move and rename the generic backend**

Copy the current implementation, then rename only model-neutral symbols:

```bash
cp \
  vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py \
  vllm_omni/model_executor/models/common/cosyvoice2_batched_token2wav.py
```

The shared module must define:

```python
@dataclass(frozen=True)
class CosyVoice2PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class CosyVoice2BatchedState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


class CosyVoice2BatchedToken2Wav(nn.Module):
    ...
```

Replace MiniCPM-specific exception prefixes in the shared file with
`CosyVoice2BatchError`. Keep optional TRT and CUDA Graph adapters injectable;
do not import LLaMA-Omni2 from the shared module.

- [ ] **Step 4: Preserve the MiniCPM-o import surface**

Replace `minicpmo_4_5/batched_token2wav.py` with explicit aliases:

```python
from vllm_omni.model_executor.models.common.cosyvoice2_batched_token2wav import (
    CosyVoice2BatchedState as BatchedToken2WavState,
    CosyVoice2BatchedToken2Wav as BatchedToken2Wav,
    CosyVoice2PromptFeatures as PromptFeatures,
    cosyvoice2_state_shape_signature as state_shape_signature,
    tensor_signature,
)

__all__ = [
    "BatchedToken2Wav",
    "BatchedToken2WavState",
    "PromptFeatures",
    "state_shape_signature",
    "tensor_signature",
]
```

If CUDA Graph wrappers require model-local imports, pass wrapper instances or
factory callables from `MiniCPMO45Code2Wav` rather than importing the
MiniCPM-o model package from `common`.

- [ ] **Step 5: Run shared and MiniCPM-o regression tests**

Run:

```bash
python -m pytest \
  tests/model_executor/models/common/test_cosyvoice2_batched_token2wav.py \
  tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py \
  -q
```

Expected: all existing MiniCPM-o tests and the new shared test pass.

- [ ] **Step 6: Commit the extraction**

Run:

```bash
git add \
  vllm_omni/model_executor/models/common/cosyvoice2_batched_token2wav.py \
  vllm_omni/model_executor/models/minicpmo_4_5/batched_token2wav.py \
  vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py \
  tests/model_executor/models/common/test_cosyvoice2_batched_token2wav.py \
  tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py
git commit -m "refactor(audio): share CosyVoice2 batching backend" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 3: Load the Real Incremental LLaMA-Omni2 Decoder

**Files:**
- Modify: `vllm_omni/model_executor/models/llama_omni2/llama_omni2_code2wav.py`
- Modify: `tests/model_executor/models/llama_omni2/test_code2wav.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: decoder directory containing one supported YAML file, `flow.pt`,
  and `hift.pt`.
- Produces:
  - `LlamaOmni2DecoderAssets(flow, hift, speaker_embedding, n_timesteps)`
  - `_load_cosy2_modules(model_dir, device, yaml_name=None)`
  - a Flow object exposing incremental encoder/decoder internals required by
    `CosyVoice2BatchedToken2Wav`.

- [ ] **Step 1: Replace the old monkey-patch loader test with a failing YAML-construction test**

Use a fake `hyperpyyaml.load_hyperpyyaml` that returns `{"flow": fake_flow}`:

```python
def test_decoder_loader_constructs_incremental_flow_from_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "cosyvoice.yaml"
    yaml_path.write_text("flow: fake", encoding="utf-8")
    (tmp_path / "flow.pt").write_bytes(b"fixture")
    (tmp_path / "hift.pt").write_bytes(b"fixture")

    fake_flow = _LoadableIncrementalFlow()
    fake_hift = _LoadableHift()
    monkeypatch.setattr(
        "hyperpyyaml.load_hyperpyyaml",
        lambda handle: {"flow": fake_flow},
    )
    monkeypatch.setattr(
        "flashcosyvoice.modules.hifigan.HiFTGenerator",
        lambda: fake_hift,
    )
    monkeypatch.setattr(torch, "load", _fake_decoder_state_load)

    flow, hift = _load_cosy2_modules(tmp_path, torch.device("cpu"))

    assert flow is fake_flow
    assert callable(flow.setup_cache)
    assert callable(flow.inference_chunk)
    assert "inference" not in flow.__dict__
    assert hift is fake_hift
```

- [ ] **Step 2: Verify the loader test fails against the old direct constructor**

Run:

```bash
python -m pytest \
  tests/model_executor/models/llama_omni2/test_code2wav.py \
  -k decoder_loader \
  -q
```

Expected: FAIL because the implementation does not call
`load_hyperpyyaml`.

- [ ] **Step 3: Implement YAML-based decoder loading**

Implement filename resolution:

```python
_DECODER_YAML_CANDIDATES = ("cosyvoice.yaml", "flow.yaml")


def _resolve_decoder_yaml(root: Path, yaml_name: str | None = None) -> Path:
    candidates = (yaml_name,) if yaml_name else _DECODER_YAML_CANDIDATES
    for candidate in candidates:
        if candidate and (root / candidate).is_file():
            return root / candidate
    raise FileNotFoundError(
        f"LLaMA-Omni2 decoder is missing a supported YAML under {root}: "
        f"{', '.join(name for name in candidates if name)}"
    )
```

Load the configured Flow:

```python
from hyperpyyaml import load_hyperpyyaml

with yaml_path.open(encoding="utf-8") as handle:
    config = load_hyperpyyaml(handle)
flow = config["flow"]
if not callable(getattr(flow, "setup_cache", None)):
    raise TypeError("LLaMA-Omni2 decoder Flow does not expose setup_cache")
if not callable(getattr(flow, "inference_chunk", None)):
    raise TypeError("LLaMA-Omni2 decoder Flow does not expose inference_chunk")
```

Load both state dicts strictly, move to the selected device, and evaluate.
Do not install an `inference` `MethodType`.

- [ ] **Step 4: Run loader and decoder-directory tests**

Run:

```bash
python -m pytest \
  tests/model_executor/models/llama_omni2/test_code2wav.py \
  -k 'decoder_loader or decoder_dir or default_english_speaker' \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the loader**

Run:

```bash
git add \
  pyproject.toml \
  vllm_omni/model_executor/models/llama_omni2/llama_omni2_code2wav.py \
  tests/model_executor/models/llama_omni2/test_code2wav.py
git commit -m "fix(models): load incremental LLaMA-Omni2 decoder" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 4: Implement Transactional Exact-Shape LLaMA-Omni2 Batching

**Files:**
- Modify: `vllm_omni/model_executor/models/llama_omni2/llama_omni2_code2wav.py`
- Modify: `tests/model_executor/models/llama_omni2/test_code2wav.py`

**Interfaces:**
- Consumes: flattened stage tokens, `seq_token_counts`,
  `runtime_additional_information`, and request IDs.
- Produces:
  - `_LlamaOmni2RequestState`
  - `_LlamaOmni2WorkItem`
  - exact-shape bucket keys
  - one output slot per scheduled sequence
  - request lifecycle cleanup through `on_requests_finished`.

- [ ] **Step 1: Add a failing true-batching test**

```python
def test_model_batches_equal_shape_requests_in_one_flow_and_hift_call():
    model, token2wav = _batched_model()
    output = _forward(
        model,
        [
            _info("request-a", 0, [10, 11]),
            _info("request-b", 0, [20, 21]),
        ],
        seq_token_counts=[2, 2],
        request_ids=["internal-a", "internal-b"],
    )

    assert token2wav.flow.encoder.calls[-1] == 2
    assert token2wav.flow.decoder.estimator.cfg_batches[-1] == 4
    assert token2wav.hift.calls[-1] == 2
    assert len(output.multimodal_outputs["model_outputs"]) == 2
```

- [ ] **Step 2: Add failing lifecycle and rollback tests**

Add focused tests for:

```text
request reordering preserves state ownership
short non-final lookahead remains pending
mixed final/non-final work forms separate buckets
duplicate chunk_seq raises without state mutation
failed backend decode leaves every prior state unchanged
empty final flush emits exactly once
on_requests_finished removes state
split cache tensors do not alias across requests
```

Use state signatures and cloned tensors to compare the state map before and
after an injected backend failure.

- [ ] **Step 3: Verify the new batching tests fail**

Run:

```bash
python -m pytest \
  tests/model_executor/models/llama_omni2/test_code2wav.py \
  -k 'batch or rollback or reordered or lookahead or final or cancel' \
  -q
```

Expected: failures show the current per-request full-prefix loop and missing
batch state.

- [ ] **Step 4: Replace cumulative state with request-owned incremental state**

Define:

```python
@dataclass(frozen=True)
class _LlamaOmni2RequestState:
    chunk_seq: int
    sequence_index: int
    consumed_units: int
    pending_tokens: torch.Tensor
    token2wav: CosyVoice2BatchedState


@dataclass(frozen=True)
class _LlamaOmni2WorkItem:
    output_index: int
    state_id: str
    request_id: str
    chunk_seq: int
    last_chunk: bool
    tokens: torch.Tensor
    previous: _LlamaOmni2RequestState | None
```

Parse one segment per `seq_token_counts` entry. Validate monotonic chunk
sequence before creating any new state.

- [ ] **Step 5: Implement empty-prompt setup and exact-shape bucketing**

Construct one shared feature bundle:

```python
CosyVoice2PromptFeatures(
    speech_tokens=torch.empty((1, 0), dtype=torch.long, device=device),
    speaker_embedding=load_default_speaker_embedding().to(device),
    mels=torch.empty((1, 0, 80), dtype=torch.float32, device=device),
)
```

Initialize new request states with `backend.setup_batch(features, N)` for all
new requests in the same setup bucket. Bucket decode work by:

```python
(
    int(tokens.numel()),
    last_chunk,
    cosyvoice2_state_shape_signature(state.token2wav),
    tokens.device.type,
    str(tokens.dtype),
)
```

Keep non-final inputs of at most `pre_lookahead_len` in `pending_tokens`.

- [ ] **Step 6: Implement transactional decode and output projection**

For each bucket:

1. stack tokens;
2. call `backend.decode_batch` once;
3. validate output count, finiteness, and state count;
4. stage new request states in a temporary dictionary;
5. commit the dictionary only after every bucket succeeds.

Emit:

```python
{
    "model_outputs": list[torch.Tensor | None],
    "sr": list[torch.Tensor | None],
    "finished": list[torch.Tensor | None],
    "sequence_index": list[torch.Tensor | None],
    "consumed_units": list[torch.Tensor | None],
    "codec_units": list[torch.Tensor],
}
```

Do not call `.cpu()` in the per-request decode path.

- [ ] **Step 7: Run the full LLaMA-Omni2 Code2Wav suite**

Run:

```bash
python -m pytest \
  tests/model_executor/models/llama_omni2/test_code2wav.py \
  -q
```

Expected: all loader, batching, lifecycle, output alignment, and rollback
tests pass.

- [ ] **Step 8: Commit batching**

Run:

```bash
git add \
  vllm_omni/model_executor/models/llama_omni2/llama_omni2_code2wav.py \
  tests/model_executor/models/llama_omni2/test_code2wav.py
git commit -m "perf(models): batch LLaMA-Omni2 Code2Wav" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 5: Align Deployment and Runtime Admission

**Files:**
- Modify: `vllm_omni/deploy/llama_omni2.yaml`
- Modify: `vllm_omni/model_executor/models/llama_omni2/pipeline.py`
- Modify: `vllm_omni/model_executor/stage_input_processors/llama_omni2.py`
- Modify: focused scheduler/connector files only when a named Task 5 test
  fails because the LLaMA-Omni2 contract is absent
- Modify: related LLaMA-Omni2 tests

**Interfaces:**
- Consumes: current async chunk scheduler and stage connector metadata.
- Produces: multiple concurrently admitted Code2Wav sequences with stable
  `request_id`, `chunk_seq`, codec delta, and terminal metadata.

- [ ] **Step 1: Add a failing deployment/config test**

Assert:

```python
def test_llama_omni2_deploy_admits_code2wav_batches():
    deploy = load_deploy_config("llama_omni2.yaml")
    code2wav = deploy.stages[2]
    assert code2wav.max_num_seqs >= 8
    assert code2wav.enforce_eager is True
```

Also assert the Talker → Code2Wav payload contains a monotonic `chunk_seq` and
explicit terminal flag.

- [ ] **Step 2: Verify the test fails with `max_num_seqs: 1`**

Run:

```bash
python -m pytest \
  tests/model_executor/models/llama_omni2/test_pipeline.py \
  tests/model_executor/models/llama_omni2/test_stage_input_processor.py \
  -q
```

Expected: deployment admission assertion fails.

- [ ] **Step 3: Raise Code2Wav admission and preserve eager correctness baseline**

Set stage 2:

```yaml
max_num_seqs: 8
dtype: float32
enforce_eager: true
enable_chunked_prefill: false
```

Do not enable CUDA Graphs in the first E2E comparison. Keep stages 0 and 1 at
their ported values for the first controlled run. If stage-2 profiler evidence
shows no overlapping ready codec chunks, set stage 1 `max_num_seqs: 8` in a
separate measured iteration and report both configurations.

- [ ] **Step 4: Run pipeline, processor, scheduler, and connector tests**

Run:

```bash
python -m pytest \
  tests/model_executor/models/llama_omni2/test_pipeline.py \
  tests/model_executor/models/llama_omni2/test_stage_input_processor.py \
  tests/core/sched/test_omni_scheduling_coordinator.py \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  tests/worker/test_gpu_ar_model_runner.py \
  tests/worker/test_gpu_generation_model_runner.py \
  tests/worker/test_omni_connector_mixin.py \
  tests/worker/test_omni_gpu_model_runner.py \
  -q
```

Expected: focused runtime integration tests pass.

- [ ] **Step 5: Commit deployment alignment**

Run:

```bash
git add \
  vllm_omni/deploy/llama_omni2.yaml \
  vllm_omni/model_executor/models/llama_omni2/pipeline.py \
  vllm_omni/model_executor/stage_input_processors/llama_omni2.py \
  tests/model_executor/models/llama_omni2/test_pipeline.py \
  tests/model_executor/models/llama_omni2/test_stage_input_processor.py
git add -u
git commit -m "feat(runtime): admit batched LLaMA-Omni2 audio decode" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 6: Add a Reproducible LLaMA-Omni2 E2E Benchmark

**Files:**
- Modify: `benchmarks/tts/model_configs.yaml`
- Modify: `benchmarks/tts/bench_tts.py`
- Modify: `benchmarks/tts/README.md`
- Create: `benchmarks/tts/summarize_llama_omni2_runs.py`
- Create: `tests/benchmarks/tts/test_llama_omni2_benchmark.py`

**Interfaces:**
- Consumes: an already-running LLaMA-Omni2 `/v1/chat/completions` endpoint and
  fixed speech-input dataset.
- Produces: per-run JSON plus median, standard deviation, min/max, and
  before/after deltas for TTFP, RTF, and audio throughput.

- [ ] **Step 1: Add failing benchmark argument tests**

```python
def test_llama_omni2_uses_openai_chat_omni_backend():
    config = load_model_configs(_CONFIG)["ICTNLP/LLaMA-Omni2-0.5B"]
    command = build_bench_args(
        host="127.0.0.1",
        port=8000,
        model="ICTNLP/LLaMA-Omni2-0.5B",
        task="speech_to_speech",
        model_cfg=config,
        locale="en",
        num_prompts=8,
        concurrency=4,
        dataset_path="/data/fixed.jsonl",
        wer_eval=False,
        output_dir="/tmp/results",
        result_filename="run.json",
        extra_cli_args=[],
    )
    assert "openai-chat-omni" in command
    assert "/v1/chat/completions" in command
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
python -m pytest tests/benchmarks/tts/test_llama_omni2_benchmark.py -q
```

Expected: FAIL because `speech_to_speech` and the model registry entry do not
exist.

- [ ] **Step 3: Add the model and task mapping**

Add:

```yaml
ICTNLP/LLaMA-Omni2-0.5B:
  supported_tasks: [speech_to_speech]
  backend: openai-chat-omni
  endpoint: /v1/chat/completions
  trust_remote_code: false
  task_extra_body:
    speech_to_speech:
      modalities: [text, audio]
      stream: true
```

Map `speech_to_speech` to the repository's OpenAI chat omni dataset name after
confirming it with `vllm bench serve --help`. When no existing adapter accepts
fixed speech-chat JSONL, add `llama-omni2-s2s` as a dataset adapter that reads
rows with exactly these keys:

```json
{"id":"sample-0001","audio":"/absolute/path/prompt.wav","text":"Respond to the speaker."}
```

- [ ] **Step 4: Implement three-run aggregation**

The summarizer accepts explicit before and after files:

```bash
python benchmarks/tts/summarize_llama_omni2_runs.py \
  --before results/before-c1-run{1,2,3}.json \
  --after results/after-c1-run{1,2,3}.json \
  --label c1
```

For each metric, emit:

```json
{
  "median": 0.0,
  "stdev": 0.0,
  "minimum": 0.0,
  "maximum": 0.0,
  "relative_change_percent": 0.0
}
```

Exit nonzero when c1 regresses beyond 5% or neither c4 nor c8 improves by at
least 10%.

- [ ] **Step 5: Run benchmark unit tests**

Run:

```bash
python -m pytest tests/benchmarks/tts/test_llama_omni2_benchmark.py -q
```

Expected: argument construction and aggregation gate tests pass.

- [ ] **Step 6: Commit benchmark support**

Run:

```bash
git add \
  benchmarks/tts/model_configs.yaml \
  benchmarks/tts/bench_tts.py \
  benchmarks/tts/README.md \
  benchmarks/tts/summarize_llama_omni2_runs.py \
  tests/benchmarks/tts/test_llama_omni2_benchmark.py
git commit -m "bench: add LLaMA-Omni2 E2E performance gate" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 7: Run Local Regression and Static Validation

**Files:**
- Modify only files required to fix failures introduced by Tasks 1-6.

**Interfaces:**
- Consumes: completed rebuilt branch.
- Produces: clean focused test suite and static validation evidence.

- [ ] **Step 1: Run all LLaMA-Omni2 tests**

Run:

```bash
python -m pytest tests/model_executor/models/llama_omni2 -q
```

Expected: all tests pass.

- [ ] **Step 2: Run shared backend and MiniCPM-o regression tests**

Run:

```bash
python -m pytest \
  tests/model_executor/models/common/test_cosyvoice2_batched_token2wav.py \
  tests/model_executor/models/minicpmo_4_5/test_code2wav_batching.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 3: Run touched runtime and benchmark tests**

Run:

```bash
python -m pytest \
  tests/core/sched/test_omni_scheduling_coordinator.py \
  tests/distributed/omni_connectors/test_chunk_transfer_adapter.py \
  tests/engine/test_arg_utils.py \
  tests/engine/test_output_processor.py \
  tests/entrypoints/test_stream_finish_reason.py \
  tests/utils/test_mm_outputs_partition.py \
  tests/worker/test_gpu_ar_model_runner.py \
  tests/worker/test_gpu_generation_model_runner.py \
  tests/worker/test_omni_connector_mixin.py \
  tests/worker/test_omni_gpu_model_runner.py \
  tests/benchmarks/tts/test_llama_omni2_benchmark.py \
  -q
```

Expected: all tests pass or an environment-only collection failure is
recorded with its missing dependency; no product-code failure remains.

- [ ] **Step 4: Run formatting and diff checks**

Run:

```bash
ruff check \
  vllm_omni/model_executor/models/common/cosyvoice2_batched_token2wav.py \
  vllm_omni/model_executor/models/llama_omni2 \
  tests/model_executor/models/llama_omni2 \
  benchmarks/tts
ruff format --check \
  vllm_omni/model_executor/models/common/cosyvoice2_batched_token2wav.py \
  vllm_omni/model_executor/models/llama_omni2 \
  tests/model_executor/models/llama_omni2 \
  benchmarks/tts
git diff --check
```

Expected: all checks pass.

- [ ] **Step 5: Commit only if validation required code changes**

Run:

```bash
git status --short
```

If fixes are present:

```bash
git add -A
git commit -m "test: harden LLaMA-Omni2 batching validation" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 8: Run Real 8×A100 Correctness, E2E, and Profiler Validation

**Files:**
- Create locally: `benchmarks/tts/results/llama_omni2/20260814-a100/`
- Create: `docs/performance/llama_omni2_code2wav_batching.md`
- Do not commit large profiler traces or model artifacts.

**Interfaces:**
- Consumes: the existing remote Docker image, model caches, prompt asset, and
  validation YAML.
- Produces: real decoder parity, online E2E metrics, profiler attribution, and
  a compact checked-in report linking commands to untracked raw artifacts.

- [ ] **Step 1: Recheck remote capacity and GPU occupancy**

Run:

```bash
ssh -S /tmp/ssh-sitian-10.232.195.203 -o ControlMaster=no \
  sitian@10.232.195.203 \
  'df -h /data00; nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv'
```

Expected: reuse existing assets; select GPUs without evicting other users.

- [ ] **Step 2: Transfer only source changes**

Create a compressed tracked-file archive under 100 MB:

```bash
git archive --format=tar.gz --output=/tmp/vllm-omni-5556-rebuild.tar.gz HEAD
```

Stream it through the existing SSH socket to a small remote work directory.
Do not copy `.git`, virtual environments, models, or profiler traces.

- [ ] **Step 3: Run decoder-only correctness parity**

Inside the existing validation image, mount the current source and existing
decoder/model caches. Run:

```text
one deterministic request through the sequential incremental reference
the same request through the batched backend with batch size one
two identical requests through the batched backend with batch size two
```

Record:

```text
sample rate
audio sample count
finite sample count
maximum absolute difference
mean absolute difference
codec units consumed
number of emitted chunks
```

Expected: output lengths match exactly, all samples are finite, and
`torch.testing.assert_close` passes with `rtol=1e-4` and `atol=1e-5`.

- [ ] **Step 4: Run the online controlled matrix**

For the rebuilt sequential reference and batched version, run:

```text
concurrency: 1, 4, 8
measured repetitions: 3
fixed prompts and row order
fixed Thinker/Talker sampling parameters
same GPU placement
same warmup count
```

Save each JSON result with implementation, concurrency, and repetition in the
filename.

- [ ] **Step 5: Capture Code2Wav profiler attribution**

Capture a representative concurrency-4 or concurrency-8 run with PyTorch
profiler or Nsight Systems. Record:

```text
Flow encoder call count and batch sizes
CFM estimator call count and CFG batch sizes
HiFT call count and batch sizes
Code2Wav CPU loop time
device-to-host copy location
GPU utilization
```

Expected: at least one encoder and HiFT call has batch size greater than one,
and CFM sees the corresponding `2*batch` CFG rows.

- [ ] **Step 6: Aggregate and enforce the performance gate**

Run the checked-in summarizer on all three repetitions.

Expected:

```text
c1 median TTFP and RTF regression <= 5%
c4 or c8 audio throughput or median RTF improvement >= 10%
no correctness mismatch
```

If the gate fails, keep the measurements and profile, identify the next
bottleneck, and do not claim an E2E performance win.

- [ ] **Step 7: Write and commit the validation report**

Create `docs/performance/llama_omni2_code2wav_batching.md` containing:

```text
exact commits
hardware and software versions
server and benchmark commands
GPU placement
raw result filenames
median and dispersion table
correctness table
profiler evidence
known limitations
pass/fail decision against every gate
```

Commit:

```bash
git add docs/performance/llama_omni2_code2wav_batching.md
git commit -m "docs: report LLaMA-Omni2 batching results" \
  -m "Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 9: Prepare the Existing PR Branch Update

**Files:**
- No additional product files unless final verification finds a defect.

**Interfaces:**
- Consumes: validated rebuild branch.
- Produces: reviewable replacement history for the existing #5556 branch.

- [ ] **Step 1: Review the complete diff and commit trailers**

Run:

```bash
git diff --stat origin/main...HEAD
git log --format='%H%n%B%n---' origin/main..HEAD
git status --short --branch
```

Expected: clean worktree; every new commit has the required trailer exactly
once.

- [ ] **Step 2: Run final focused verification**

Repeat the Task 7 test commands and inspect the Task 8 report.

Expected: local checks pass and the report truthfully records the remote gate.

- [ ] **Step 3: Update the fork branch only after verification**

Fetch the fork, confirm the existing PR head still points to the expected old
history, then update it with `--force-with-lease`:

```bash
git fetch fork feat/llama-omni2-support
git push --force-with-lease=refs/heads/feat/llama-omni2-support:2806cbac4f27abda6ed710b9414769f0294dc7af \
  fork HEAD:feat/llama-omni2-support
```

Expected: #5556 points to the rebuilt, validated history without overwriting
an unexpected collaborator update.
