# LLaMA-Omni 2 Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete native vLLM-Omni support for `ICTNLP/LLaMA-Omni2-0.5B`, including paged-KV Qwen2 Thinker/Talker stages, speech input, streaming 24 kHz speech output, tensor parallelism, and checkpoint-compatible weight loading.

**Architecture:** Implement a three-stage pipeline: a Whisper-plus-vLLM-Qwen2 Thinker, a second vLLM-Qwen2 Talker driven by transferred text embeddings and hidden states, and an `ICTNLP/cosy2_decoder` Code2Wav generation stage. Both autoregressive stages reuse vLLM's native Qwen2 implementation so attention, PagedAttention, sampling, and tensor-parallel sharding stay inside the vLLM runtime; stage processors own only typed cross-stage payload construction and per-request streaming state.

**Tech Stack:** Python 3.10-3.13, PyTorch, vLLM model executor, Hugging Face Transformers configs/processors, Whisper, CosyVoice2 flow/HiFT, OmegaConf YAML deployment, pytest.

## Global Constraints

- Target checkpoint is exactly `ICTNLP/LLaMA-Omni2-0.5B`; decoder checkpoint is exactly `ICTNLP/cosy2_decoder`.
- Preserve the checkpoint architecture `Omni2Speech2SQwen2ForCausalLM` and model type `omni2_speech2s_qwen2`.
- Reuse vLLM's native Qwen2 implementation for both autoregressive backbones; do not call Transformers `.generate()` and do not introduce `DynamicCache`.
- Stage 0 and Stage 1 must use vLLM Attention/PagedAttention, KV-cache management, logits processing, sampling, and tensor-parallel linear layers through native Qwen2 composition.
- Implement upstream stream scheduling `(N, M) = (3, 10)` without `eval`: every three new Thinker tokens may schedule at most ten Talker tokens; text completion appends `<sep>` and drains with a finite safety cap.
- Stage 2 output is 24 kHz and supports synchronous output plus `async_chunk` multi-chunk streaming, final flush, cancellation cleanup, and concurrent request isolation.
- Preserve exact typed handoffs: Thinker to Talker uses `ids.output`, `embed.decode`, `hidden_states.output`, and `meta.finished`; Talker to Code2Wav uses `codes.audio` and `meta.finished`.
- Stage 2 must be loadable from a per-stage `model: ICTNLP/cosy2_decoder` deploy override while Stages 0 and 1 use the root model.
- Validate tensor-parallel parity at TP=2 and verify no full unsharded Qwen2 linear weights remain on each rank.
- The target model/checkpoint is for academic, non-commercial use; state this in the recipe and supported-model documentation.
- Run Python commands only through project `uv` or an activated project virtual environment; do not use bare `python`/`python3` or `pip`.
- Use TDD for production behavior: add one focused failing test, observe the intended failure, add the minimum implementation, and rerun to green before proceeding.

---

## File Map

- `vllm_omni/transformers_utils/configs/llama_omni2.py`: local, trust-remote-code-independent HF config classes and nested Thinker/Talker config normalization.
- `vllm_omni/model_executor/models/llama_omni2/config.py`: validated runtime constants and helpers, including safe stream-parameter parsing and checkpoint prefix maps.
- `vllm_omni/model_executor/models/llama_omni2/llama_omni2.py`: stage-selecting unified model wrapper.
- `vllm_omni/model_executor/models/llama_omni2/llama_omni2_thinker.py`: speech encoder/projector integration around native vLLM Qwen2.
- `vllm_omni/model_executor/models/llama_omni2/llama_omni2_talker.py`: projected/gated hidden-state inputs around native vLLM Qwen2 and codec-token logits.
- `vllm_omni/model_executor/models/llama_omni2/llama_omni2_code2wav.py`: CosyVoice2 flow/HiFT loading and request-scoped 24 kHz streaming synthesis.
- `vllm_omni/model_executor/models/llama_omni2/pipeline.py`: immutable three-stage topology and sampling constraints.
- `vllm_omni/model_executor/stage_input_processors/llama_omni2.py`: typed handoffs, stream scheduler, chunk deduplication, drain/flush state, and cleanup.
- `vllm_omni/deploy/llama_omni2.yaml`: default two-GPU deployment with a Stage 2 model override.
- `recipes/ICTNLP/LLaMA-Omni2.md`: setup, serving, API, validation, license, and limitations.

### Task 1: Per-Stage Model Override Infrastructure

**Files:**
- Modify: `vllm_omni/config/stage_config.py`
- Modify: `vllm_omni/engine/arg_utils.py`
- Test: `tests/test_config_factory.py`

**Interfaces:**
- Produces: `StageDeployConfig.model: str | None`.
- Produces: merged stage engine args where a non-null stage `model` overrides the root CLI model only for that stage.
- Preserves: top-level `model` remains orchestrator-owned and does not leak globally through runtime overrides.

- [ ] **Step 1: Write the failing deploy parsing test**

```python
def test_stage_deploy_config_accepts_per_stage_model_override(tmp_path):
    path = tmp_path / "llama_omni2.yaml"
    path.write_text(
        "stages:\n"
        "  - stage_id: 0\n"
        "  - stage_id: 2\n"
        "    model: ICTNLP/cosy2_decoder\n"
    )
    deploy = load_deploy_config(path)
    assert deploy.stages[0].model is None
    assert deploy.stages[1].model == "ICTNLP/cosy2_decoder"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/test_config_factory.py::test_stage_deploy_config_accepts_per_stage_model_override -q`

Expected: FAIL because `StageDeployConfig` has no `model` attribute or parsing puts `model` in `engine_extras`.

- [ ] **Step 3: Add the explicit stage model field**

Add `model: str | None = None` to `StageDeployConfig`, include it in the deploy-schema positive contract, and ensure `_parse_stage_deploy()` reads it as a typed stage field rather than an arbitrary extra.

- [ ] **Step 4: Add a merge test proving stage-local precedence**

```python
def test_merge_pipeline_deploy_uses_stage_model_override():
    pipeline = PipelineConfig(
        model_type="test",
        stages=(
            StagePipelineConfig(stage_id=0, model_stage="thinker"),
            StagePipelineConfig(stage_id=2, model_stage="code2wav", input_sources=(0,)),
        ),
    )
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(stage_id=0),
            StageDeployConfig(stage_id=2, model="ICTNLP/cosy2_decoder"),
        ]
    )
    stages = merge_pipeline_deploy(
        pipeline,
        deploy,
        {"model": "ICTNLP/LLaMA-Omni2-0.5B"},
    )
    assert stages[0].yaml_engine_args["model"] == "ICTNLP/LLaMA-Omni2-0.5B"
    assert stages[1].yaml_engine_args["model"] == "ICTNLP/cosy2_decoder"
```

- [ ] **Step 5: Run focused config tests to GREEN**

Run: `uv run pytest tests/test_config_factory.py -q`

Expected: all config-factory tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_omni/config/stage_config.py vllm_omni/engine/arg_utils.py tests/test_config_factory.py
git commit -m "feat: support per-stage model overrides

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 2: Local Transformers Configuration

**Files:**
- Create: `vllm_omni/transformers_utils/configs/llama_omni2.py`
- Modify: `vllm_omni/transformers_utils/configs/__init__.py`
- Create: `tests/model_executor/models/llama_omni2/test_config.py`

**Interfaces:**
- Produces: `LlamaOmni2Config(Qwen2Config)` with `model_type = "omni2_speech2s_qwen2"`.
- Produces: `thinker_config`, `talker_config`, validated `(stream_text_tokens, stream_unit_tokens)`, unit vocabulary, speech encoder/projector settings, and special token IDs.
- Produces: `AutoConfig.for_model("omni2_speech2s_qwen2", ...)` support without importing upstream remote code.

- [ ] **Step 1: Add failing config round-trip tests**

Construct a minimal dictionary matching the real checkpoint fields and assert:

```python
config = LlamaOmni2Config(**raw_config)
assert config.model_type == "omni2_speech2s_qwen2"
assert config.architectures == ["Omni2Speech2SQwen2ForCausalLM"]
assert config.thinker_config.hidden_size == 896
assert config.talker_config.hidden_size == 896
assert config.stream_text_tokens == 3
assert config.stream_unit_tokens == 10
```

Also assert malformed strings, zero values, negative values, and arbitrary expressions in `stream_params` raise `ValueError`.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_config.py -q`

Expected: import failure because `LlamaOmni2Config` does not exist.

- [ ] **Step 3: Implement safe config normalization**

Use `ast.literal_eval` only for backward-compatible tuple strings, reject non-two-integer tuples, normalize the root Qwen2 fields into `thinker_config`, and normalize `speech_generator` into `talker_config`. Register with `AutoConfig.register`.

- [ ] **Step 4: Register lazy exports and eager side effects**

Add all LLaMA-Omni2 config classes to `_CLASS_TO_MODULE`, `__all__`, and the bottom eager-import block in `configs/__init__.py`.

- [ ] **Step 5: Run config tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_config.py -q`

Expected: all config tests pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_omni/transformers_utils/configs tests/model_executor/models/llama_omni2/test_config.py
git commit -m "feat: add LLaMA-Omni 2 configuration

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 3: Pipeline, Registry, and Default Deployment

**Files:**
- Create: `vllm_omni/model_executor/models/llama_omni2/__init__.py`
- Create: `vllm_omni/model_executor/models/llama_omni2/pipeline.py`
- Modify: `vllm_omni/config/pipeline_registry.py`
- Modify: `vllm_omni/model_executor/models/registry.py`
- Create: `vllm_omni/deploy/llama_omni2.yaml`
- Create: `tests/model_executor/models/llama_omni2/test_pipeline.py`
- Modify: `tests/test_config_factory.py`

**Interfaces:**
- Produces: `LLAMA_OMNI2_PIPELINE` under registry key `omni2_speech2s_qwen2`.
- Produces registry aliases for `Omni2Speech2SQwen2ForCausalLM`, `LlamaOmni2ThinkerForConditionalGeneration`, `LlamaOmni2TalkerForConditionalGeneration`, and `LlamaOmni2Code2Wav`.
- Produces default deploy topology with `async_chunk: true` and Stage 2 `model: ICTNLP/cosy2_decoder`.

- [ ] **Step 1: Write failing topology and registry tests**

Assert three stages with execution types `LLM_AR`, `LLM_AR`, `LLM_GENERATION`; Stage 0 owns the tokenizer and accepts multimodal data; Stage 0 emits text and latent data; Stage 2 emits audio; processor dotted paths import; and all four architectures resolve through `OmniModelRegistry`.

- [ ] **Step 2: Write failing default-deploy tests**

Load `llama_omni2.yaml` and assert `async_chunk is True`, stage devices are `0`, `1`, `1`, Stage 2's model is `ICTNLP/cosy2_decoder`, and merged Stage 0/1 models remain the root checkpoint.

- [ ] **Step 3: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_pipeline.py tests/test_config_factory.py -q`

Expected: missing pipeline/module/registry/deploy failures.

- [ ] **Step 4: Add immutable pipeline and registry entries**

Use exact handoff function names:

```python
_PROC = "vllm_omni.model_executor.stage_input_processors.llama_omni2"
```

Stage 0 uses `thinker2talker_full_payload` and `thinker2talker_async_chunk`; Stage 1 uses `thinker2talker_token_only`, `talker2code2wav_full_payload`, and `talker2code2wav_async_chunk`; Stage 2 consumes generation input.

- [ ] **Step 5: Add the production deploy YAML**

Keep prefix caching disabled, enable shared-memory connectors, use greedy Thinker defaults, Talker sampling compatible with upstream (`temperature=1.0`, `top_p=1.0`), and finite Talker/Code2Wav token caps.

- [ ] **Step 6: Run registry/config tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_pipeline.py tests/test_config_factory.py -q`

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add vllm_omni/model_executor/models/llama_omni2 vllm_omni/config/pipeline_registry.py vllm_omni/model_executor/models/registry.py vllm_omni/deploy/llama_omni2.yaml tests
git commit -m "feat: register the LLaMA-Omni 2 pipeline

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 4: Thinker Speech Feature Preparation

**Files:**
- Create: `vllm_omni/model_executor/models/llama_omni2/config.py`
- Create: `vllm_omni/model_executor/models/llama_omni2/llama_omni2_thinker.py`
- Create: `tests/model_executor/models/llama_omni2/test_thinker.py`

**Interfaces:**
- Produces: `LlamaOmni2ThinkerForConditionalGeneration`.
- Produces: speech placeholder replacement that returns vLLM-compatible flattened `inputs_embeds` plus multimodal output tensors.
- Produces: native Qwen2 `forward`, `compute_logits`, `sample`, and PagedAttention behavior by composing vLLM's registered Qwen2 causal LM.

- [ ] **Step 1: Add failing projector and splice tests**

Test the real downsampling rule: Whisper encoder lengths first become `(length + 1) // 2`, then projector grouping truncates to a multiple of five and divides by five. Verify text embeddings before/after each speech placeholder retain order, multiple audio items map one-to-one, and malformed counts raise a descriptive `ValueError`.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_thinker.py -q`

Expected: missing Thinker implementation.

- [ ] **Step 3: Implement the speech projector**

Implement checkpoint-compatible `linear1 -> ReLU -> linear2` with input width `speech_encoder_hidden_size * speech_encoder_ds_rate`, preserving checkpoint names `speech_projector.linear1` and `speech_projector.linear2`.

- [ ] **Step 4: Implement Whisper feature extraction and prompt splice**

Use the checkpoint's Whisper config and vLLM multimodal input conventions. Do not pad speech features into fake token IDs; replace the negative speech placeholder with projected feature rows before native Qwen2 execution.

- [ ] **Step 5: Compose native vLLM Qwen2**

Construct the registered Qwen2 causal LM using `thinker_config`, delegate logits/sampling methods, and expose required pooling hidden layers for the Talker handoff.

- [ ] **Step 6: Add a native-attention structure test**

Assert the Thinker Qwen2 layers contain vLLM attention modules and tensor-parallel linear classes, and that no Transformers `Qwen2ForCausalLM` or `DynamicCache` is instantiated.

- [ ] **Step 7: Run Thinker tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_thinker.py -q`

Expected: all Thinker CPU/meta tests pass.

- [ ] **Step 8: Commit**

```bash
git add vllm_omni/model_executor/models/llama_omni2 tests/model_executor/models/llama_omni2/test_thinker.py
git commit -m "feat: implement the LLaMA-Omni 2 thinker

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 5: Talker Projection, Fusion, and Native Sampling

**Files:**
- Create: `vllm_omni/model_executor/models/llama_omni2/llama_omni2_talker.py`
- Create: `tests/model_executor/models/llama_omni2/test_talker.py`

**Interfaces:**
- Produces: `LlamaOmni2TalkerForConditionalGeneration`.
- Consumes: aligned Thinker token embeddings, last hidden states, generated token IDs, and terminal state.
- Produces: codec-unit logits shaped `[num_tokens, unit_vocab_size]` through vLLM's normal Qwen2 logits processor/sampler.

- [ ] **Step 1: Add failing projection/fusion tests**

Verify the checkpoint-compatible MLP names and dimensions:

```text
input_proj.0: 896 -> 1792
input_proj.2: 1792 -> 896
gate.0: 1792 -> 896
```

Verify `fusion(rep, emb) = rep * sigmoid(gate(cat(rep, emb))) + emb * (1 - gate)`.

- [ ] **Step 2: Add failing logits and stop tests**

Assert one input row produces one Talker vocabulary row, EOS stops normal scheduling, `<sep>` is appended exactly once on terminal drain, and malformed hidden/token row alignment raises before model execution.

- [ ] **Step 3: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_talker.py -q`

Expected: missing Talker implementation.

- [ ] **Step 4: Implement native Qwen2 Talker composition**

Instantiate vLLM Qwen2 with the normalized Talker config, preserve vLLM logits/sampling entry points, and pass projected/fused embeddings through the native Qwen2 forward path.

- [ ] **Step 5: Add native-attention and TP structure assertions**

Assert native vLLM attention and `ColumnParallelLinear`/`RowParallelLinear`-backed layers exist, with no Transformers generation/cache objects.

- [ ] **Step 6: Run Talker tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_talker.py -q`

Expected: all Talker tests pass.

- [ ] **Step 7: Commit**

```bash
git add vllm_omni/model_executor/models/llama_omni2/llama_omni2_talker.py tests/model_executor/models/llama_omni2/test_talker.py
git commit -m "feat: implement the LLaMA-Omni 2 talker

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 6: Cross-Stage Streaming Scheduler and Typed Payloads

**Files:**
- Create: `vllm_omni/model_executor/stage_input_processors/llama_omni2.py`
- Create: `tests/model_executor/models/llama_omni2/test_stage_input_processor.py`

**Interfaces:**
- Produces: `thinker2talker_full_payload`, `thinker2talker_async_chunk`, `thinker2talker_token_only`, `talker2code2wav_full_payload`, `talker2code2wav_async_chunk`.
- Produces: request-scoped `LlamaOmni2StreamState` keyed by external request ID.
- Enforces: `(3, 10)` scheduling, monotonic offsets, duplicate rejection, terminal drain, cancellation cleanup, and request isolation.

- [ ] **Step 1: Add failing Thinker handoff tests**

Build typed payloads and assert exact keys and row counts. Verify fewer than three new text tokens do not schedule Talker work, the third token does, and six tokens schedule two independent Talker bursts.

- [ ] **Step 2: Add failing terminal and concurrency tests**

Assert `<sep>` is scheduled once, terminal drain remains active until Talker EOS or safety cap, cancelling one request deletes only its state, and interleaved request IDs never share offsets or buffers.

- [ ] **Step 3: Add failing Talker-to-Code2Wav tests**

Assert only newly generated codec IDs are transferred, duplicate/out-of-order chunks are rejected, `meta.finished` propagates exactly once, and empty nonterminal chunks return `None`.

- [ ] **Step 4: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_stage_input_processor.py -q`

Expected: missing processor module/functions.

- [ ] **Step 5: Implement the smallest typed state machine**

Store consumed Thinker rows, emitted codec offset, separator state, drain count, and terminal/flush state per request. Parse stream limits from normalized integer config fields only.

- [ ] **Step 6: Run stage processor tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_stage_input_processor.py -q`

Expected: all scheduler/payload/concurrency tests pass.

- [ ] **Step 7: Commit**

```bash
git add vllm_omni/model_executor/stage_input_processors/llama_omni2.py tests/model_executor/models/llama_omni2/test_stage_input_processor.py
git commit -m "feat: add LLaMA-Omni 2 streaming handoffs

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 7: Unified Stage Wrapper

**Files:**
- Create: `vllm_omni/model_executor/models/llama_omni2/llama_omni2.py`
- Modify: `vllm_omni/model_executor/models/llama_omni2/__init__.py`
- Create: `tests/model_executor/models/llama_omni2/test_model_wrapper.py`

**Interfaces:**
- Produces: `Omni2Speech2SQwen2ForCausalLM`.
- Selects: Thinker, Talker, or Code2Wav from `vllm_config.model_config.model_stage`.
- Delegates: `forward`, logits, sampling, multimodal embedding, and `load_weights` to the selected concrete stage.

- [ ] **Step 1: Add failing stage selection tests**

Patch constructors and assert `model_stage="thinker"`, `"talker"`, and `"code2wav"` each instantiate exactly one expected class; unknown stages raise `ValueError` listing supported names.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_model_wrapper.py -q`

Expected: missing wrapper.

- [ ] **Step 3: Implement delegation without duplicate parameters**

Store one concrete stage as `self.model`; avoid constructing unused stages so memory and weight loading remain stage-local.

- [ ] **Step 4: Run wrapper and registry tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_model_wrapper.py tests/model_executor/models/llama_omni2/test_pipeline.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add vllm_omni/model_executor/models/llama_omni2 tests/model_executor/models/llama_omni2
git commit -m "feat: add the LLaMA-Omni 2 stage wrapper

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 8: Checkpoint Weight Mapping and TP-Safe Loading

**Files:**
- Modify: `vllm_omni/model_executor/models/llama_omni2/llama_omni2_thinker.py`
- Modify: `vllm_omni/model_executor/models/llama_omni2/llama_omni2_talker.py`
- Create: `tests/model_executor/models/llama_omni2/test_weight_loading.py`

**Interfaces:**
- Produces: complete Thinker mapping for `model.*`, `model.speech_encoder.*`, `model.speech_projector.*`, `lm_head.*`, and final norm/embedding aliases.
- Produces: complete Talker mapping for `speech_generator.model.*`, `speech_generator.input_proj.*`, and `speech_generator.gate.*`.
- Preserves: vLLM packed QKV and gate/up loaders and TP shard selection.

- [ ] **Step 1: Add failing synthetic packed-weight tests**

Generate small Qwen2-shaped tensors and assert separate checkpoint `q_proj`, `k_proj`, `v_proj` keys call the packed `qkv_proj` loader with shard IDs `"q"`, `"k"`, `"v"`; assert `gate_proj`/`up_proj` map to packed `gate_up_proj` shards.

- [ ] **Step 2: Add failing real-header coverage test**

Read the committed fixture containing the 1,079 real safetensors key names and assert every supported-stage key maps to exactly one parameter or an explicitly documented skipped tensor. Fail on unmapped trainable keys and duplicate destinations.

- [ ] **Step 3: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_weight_loading.py -q`

Expected: missing/incomplete `load_weights`.

- [ ] **Step 4: Implement prefix stripping and packed mapping**

Reuse vLLM Qwen2 weight-loader conventions. Keep encoder/projector/input-projection/gate names checkpoint-compatible and return the loaded parameter-name set.

- [ ] **Step 5: Add TP rank-shard tests**

Initialize TP metadata for two ranks with reduced tensor sizes and assert each rank loads only its expected column/row shards while replicated norms and biases remain equal.

- [ ] **Step 6: Run weight tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_weight_loading.py -q`

Expected: all mapping, packed-weight, and TP shard tests pass.

- [ ] **Step 7: Commit**

```bash
git add vllm_omni/model_executor/models/llama_omni2 tests/model_executor/models/llama_omni2/test_weight_loading.py
git commit -m "feat: load LLaMA-Omni 2 checkpoint weights

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 9: 24 kHz CosyVoice2 Code2Wav

**Files:**
- Create: `vllm_omni/model_executor/models/llama_omni2/llama_omni2_code2wav.py`
- Create: `tests/model_executor/models/llama_omni2/test_code2wav.py`

**Interfaces:**
- Produces: `LlamaOmni2Code2Wav`.
- Consumes: monotonically appended `codes.audio` plus `meta.finished`.
- Produces: finite float waveform chunks with `sample_rate=24000`, request ID, sequence index, and terminal marker.
- Owns: request-scoped flow pre-lookahead, eight-frame mel cache, HiFT source cache, overlap window, consumed-unit offset, and final-flush flag.

- [ ] **Step 1: Add failing load-contract tests**

Assert missing `cosyvoice.yaml`, `flow.pt`, `hift.pt`, or required config objects produce a descriptive startup error naming the missing artifact.

- [ ] **Step 2: Add failing fake-module streaming tests**

With lightweight fake flow/HiFT modules, send multiple codec chunks and assert:

```text
sample_rate == 24000
chunk_count > 1
consumed offsets increase monotonically
overlap removes duplicated boundary samples
terminal input causes exactly one final flush
post-terminal duplicate input is rejected
```

- [ ] **Step 3: Add failing concurrent-state and cleanup tests**

Interleave two request IDs, assert caches differ, cancel one request, and verify the other request continues unchanged.

- [ ] **Step 4: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_code2wav.py -q`

Expected: missing Code2Wav implementation.

- [ ] **Step 5: Adapt the existing CosyVoice generation model**

Reuse vLLM-Omni's CosyVoice flow/HiFT helpers while preserving LLaMA-Omni2's reference values: 24 kHz, pre-lookahead, eight-frame mel cache, source cache, overlap fade, and terminal flush. Do not inherit the CosyVoice3 22050 Hz output contract.

- [ ] **Step 6: Run Code2Wav tests to GREEN**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_code2wav.py -q`

Expected: all load, chunk, flush, and isolation tests pass.

- [ ] **Step 7: Commit**

```bash
git add vllm_omni/model_executor/models/llama_omni2/llama_omni2_code2wav.py tests/model_executor/models/llama_omni2/test_code2wav.py
git commit -m "feat: synthesize LLaMA-Omni 2 streaming audio

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 10: CPU/Meta Integration and Contract Audit

**Files:**
- Create: `tests/model_executor/models/llama_omni2/test_integration.py`
- Modify: all LLaMA-Omni2 production files as required by failures

**Interfaces:**
- Verifies: stage construction from the real config, native Qwen2 attention/sampling, speech splice, Thinker-to-Talker alignment, Talker-to-Code2Wav unit transfer, and terminal cleanup in one process.

- [ ] **Step 1: Add a tiny-config end-to-end test**

Construct reduced Thinker/Talker configs, fake Whisper and Code2Wav modules, then execute one text prompt and one speech-placeholder prompt through all three stages. Assert text logits rank, codec ID rank, audio rank, typed payload keys, and no leaked request state.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest tests/model_executor/models/llama_omni2/test_integration.py -q`

Expected: first cross-component mismatch fails with a precise assertion.

- [ ] **Step 3: Fix only integration contract mismatches**

Do not add new behavior; align tensor rows, token offsets, payload names, and terminal handling with the already-tested component contracts.

- [ ] **Step 4: Run the complete LLaMA-Omni2 CPU suite**

Run: `uv run pytest tests/model_executor/models/llama_omni2 tests/test_config_factory.py -q`

Expected: all selected tests pass without unexpected warnings.

- [ ] **Step 5: Run static checks**

Run: `uv run pre-commit run --files $(git diff --name-only f775e807)`

Expected: all configured checks pass.

- [ ] **Step 6: Commit**

```bash
git add vllm_omni tests
git commit -m "test: validate LLaMA-Omni 2 stage integration

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 11: Real CUDA E2E and TP=2 Parity

**Files:**
- Create: `tests/e2e/models/llama_omni2/test_llama_omni2.py`
- Create: `tests/e2e/models/llama_omni2/run_llama_omni2_e2e.sh`
- Create: `tests/e2e/models/llama_omni2/README.md`

**Interfaces:**
- Verifies real checkpoint loading, PagedAttention decode, deterministic text parity, speech input, synchronous 24 kHz output, streaming multi-chunk output, concurrent isolation, and TP=2 parity.

- [ ] **Step 1: Add the real-checkpoint E2E harness**

Pin model IDs, prompt audio, seeds, token caps, and output artifact paths. Mark tests with the repository's CUDA/model-download markers so CPU CI does not collect heavy cases accidentally.

- [ ] **Step 2: Run single-GPU text-only parity**

Run the upstream implementation and vLLM-Omni with greedy decoding on the same short text prompt. Compare generated token IDs and record the first divergence if any.

Expected: deterministic text token parity through the configured token cap.

- [ ] **Step 3: Run single-GPU speech input**

Feed a 16 kHz waveform, assert non-empty finite projected features, successful text generation, and native vLLM KV-cache allocation during decode.

- [ ] **Step 4: Run synchronous speech output**

Assert non-empty finite waveform, exact `sample_rate=24000`, terminal flush count one, and no residual request cache.

- [ ] **Step 5: Run async multi-chunk speech output**

Assert more than one audio chunk, monotonic unit consumption, ordered sequence numbers, no duplicate units, one terminal chunk, and concatenated duration within the reference tolerance.

- [ ] **Step 6: Run concurrent request isolation**

Interleave at least two requests with different prompts and assert no cross-request units, hidden states, mel/source caches, or terminal state.

- [ ] **Step 7: Run TP=2 parity**

Run identical seeded requests at TP=1 and TP=2. Assert matching greedy text/codec tokens and waveform numerical similarity within documented precision tolerance; inspect parameter shards to ensure Qwen2 linear weights are not duplicated in full on both ranks.

- [ ] **Step 8: Preserve evidence**

Write exact commands, GPU model, CUDA/driver versions, checkpoint revisions, hashes, token outputs, waveform metadata, and pass/fail assertions to `tests/e2e/models/llama_omni2/README.md`.

- [ ] **Step 9: Commit**

```bash
git add tests/e2e/models/llama_omni2
git commit -m "test: add LLaMA-Omni 2 CUDA coverage

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

### Task 12: Documentation, Supported Models, and Final Verification

**Files:**
- Create: `recipes/ICTNLP/LLaMA-Omni2.md`
- Modify: `docs/models/supported_models.md`
- Modify: `vllm_omni/deploy/llama_omni2.yaml`

**Interfaces:**
- Documents: model/decoder downloads, two-GPU and TP=2 launch commands, chat request shape, 16 kHz input, 24 kHz output, streaming semantics, academic non-commercial restriction, and known limitations.

- [ ] **Step 1: Write the recipe from verified commands**

Include exact working commands from Task 11, not speculative placeholders. Explain Stage 2's independent model override and how to relocate stages/devices.

- [ ] **Step 2: Update the supported-model table**

List text input, speech input, text output, streaming speech output, NVIDIA CUDA, TP=2, root checkpoint, decoder checkpoint, and the non-commercial checkpoint caveat.

- [ ] **Step 3: Run documentation and config checks**

Run: `uv run pre-commit run --files recipes/ICTNLP/LLaMA-Omni2.md docs/models/supported_models.md vllm_omni/deploy/llama_omni2.yaml`

Expected: all checks pass.

- [ ] **Step 4: Run the full focused regression suite**

Run:

```bash
uv run pytest \
  tests/model_executor/models/llama_omni2 \
  tests/test_config_factory.py \
  tests/engine/test_orchestrator_stage_input_bridge.py \
  -q
```

Expected: all focused CPU tests pass.

- [ ] **Step 5: Rerun the required CUDA matrix**

Run the Task 11 script for single-GPU sync, single-GPU async, concurrent async, and TP=2. Do not claim completion if any required case is skipped or unavailable.

- [ ] **Step 6: Audit the diff against the design**

Check every acceptance criterion in `docs/superpowers/specs/2026-07-28-llama-omni2-support-design.md`, scan for `TODO`, `NotImplemented`, Transformers `.generate`, `DynamicCache`, unsafe `eval`, 22050 sample-rate leakage, unbounded per-request dictionaries, and unmapped checkpoint keys.

- [ ] **Step 7: Commit**

```bash
git add recipes/ICTNLP/LLaMA-Omni2.md docs/models/supported_models.md vllm_omni/deploy/llama_omni2.yaml
git commit -m "docs: document LLaMA-Omni 2 support

Co-authored-by: TRAE CLI <noreply@bytedance.com>"
```

## Final Acceptance Checklist

- [ ] Real `ICTNLP/LLaMA-Omni2-0.5B` config and all stage weights load with no unexplained missing/unexpected trainable keys.
- [ ] Thinker and Talker execute through native vLLM Qwen2 Attention/PagedAttention and KV cache.
- [ ] Thinker and Talker logits have the correct vocabulary dimension and use vLLM sampling.
- [ ] Speech placeholders are replaced by Whisper/projector features with correct lengths and ordering.
- [ ] `(3, 10)` streaming scheduling, separator insertion, finite drain, deduplication, cancellation, and concurrent isolation are verified.
- [ ] Code2Wav loads `ICTNLP/cosy2_decoder`, emits finite 24 kHz waveform, streams multiple chunks, and flushes once.
- [ ] TP=2 produces parity with TP=1 and uses sharded Qwen2 weights.
- [ ] Synchronous and async API paths pass with real audio.
- [ ] Focused CPU tests, static checks, and the full required CUDA matrix pass.
- [ ] Recipe and supported-model docs contain verified commands and the academic non-commercial checkpoint restriction.
