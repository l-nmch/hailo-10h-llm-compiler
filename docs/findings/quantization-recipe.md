# Finding 7 — quantization recipe for KV-cache LLMs

**Status: validated on hardware. This is the recipe implemented by pipeline
step 4.**

## Context

Generic CNN-style DFC quantization advice does not transfer to LLM graphs
with KV-cache duplication. The recipe below was derived by direct comparison
with Hailo's official LLM recipe for a production chat model (read at the
`.alls` recipe level), then validated on hardware.

## The recipe and why each line exists

```text
pre_quantization_optimization(ew_add_fusing, policy=disabled)
set_kv_cache_global_params(PREFILL_SIZE, CACHE_SIZE)
model_optimization_config(globals, multiproc_policy=disabled)
model_optimization_config(calibration, batch_size=1, calibset_size=32,
                          use_saitama=True, device=cuda)
model_optimization_flavor(compression_level=4, optimization_level=0)
post_quantization_optimization(bias_correction, policy=enabled, use_saitama=True, device=cuda)
post_quantization_optimization(adaround, policy=disabled)
post_quantization_optimization(finetune, policy=disabled)
quantization_param([<scope>/input_layer1], precision_mode=a16_w16)
quantization_param([<conv list>], precision_mode=a8_w4)
```

- **`ew_add_fusing, policy=disabled`** — enabled by default; the official
  recipe disables it explicitly. Leaving it on fuses residual adds into
  convs, which misaligns inputs in the duplicated `__prefill`/`__tbt`
  scopes and (measured) corrupts argmax behavior. Disabling it plus the two
  removals below produced the project's first exactly-correct argmax on
  silicon.
- **`bias_correction` enabled, `adaround`/`finetune` disabled** — earlier
  drafts of this recipe followed the official Qwen2-1.5B `.alls` in
  excluding all three. That's no longer this project's recipe: the ablation
  table below shows `bias_correction` alone measurably *improves* cosine
  (0.960, better than native fp32). `adaround` combined with it is a mild
  net negative (not broken, just not worth it). `finetune` (QAT) combined
  with it measured catastrophic — not a bug in finetune itself, a real
  measured incompatibility between the two. Both stay explicitly disabled.
  `use_saitama=True, device=cuda` must be set **on the
  `bias_correction` directive itself**, not just the global calibration
  one — otherwise it silently falls back to CPU (minutes instead of
  seconds). **This is validated for causal LLM decoders specifically, not
  a universal rule** — Fernando_Soria's [public report on the Hailo
  forum](https://community.hailo.ai/t/hailo-10h-dfc-v5-3-0-a16-w16-on-a-transformer-encoder-is-not-a-blanket-allocator-wall-its-a-3-stage-cascade-attention-crash-a16-conv-nan-exponent-needs-super-defuse-what-is-the-intended-16-bit-path/19530/3)
  found the opposite on a bidirectional transformer encoder (accuracy
  stages collapsing retrieval top-1 from 58.3% to 4.2%; see the caution
  note in `encoder-model-keras-registration.md`). Re-verify on a new
  architecture class rather than assuming either direction.
- **No `weight_group_size`** — incompatible with `bias_correction`+saitama
  specifically (`FusedQWGModule` has no `mac` attribute in the SDK's
  `bias_accumulator.py` — a real bug, not just "unnecessary").
- **`optimization_level=0`** — any higher level silently re-enables
  `adaround`/`finetune` too, which we still want to stay off.
- **`use_saitama=True`** — routes optimization through the PyTorch engine;
  required off-CUDA GPUs and also sidesteps TensorFlow-only code paths.
- **Embeddings at `a16_w16`** — input_layer1 is read host-side as uint16
  codes; 8-bit embeddings break that contract.
- **Raw positions into RoPE inputs during calibration** — feed integer
  positions `[0..SEQ)`, not precomputed cos/sin. DFC's `conversion_type`
  mechanism derives cos/sin itself for calibration; that software emulation
  never executes on-chip.
- **Conv selection**: exclude near-empty kernels (sparsity > 0.9) and
  ultra-narrow convs (min input width ≤ head_dim) from INT4.

## This recipe is not Hailo's actual production recipe — and that's measurable

Everything above is generic DFC accuracy tooling (bias_correction,
adaround) that happens to work reasonably well through robustness, not
because it's the LLM-specific method. Hailo's own public engineering blog
post on bringing generative AI to the Hailo-10H states their production
LLM quantization uses **QuaROT** (a Hadamard-transform outlier mitigation
technique on weight matrices) fused with **GPTQ** (second-order-aware
one-shot post-training quantization) as dedicated stages of the Dataflow
Compiler's optimization flow — techniques never applied in this project.
Their own published benchmark on a production 1.5B chat model shows
near-zero accuracy loss end to end at 4-bit, in a different regime than
anything measured here. A `use_prequantized_weights` directive exists in
the SDK, accepting weights already quantized by an external tool
(stored as `value × scale`) before DFC sees them — a plausible mechanism
for where GPTQ actually happens, upstream of DFC entirely, and untested
in this project.

### Isolated ablation: what each accuracy stage actually buys (cosine vs HF, single batched `infer()`)

| Configuration | cosine(HF, ·) |
|---|---|
| HAR fp32 (no quantization) | 0.938 |
| INT4 + `bias_correction` only | **0.960** (better than native fp32) |
| INT4 + `bias_correction` + `adaround` | 0.953 |
| INT4 + `bias_correction` + `finetune` (QAT) together | **-0.72 — catastrophic**, output is pure noise |

`finetune` combined with `bias_correction` is confirmed to break the model
outright — never combine them. `adaround` alone is a slight net negative
versus `bias_correction` alone here, consistent with the recipe above
dropping both in favor of neither. Consistent with Hailo staff confirming
on the public forum that `adaround` and `finetune` are mutually exclusive
within a single optimization pass in general (not LLM-specific) — this
isn't a quirk of our recipe, it's a documented SDK constraint.

### The `adaround` crash: two diagnoses, only the second one was right

An earlier `bias_correction`+`adaround` run crashed with a
`FileNotFoundError` on a cache file at block 48/49. First hypothesis —
premature cache eviction inside the SDK's cache-cleanup logic — was
**tested and refuted**: patching cache-cleanup to never evict produced
the identical crash at the identical block, disproving the theory.

Real cause: `bias_correction_count` (an SDK config field, default value
**64**) is independent from `calibset_size` (set to 32 here) and was never
synced — `adaround` tried to re-read up to 64 cached calibration samples
when only 32 had ever been written. Fix is one config line:
`bias_correction_count=<your calibset_size>` on the `adaround` directive.
No SDK patch needed. **Method lesson**: a plausible mechanism read
straight from SDK source is still a hypothesis until tested — this one
looked completely coherent and was still wrong.

## Layer Noise Analysis — works on a non-KV-cache-duplicated HAR, blocked only when KV-cache duplication is applied

DFC ships a `Layer Noise Analysis` checker (source class `HailoQuantAnalyzer`)
that infers the model both natively and quantized and reports per-layer SNR.
It's read-only diagnostics — confirmed from the algorithm's own source: it
computes statistics into a work directory and does not write back into the
model, so enabling it cannot change the resulting `.Q.HAR`.

Earlier notes in this project described this as simply "never run" —
that undersold the finding. The real CLI entry point is `hailo
analyze-noise <har-path> --data-path <data-path>` (found in the official
DFC user guide; an internal `model_optimization_config(checker_cfg,
policy=enabled, analyze_mode=advanced)` model-script directive also
exists but was never the actual blocker).

**First attempt, on this project's standard `quantized.har` (with
`set_kv_cache_global_params` applied, `__prefill`/`__tbt` scopes
present)**: fails with `TypeError: 'in <string>' requires string as left
operand, not NoneType` traced through `Cache.__init__` ->
`_get_prefill_size` — `analyze-noise` calls
`build_acceleras_model(InferenceContext.SDK_QUANTIZED)` internally, the
same broken code path already documented in "Quantized emulator is
structurally broken on KV-cache graphs" below. `--adapter-name
<scope>__prefill` does not help — the CLI flag does not appear to reach
`lora_adapter_name` correctly.

**Second attempt (correcting the first pass's conclusion): quantized the
same checkpoint a second time with the identical recipe below, minus the
one line `set_kv_cache_global_params(...)`** (no `__prefill`/`__tbt`
duplication at all — a graph this project would never actually ship,
since it has no KV-cache, but useful specifically to isolate whether the
`Cache` bug is the *only* blocker). Re-ran `analyze-noise` against this
HAR with a correctly-shaped 6-input calibration `.npz`
(`{scope}/input_layer1..6`, matching `build_calibration()`'s layout
below) — **it gets past the `Cache` bug entirely** and reaches real
per-layer noise computation (`LATModel.call()`, comparing native vs.
quantized activations layer by layer), before hitting a *different*,
unrelated error: a shape mismatch (`256` vs `272`) between native and
quantized activations at `conv11` — plausibly a GQA/`repeat_kv`-related
discrepancy, not investigated further. This is real, if incomplete,
progress: it proves the `Cache`/`SDK_QUANTIZED` bug (confirmed elsewhere
as blocking *all* post-quantization emulation on KV-cache graphs) is
specifically about `set_kv_cache_global_params`'s scope duplication, not
some more general property of this project's graphs.

**Revised conclusion**: Layer Noise Analysis is achievable on a
quantized HAR *without* KV-cache duplication, but this project's
standard recipe always applies `set_kv_cache_global_params` (the whole
point of this pipeline). Running it meaningfully on the shipped
KV-cache-duplicated graph remains blocked by the `Cache` bug, and even
without that bug, a second, real shape-mismatch issue (`conv11`,
256 vs 272) surfaced once the tool actually ran — not yet resolved.
Useful as a diagnostic on the *non-KV-cache* base architecture (e.g. to
sanity-check a new checkpoint's per-layer quantization sensitivity
before ever touching KV-cache duplication), not on the final shipped
recipe.

## What could NOT be verified locally

Nothing downstream of step 4 can be checked in software: the quantized
emulator crashes structurally on KV-cache graphs
([sdk-behavior-notes.md](sdk-behavior-notes.md)). Hardware testing
([manual_prefill_tbt_test.py](../../runtime/diagnostics/manual_prefill_tbt_test.py))
is the only judge.

## Results

- Prefill numerics: exact vs float32 (per-position cosines ≈ 1.0).
- Base-scope greedy generation: coherent English.
- KV-cache multi-token generation: still degraded — see
  [open-tbt-cache-read.md](open-tbt-cache-read.md). Nothing suggests the
  recipe is at fault; control experiments isolate the issue to cache reads.
