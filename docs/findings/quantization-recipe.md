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
quantization_param([<scope>/input_layer1], precision_mode=a16_w16)
quantization_param([<conv list>], precision_mode=a8_w4)
```

- **`ew_add_fusing, policy=disabled`** — enabled by default; the official
  recipe disables it explicitly. Leaving it on fuses residual adds into
  convs, which misaligns inputs in the duplicated `__prefill`/`__tbt`
  scopes and (measured) corrupts argmax behavior. Disabling it plus the two
  removals below produced the project's first exactly-correct argmax on
  silicon.
- **No `bias_correction`, no `adaround`** — the official recipe uses
  neither. Adding them did not improve and did complicate attribution.
- **No `weight_group_size`** — incompatible with the PyTorch optimizer
  (SDK bug); unnecessary without group-wise INT4 anyway.
- **`optimization_level=0`** — any higher level silently re-enables
  adaround/bias_correction/finetune, undoing the choices above.
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
dropping both in favor of neither.

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

## Unexploited: Layer Noise Analysis

DFC ships a `Layer Noise Analysis` checker (source class `HailoQuantAnalyzer`)
that infers the model both natively and quantized and reports per-layer SNR.
It's read-only diagnostics — confirmed from the algorithm's own source: it
computes statistics into a work directory and does not write back into the
model, so enabling it cannot change the resulting `.Q.HAR`. Never run on
this project; see [docs/status.md](../status.md) "What would move the
needle next" for why it's worth doing before the next round on the
cache-read issue.

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
