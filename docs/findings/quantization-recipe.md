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
