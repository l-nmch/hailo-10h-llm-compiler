# Finding 9 — OPEN: `__tbt` cache reads return truncated tensors

**Status: OPEN — the single remaining blocker.** Everything else in this
repository is validated. This page records the full evidence chain so the
next person starts where we stopped.

## Symptom

Multi-token generation through the KV-cache path (`__tbt` scope, via genai
or hailo-ollama) produces real words in incoherent order — no grammar, no
continuity beyond one or two tokens. Prefill is unaffected.

## What is already proven (control experiments)

| Experiment | Result | What it rules out |
|---|---|---|
| Base-scope greedy generation on hardware, KV-cache bypassed | cosine 0.99 vs float32 AND genuinely coherent text ("...a small house near a park. The little girl loved") | model weights, RoPE, GQA, embeddings, lm_head, INT4 quantization |
| Official KV-cache LLM served through the same genai/hailo-ollama stack | long, coherent generations | DFC in general, the cache mechanism in general, the device, the runtime stack |
| Our HEF's **prefill** on hardware | per-position cosines ≈ 1.000000 vs float32 reference | the compiled `__prefill` instance, embeddings encoding, mask/RoPE construction at prefill lengths |
| K/V tensors **written** into the cache by our `__tbt` instance | cosines ≥ 0.99 vs float32 | the compute side of token-by-token attention |

Conclusion: the defect is isolated to **reading the cache back** inside our
compiled `__tbt` scope. Neither the model, nor the recipe's arithmetic, nor
the hardware, nor the server are implicated by any test so far.

## The sharpest observation

Driving `__tbt` manually ([manual_prefill_tbt_test.py](../../runtime/diagnostics/manual_prefill_tbt_test.py))
and tapping the cache-read outputs:

- the tensors that go **into** the cache are numerically correct;
- the tensors that come **back** are truncated: roughly 30% of the cache
  tensor's columns return as structural zeros — the same column pattern
  every run, reproducible across sessions and restarts.

A deterministic, position-independent zeroing pattern points at an
addressing/layout declaration problem (which slice of the cache each read
consumes), not at noise, quantization, or flakiness.

## Hypotheses eliminated

- **Tokenization/BOS drift** between host and server — was real (see
  [tokenizer-bos-mismatch.md](tokenizer-bos-mismatch.md)) but fixing it
  did not fix tbt generation.
- **KV-cache enable flag / cache offset bookkeeping** driven manually —
  no effect; manual `update_cache_offset` sequences reproduce the same
  truncation.
- **Continuation/state leakage between attempts** — fresh processes show
  the identical pattern.
- **Quantization recipe** — base-scope coherence plus correct cache writes
  under the same recipe make this implausible.

## Current best hypothesis

The duplicated `__tbt` graph instance declares/consumes cache slices with
widths that disagree with how the cache was written (by `__prefill` and by
`__tbt` itself). Candidates worth checking first:

1. compare cache-related layer declarations (`input_shapes` of consumers
   reading cached K/V) between our `.hn` and an official LLM graph at the
   same level of detail as the RoPE/mask fixes;
2. check whether `cache_size == SEQ` assumptions hold inside both
   duplicated scopes (try other combinations);
3. instrument which physical cache slots the zeroed columns correspond to
   (do they track `mask_cache_usage` boundaries?);

## Reproduce it

```bash
# 1. audit structure
python diagnostics/hef_audit.py workdir/model.hef --out-dir audit/

# 2. confirm base-scope coherence (model is fine)
python diagnostics/generate_base_scope.py --hef workdir/model.hef --wte workdir/wte.npy

# 3. localize: prefill exact, tbt degraded
python diagnostics/manual_prefill_tbt_test.py \
    --hef workdir/model.hef --reference refs.npz
```

## If you fix it

Please open a PR touching only what the fix requires, and add the
counter-evidence to this page (what you tried that did NOT work matters as
much as the fix).
