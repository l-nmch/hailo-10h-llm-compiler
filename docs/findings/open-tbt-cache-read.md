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

## Supporting evidence: token collapse, not context loss

An independent experiment on an earlier version of the pipeline (distilgpt2
substrate, before the tinystories/RoPE work) generated 1/2/3/4 tokens
across several prompts and a wide temperature range:

- with only 1 token generated (prefill only), the first token varies by
  prompt — context-dependent, as expected;
- from 2 tokens onward (once `__tbt` is exercised), generation always
  converges to the **same specific token**, regardless of prompt,
  temperature, or seed.

That total a dominance rules out "context is simply lost and the model
falls back to a generic distribution" — a lost-context fallback would
still vary with temperature. It points at a saturation/overflow in the
`__tbt` compute path rather than a loss of information. Also checked and
ruled out: the token wasn't some degenerate "default" embedding — its
embedding norm was unremarkable (within one standard deviation of the
mean, nowhere near the extremes).

A separate introspection of `genai.LLM`'s undocumented methods
(`get_context_usage_size()`, `max_context_capacity()`, `save_context()`/
`load_context()`) found the context-usage counter occasionally jumps by
more than one and freezes after several correct steps. This is most
likely cosmetic: token collapse toward the fixed value starts at the very
first generated token, well before any counter anomaly appears — treat
the counter jump as a distraction unless a low-level HailoRT trace
connects it to the cache-read defect directly.

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
