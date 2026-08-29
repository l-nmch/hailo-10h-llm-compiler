# Finding 14 — OPEN: real hardware prefill numeric drift on a real 24-layer checkpoint

**Status: open, not investigated further yet.** Isolated to real hardware
(not SDK emulation), on the first real large-vocabulary checkpoint to
compile and serve end to end through this pipeline
(`Qwen/Qwen2.5-0.5B-Instruct`, 24 layers, `VOCAB=151936`, lm_head split
via the pre-quantization HN surgery in `s3_surgery_and_resources.py`
— see [large-vocab-lm-head-sharding.md](large-vocab-lm-head-sharding.md)).

## Context

Once the lm_head sharding fix let this checkpoint compile and register
with hailo-ollama, a `curl .../api/generate` request returned text that
was not coherent — but before assuming this was the already-documented
`__tbt` cache-read bug ([open-tbt-cache-read.md](open-tbt-cache-read.md)),
the user asked whether it could instead be a tokenizer desync
([tokenizer-bos-mismatch.md](tokenizer-bos-mismatch.md) documents this
exact failure class from an earlier checkpoint). Both were ruled out by
an isolated low-level test.

## Investigation

Bypassed hailo-ollama and genai entirely: drove the `__prefill` network
group directly through the low-level `InferModel` API
(`manual_prefill_tbt_test.py`-style), feeding embeddings/mask/RoPE built
by hand from `runtime_inputs.py`, using the exact reference prompt and
embedding rows already captured by step 1's `hf_reference.npz`. This
removes the tokenizer and the KV-cache read path from the picture
entirely — the only thing being measured is prefill correctness on real
silicon.

Result:

```
HW argmax at position 16: 220
HF-reference logits argmax: 264
cosine(HW logits, HF logits): 0.858
```

Every checkpoint previously validated in this project (TinyStories,
`tinyllama-15M`, etc.) showed prefill cosine ≈ 0.998-1.0 with an exact
argmax match on real hardware. `0.858` with a wrong argmax is a
qualitatively different, much worse result — and it's measured on real
hardware, not `SDK_NATIVE`/`SDK_QUANTIZED` emulation, so the
already-documented "don't trust the emulator's cosine" caveat
([sdk-native-cosine-drift.md](sdk-native-cosine-drift.md)) does not apply
here.

## What this rules out

- **Not a tokenizer/BOS desync**: the test above never tokenizes
  anything — it feeds known-correct embedding rows directly, matching
  exactly what `hf_reference.npz` used for the (passing) step-1 cosine
  gate.
- **Not the `__tbt` cache-read bug**: this measurement is on `__prefill`
  only, no cache read involved.
- **Not `hailo-ollama`/genai's prompt handling**: the low-level
  `InferModel` path bypasses both entirely.

## Update: the lm_head-splitting surgery is cleared as a cause

[tinymistral-base-scope-degenerate.md](tinymistral-base-scope-degenerate.md)
found the same class of drift on a second, independent checkpoint
(`Locutusque/TinyMistral-248M`), and ran a direct isolation test: forced
the exact same lm_head-splitting surgery code path
(`LM_HEAD_SHARDS=2`) onto `Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA` —
the project's original, always-coherent baseline, which never naturally
needs sharding. Base-scope generation on real hardware was **fully
coherent** ("Once upon a time there was a little girl who lived in a
small house" → ". She was very excited about her family"). This
conclusively rules out candidate 1 below: the surgery itself is not the
cause. Candidate 2 (scale/GQA ratio) is now the leading explanation.

## What's still open

The drift's source is unconfirmed. Candidates, none yet tested:

1. The new pre-quantization lm_head-splitting surgery
   (`s3_surgery_and_resources.py`'s layer duplication for the
   `normalization_optimizer.py` fan-out workaround — see
   [large-vocab-lm-head-sharding.md](large-vocab-lm-head-sharding.md))
   introduces a numeric discrepancy somewhere in the duplicated
   slice/normalization chain.
2. Quantization fidelity genuinely degrades at this checkpoint's scale
   or its odd (non-power-of-2) GQA ratio (`NREP=7`, unlike every
   previously-tested checkpoint's `NREP` of 1, 2, or 4) — consistent
   with the already-documented general pattern that larger checkpoints
   show worse cosine
   ([sdk-native-cosine-drift.md](sdk-native-cosine-drift.md)'s "Downstream
   symptom" section, though that finding was about a different,
   smaller-vocabulary checkpoint).
3. Something specific to the 24-layer scale itself, independent of both
   of the above — the same scale that also produced the separate
   `__tbt` context-partition topology error when this checkpoint was
   tested with the (now-abandoned) `defuse()` sharding approach
   ([large-body-multicontext-topology.md](large-body-multicontext-topology.md)).

## Next steps (not started)

1. Compare prefill cosine on the *unsharded* HN (same checkpoint,
   `VOCAB` truncated or lm_head disabled) to isolate whether the
   sharding surgery itself is the source.
2. Compare against a shallower real checkpoint with a similarly odd GQA
   ratio, if one can be found, to separate "scale" from "GQA ratio" as
   variables.
3. Re-run the same isolated prefill test on `tabularisai/Qwen3-0.3B-distil`
   (14 layers, `NREP=2`, once its lm_head is sharded via the new
   pre-quantization approach) as a middle data point between TinyStories
   (4 layers, works) and this checkpoint (24 layers, degraded).
