# Findings

Everything this project learned that is not on Hailo's official
documentation. Two levels: this page is the summary table; each link goes
to a detailed write-up with evidence and reproduction pointers.

## Compile-contract fixes (all applied by the pipeline)

| # | Finding | One-line summary | Fixed in | Details |
|---|---|---|---|---|
| 1 | Missing `lm_head` | genai argmaxes the raw HEF output — a graph stopping at the hidden state samples garbage over 256 values instead of 32000 tokens | step 1 | [missing-lm-head.md](missing-lm-head.md) |
| 2 | Output shape must be `[1,1,vocab]` | the runtime predicts exactly one token from the last position; full-sequence logits also made lm_head unplaceable | step 1 | [output-shape-last-position.md](output-shape-last-position.md) |
| 3 | RoPE inputs are asymmetrically tiled | runtime writes K=128/Q=256-wide cos/sin buffers, not uniform HD-wide ones; parser's compensation convs double-tile | step 3 | [rope-input-widths.md](rope-input-widths.md) |
| 4 | Attention-mask broadcast semantics | DFC `input_repeats` repeats (AABBCC) where head-broadcast needs tiling (ABCABC); `input_tiles` unsupported in the optimizer → wire the mask directly | step 3 | [attention-mask-broadcast.md](attention-mask-broadcast.md) |
| 5 | `hailo-config.json` key name | server reads `prefill_input_tokens_count`; a `_size` variant is silently ignored (falls back to hardcoded default 96) | step 3 | [kv-cache-config-key.md](kv-cache-config-key.md) |

## Runtime-behavior findings

| # | Finding | Status | Details |
|---|---|---|---|
| 6 | Tokenizer/BOS mismatch between host tokenizers and the embedded one shifts all positions; the genai server also ignores the tokenizer's post-processor (no BOS) and detokenizes one token at a time (a decoder `Strip` eats every space) — both handled in step 3 | fixed in step 3 | [tokenizer-bos-mismatch.md](tokenizer-bos-mismatch.md) |
| 7 | Quantization recipe for KV-cache LLMs (ew_add_fusing disabled; bias_correction enabled on saitama/GPU, no adaround/finetune/group_size; optimization_level=0) | validated | [quantization-recipe.md](quantization-recipe.md) |
| 8 | SDK behaviors: broken quantized emulator on KV-cache graphs, implicit adaround re-enables, Keras registration, paths patch, EINTR interruptions | documented | [sdk-behavior-notes.md](sdk-behavior-notes.md) |
| 9 | KV-cache generation degraded because `__prefill` and `__tbt` laid the shared cache out differently in memory, while HailoRT rotates it as a ring of one-token entries — fixed by explicit `tf_rgb` ↔ `hailo_rgb` format conversions on every cache edge of both groups (as the official LLM compile scripts do); KV-cache greedy generation now matches float32 HF token for token, hailo-ollama serves coherent text | **fixed** | [tbt-cache-read.md](tbt-cache-read.md) |
| 10 | Compiling encoder-only models (BERT/MiniLM-style) — out of main scope, validated side-path | done, scope note | [encoder-model-keras-registration.md](encoder-model-keras-registration.md) |
| 11 | **ROOT CAUSE FOUND** — DFC's `SoftmaxOp.call_hw_sim()` ignores the HN's own `groups=NHEAD` metadata, computing one softmax shared across all attention heads instead of per-head; confirmed bit-exact, present on every checkpoint including the original TinyStories default; open question is whether real silicon shares the bug | **top-priority open issue** | [sdk-native-cosine-drift.md](sdk-native-cosine-drift.md) |
| 12 | Large-vocabulary `lm_head` (e.g. Qwen3's ~152K tokens) fails HEF placement as a single monolithic matmul; root cause pinned to a DFC post-fuser bug at shard fan-out ≥3, fixed by duplicating the shared ancestor normalization/slice chain per shard before quantization — verified end to end (compile, hailo-ollama registration, live `genai` generation) on a real 24-layer Qwen2.5-0.5B checkpoint | **fixed** | [large-vocab-lm-head-sharding.md](large-vocab-lm-head-sharding.md) |
| 13 | Real 24-layer checkpoints (`Qwen2.5-0.5B-Instruct`) failed `__tbt` compilation with a deterministic context-partition topology error via the now-abandoned `defuse()` sharding path — moot once Finding 12 switched to the pre-quantization surgery fix, which compiled successfully | **superseded, kept for the record** | [large-body-multicontext-topology.md](large-body-multicontext-topology.md) |
| 14 | Real hardware prefill numeric drift (cosine 0.858, wrong argmax) on the first real large-vocabulary checkpoint to run end to end — isolated via a low-level test to be neither a tokenizer/BOS desync nor the `__tbt` cache-read bug; tracks Finding 16's `hidden`-size threshold, root cause not yet found | **open** | [large-checkpoint-prefill-drift.md](large-checkpoint-prefill-drift.md) |
| 15 | Sliding-window attention (Mistral-style) needed zero exporter changes — the attention mask is entirely host-computed and content-agnostic to the graph; verified end to end (compile, hailo-ollama, live `genai` response) on a real checkpoint | **validated** | [sliding-window-attention.md](sliding-window-attention.md) |
| 16 | Base-scope (no-cache) generation degenerates to constant token repetition on every checkpoint with `hidden≥576` (8 checkpoints on record, merged from this finding and Finding 11's own hardware data) — lm_head-splitting surgery, non-power-of-2 GQA ratio, AND checkpoint depth (`NLAYERS`) all directly cleared as the cause (a 6-layer, `NREP=1` checkpoint already degrades); `hidden` size is the one variable that separates every coherent checkpoint (`hidden≤288`) from every degraded one | **open, `hidden`-size threshold confirmed, root cause of why not found** | [tinymistral-base-scope-degenerate.md](tinymistral-base-scope-degenerate.md) |
| 17 | Step 6 spent about an hour per network group on deep models in the compiler's single-threaded "Spatial Reshapes Flow", a search that ended with no reshape inserted on every model compiled here — skipped by default with `allocator_param(enable_auto_spatial_reshapes=False)` (`__tbt` placement 1 h 22 → 15 min on SmolLM2-135M, identical placement decisions) | **fixed** | [compile-time-spatial-reshape-search.md](compile-time-spatial-reshape-search.md) |

## How to read these

Each fix page follows the same shape: *symptom* → *investigation* → *root
cause* → *fix* → *verification*. Evidence types used throughout:

- **structural comparison** against official Hailo LLM HEFs
  ([../runtime/diagnostics/hef_audit.py](../../runtime/diagnostics/hef_audit.py));
- **source reading** of the public MIT-licensed
  [HailoRT repository](https://github.com/hailo-ai/hailort) (LLM server C++
  and HEF format) and of DFC Python internals;
- **on-hardware numerics**: cosine similarity vs float32 references via
  the low-level InferModel API
  ([manual_prefill_tbt_test.py](../../runtime/diagnostics/manual_prefill_tbt_test.py)).

No proprietary artifacts are reproduced here — findings are described in
prose with pointers to the public sources they came from.
