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
| 6 | Tokenizer/BOS mismatch between host tokenizers and the embedded one shifts all positions | understood & worked around | [tokenizer-bos-mismatch.md](tokenizer-bos-mismatch.md) |
| 7 | Quantization recipe for KV-cache LLMs (ew_add_fusing disabled; bias_correction enabled on saitama/GPU, no adaround/finetune/group_size; optimization_level=0) | validated | [quantization-recipe.md](quantization-recipe.md) |
| 8 | SDK behaviors: broken quantized emulator on KV-cache graphs, implicit adaround re-enables, Keras registration, paths patch, EINTR interruptions | documented | [sdk-behavior-notes.md](sdk-behavior-notes.md) |
| 9 | **OPEN** — __tbt cache reads return truncated tensors (~30% structurally zeroed columns) degrading multi-token generation | **open issue** | [open-tbt-cache-read.md](open-tbt-cache-read.md) |
| 10 | Compiling encoder-only models (BERT/MiniLM-style) — out of main scope, validated side-path | done, scope note | [encoder-model-keras-registration.md](encoder-model-keras-registration.md) |
| 11 | **ROOT CAUSE FOUND** — DFC's `SoftmaxOp.call_hw_sim()` ignores the HN's own `groups=NHEAD` metadata, computing one softmax shared across all attention heads instead of per-head; confirmed bit-exact, present on every checkpoint including the original TinyStories default; open question is whether real silicon shares the bug | **top-priority open issue** | [sdk-native-cosine-drift.md](sdk-native-cosine-drift.md) |
| 12 | Large-vocabulary `lm_head` (e.g. Qwen3's ~152K tokens) fails HEF placement as a single monolithic matmul; fixed by sharding it into multiple output convs *after* parsing (post-parse HN edit), proven on a TinyStories proof of concept — not yet integrated into the pipeline | **fix proven, integration pending** | [large-vocab-lm-head-sharding.md](large-vocab-lm-head-sharding.md) |

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
