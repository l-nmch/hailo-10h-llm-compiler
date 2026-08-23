# Finding 1 — the graph needs an explicit `lm_head`

**Status: fixed (pipeline step 1). Likely root cause of the longest-running
symptom of this project.**

## Symptom

The compiled HEF loaded fine and ran, but generation produced byte-salad —
nonsense token ids unrelated to any prompt.

## Investigation

Reading the public HailoRT LLM server sources showed how a prediction is
produced (`post_process.cpp::get_next_token`): the server takes the HEF's
raw output tensor and applies argmax / top-k **directly on it**. There is no
separate notion of a vocabulary projection anywhere in the runtime.

Our first exported graphs stopped at the final hidden state (a
`[1, seq, 256]` tensor), because the reference model's `lm_head` is a
separate module and we had assumed the runtime would handle the projection.

## Root cause

The runtime has no vocab concept. Whatever tensor the HEF emits, its
argmax over the last axis *is* the predicted token id. A graph without
`lm_head` makes the runtime sample over hidden-state dimensions as if they
were token logits.

## Fix

Append an explicit matmul `lm_head` at the end of the export:

- weights are the model's own `lm_head.weight`, transposed for `x @ W`
  layout;
- critically, this checkpoint does **not** tie embeddings to output weights
  (`tie_word_embeddings=False`, verified by identity check on the loaded
  tensors) — so the embedding table attached as an external resource and
  the lm_head weights baked into the graph are different matrices. Assuming
  tied weights here silently produces a plausible-looking but wrong
  projection.

## Verification

- cosine of exported-graph logits vs float32 HF: 1.000000;
- on hardware, with all other fixes applied, greedy generation through the
  base scope produced real English words — impossible before this fix.
