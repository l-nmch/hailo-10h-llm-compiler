# Finding 16 — OPEN: base-scope generation degenerates on TinyMistral-248M, unlike every prior checkpoint

**Status: open, just discovered, not yet root-caused.** Base-scope
generation (no KV-cache — the pipeline's control test for "is the
compiled model itself sound", used throughout this project to isolate
the `__tbt` cache-read bug from everything else) fails on
`Locutusque/TinyMistral-248M`, the checkpoint used to validate
sliding-window attention support
([sliding-window-attention.md](sliding-window-attention.md)). This is a
new, different failure mode from every prior checkpoint's base-scope
result (TinyStories: coherent English; this one: degenerate repetition).

## Symptom

Greedy generation through the base network scope (`generate_base_scope.py`,
patched locally to reconstruct the checkpoint's 2-way sharded lm_head
output — `VOCAB=32005` needs `LM_HEAD_SHARDS=2`, see
[large-vocab-lm-head-sharding.md](large-vocab-lm-head-sharding.md)),
prompt `[32000, 5713, 3714, 264, 727, 736, 403, 264]` (BOS + 7 real
tokens):

```
step 0: token=28705
step 1: token=13
step 2: token=28705
step 3: token=272
step 4-15: token=28705 (repeats every step)
```

Degenerates to constant repetition almost immediately.

## Control: the same prompt on pure float32 HF, greedy, is coherent

Ran the identical greedy loop directly through
`transformers.AutoModelForCausalLM` (no DFC/HAR/HEF/hardware involved)
on the exact same prompt:

```
man named John. He was a man who was a man of many talents.
```

This rules out "the checkpoint itself is just a low-quality tiny model
that degenerates under greedy decoding" — the reference model produces
real, coherent, plausible English. The hardware/pipeline path
diverges significantly from the true model.

## What's different about this checkpoint vs. every prior base-scope success

- `VOCAB=32005` needs exactly `LM_HEAD_SHARDS=2` — the **only** production
  checkpoint tested so far that lands on `N=2`. Finding 12's
  investigation flagged `N=2` as a special case the DFC fuser's
  `swap_layers_order()` happens not to crash on (unlike `N≥3`), but
  never confirmed `N=2`'s *output* is numerically correct, only that it
  compiles. The two output shards are uneven width (`16002` +
  `16003` = `32005`) — worth checking whether the column split point
  and per-shard weight slicing are exactly consistent with how the
  reconstruction (`np.concatenate` in shard-index order) expects them.
- `NREP=4` (Mistral GQA 32/8) — a different ratio from every checkpoint
  whose base-scope was previously proven coherent (TinyStories:
  `NREP=2`).
- `sliding_window=32` is declared in the checkpoint's HF config, though
  the test geometry (`SEQ=24 < sliding_window`) never actually
  exercises it — per
  [sliding-window-attention.md](sliding-window-attention.md), the
  export graph doesn't structurally encode this, so it shouldn't matter,
  but it hasn't been ruled out as a factor by direct testing.
- Compare with [large-checkpoint-prefill-drift.md](large-checkpoint-prefill-drift.md):
  a different real checkpoint (`Qwen2.5-0.5B-Instruct`, `NREP=7`,
  `LM_HEAD_SHARDS=5`, no sliding window declared) also showed hardware
  numeric drift from the float32 reference (prefill cosine 0.858,
  wrong argmax) — not proven to be the same root cause, but a second
  independent data point suggesting something about checkpoints beyond
  the originally-validated TinyStories/small-checkpoint set has a
  hardware fidelity gap, possibly connected to the newer lm_head
  sharding surgery, possibly not.

## Not yet done

- Isolate whether the `N=2` shard reconstruction itself is correct:
  compare hardware shard0/shard1 logits directly against the
  corresponding column ranges of the HF float32 reference logits
  (cosine per shard, not just the concatenated whole) — would confirm
  or rule out a shard-boundary/ordering bug specific to the `N=2` case.
- Re-run the same base-scope test on a checkpoint sharing this
  architecture's `NREP=4` but a `VOCAB` that doesn't need lm_head
  sharding at all (`LM_HEAD_SHARDS=1`), to separate "sharding
  correctness" from "GQA ratio" as variables.
- Check whether this is connected to
  [large-checkpoint-prefill-drift.md](large-checkpoint-prefill-drift.md)'s
  open question, or a fully independent bug.

## Impact

Does **not** invalidate [sliding-window-attention.md](sliding-window-attention.md)'s
core claim (the exported graph is structurally mask-content-agnostic —
that's a fact about the ONNX/HN graph shape, unaffected by this).
It does mean this specific checkpoint's actual generation quality is
unverified/likely broken beyond the already-known `__tbt` cache-read
bug, and the lm_head `N=2` sharding case needs its own correctness
check independent of `N≥3` cases already validated elsewhere.
