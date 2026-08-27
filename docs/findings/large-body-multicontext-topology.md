# Finding 13 — OPEN: `__tbt` context-partition topology error on large (24-layer) bodies

**Status: open, deprioritized.** Blocks HEF compilation (step 6) for real
Qwen2.5/Qwen3-scale checkpoints (24 layers tested; unclear where the
threshold is). Distinct from, and downstream of, the large-vocabulary
`lm_head` sharding fix ([large-vocab-lm-head-sharding.md](large-vocab-lm-head-sharding.md)),
which is fully proven and integrated — this finding only appears once
that fix lets compilation get far enough to reach it.

## Symptom

Step 6 fails, specifically on the `__tbt` network group only (`__prefill`
of the same checkpoint compiles fine), with:

```
Context-Partition topology error: <scope>__tbt/input_layer2 (bucket_1) is
in later context than its successor <scope>__tbt/ew_add3 (bucket_0).
<scope>__tbt/input_layer2 (bucket_1) is in later context than its
successor <scope>__tbt/ew_add8 (bucket_0). <scope>__tbt/input_layer3
(bucket_1) is in later context than its successor <scope>__tbt/ew_mult2
(bucket_0). [... 10 lines total, all input_layer2-6 vs a small handful
of ew_add/ew_mult ops]
```

`input_layer2`-`input_layer6` are the mask and RoPE cos/sin inputs,
consumed identically by every decoder layer. The conflicting op indices
are consistently ~5 apart (`ew_add3`/`ew_add8`, `ew_mult1`/`ew_mult6`,
`ew_mult4`/`ew_mult9`, `ew_mult3`/`ew_mult8`) — consistent with ~5
RoPE/mask elementwise ops per layer, meaning only the **first 1-2 of 24
layers** are implicated, not the whole network. The compiler's automatic
context-partitioner has placed a handful of early-layer ops into an
earlier bucket than the shared inputs they consume.

## Reproduction

`Qwen/Qwen2.5-0.5B-Instruct` (real, non-distilled, 24 layers,
`hidden=896`, GQA 14/2, `VOCAB=151936`) through step 6 with `lm_head`
sharded via `defuse(layer, N=24)` (needed to get past the separate
large-vocab placement wall first). Confirmed **not** a lm_head-sharding
artifact: identical error reproduces with `N=10` (still hit the SC wall
first, but the sequence establishes the checkpoint's scale is the
relevant variable, not `N`) and persists unchanged across four full
compile attempts (~32 min each) with different model-script tuning.

## What was tried (all four produced the byte-for-byte identical error)

1. **`performance_param(compiler_optimization_level=1)`** (up from the
   pipeline's default `0`) — no effect. Rules out "the level-0 search is
   too crude to find a valid partition."
2. **`context_switch_param(allow_auto_merge_in_multicontext=True)`** — no
   effect.
3. **`context_switch_param(mode=disabled)`** (forces single-context
   compilation) — no effect: **the exact same multi-context-flavored
   error still appears**, meaning either this directive isn't being
   respected for this graph shape, or the `bucket_0`/`bucket_1` naming in
   this error is not literally about HEF execution contexts but some
   other internal resource-partitioning concept unaffected by
   `context_switch_param`. Not yet resolved which.

All three are architecturally-motivated model-script directives (found
in the official DFC user guide) targeting exactly this class of
symptom — none of them are the fix. This strongly suggests the bug is
baked into the automatic partitioner's handling of this specific graph
shape (a body large enough to need multi-context, combined with inputs
broadcast identically into every layer), not something reachable from
the model-script surface at all.

## Root cause

Not yet identified. Leading hypothesis: the `__tbt` scope, once large
enough (checkpoint- and layer-count-dependent; TinyStories at 4 layers
and `tabularisai/Qwen3-0.3B-distil` at 14 layers never hit this) to need
multiple hardware contexts, has a latent bug in how the automatic
partitioner assigns context/bucket membership to the handful of ops in
the very first 1-2 layers that consume the always-shared RoPE/mask
inputs. Untested: whether this is `NREP=7` (Qwen2.5-0.5B's odd,
non-power-of-2 GQA ratio) specific, or purely a layer-count/graph-size
threshold independent of GQA ratio — `tabularisai/Qwen3-0.3B-distil`
(14 layers, `NREP=2`) never reached this error, but it was also never
pushed past the large-vocab wall before this fix existed, so it hasn't
actually been retested since.

## Cost note

Each compile attempt at this checkpoint's scale takes ~30 min on the
project's standard GPU environment (`dfc-gfx906:5.3.0`, AMD gfx906).
Further investigation should prefer cheap reproduction first (a small
synthetic checkpoint at the same layer count/GQA ratio, or bisecting
`NLAYERS` on a real checkpoint) over further full-scale blind attempts.

## Next steps (not started)

1. Bisect on layer count: try the same architecture family at
   intermediate depths (e.g. 16, 20 layers) to find whether there's a
   sharp threshold, and whether it correlates with when `__tbt` first
   needs multiple contexts.
2. Test whether `NREP` (GQA ratio) is a factor by trying an
   even-power-of-2-ratio checkpoint at a similar layer count.
3. Inspect the HN's own context/bucket assignment directly
   (`get_hn_model()`) before compilation to understand what triggers
   the partitioner's decision, rather than treating the compiler as a
   black box.
4. If no fix is found, consider whether this needs to be reported
   upstream (compare against the community PR pattern already used for
   other confirmed serverside bugs — see the master project notes) since
   three separate documented model-script directives for exactly this
   symptom class all failed to change the outcome.
