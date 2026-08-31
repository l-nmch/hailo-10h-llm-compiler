# Finding 16 — OPEN: base-scope generation degenerates on deeper/larger-hidden checkpoints

**Status: open, reproduced on three independent checkpoints. lm_head
surgery AND GQA ratio both cleared as the cause — depth/hidden scale
is the leading remaining hypothesis, root cause not yet found.** Base-scope
generation (no KV-cache — the pipeline's control test for "is the
compiled model itself sound", used throughout this project to isolate
the `__tbt` cache-read bug from everything else) fails on
`Locutusque/TinyMistral-248M`, the checkpoint used to validate
sliding-window attention support
([sliding-window-attention.md](sliding-window-attention.md)), and on
`HuggingFaceTB/SmolLM2-135M` and `Felladrin/Llama-160M-Chat-v1` (see
"Second confirmation" and "GQA-ratio hypothesis refuted" below). This
is a new, different failure mode from every prior checkpoint's
base-scope result (TinyStories: coherent English; these: degenerate
repetition).

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

## Per-shard investigation: rules out the `N=2` shard-boundary hypothesis

Compared hardware output directly against float32 HF reference logits
for the same 8-token prompt's last position, split at the exact same
column boundary the compiled HEF uses (`shard0`: columns `[0:16002)`,
`shard1`: `[16002:32005)`):

```
conv137_shard0: cosine=0.842972  argmax_local hw=272   hf=676
conv137_shard1: cosine=0.917088  argmax_local hw=12703 hf=12703
FULL cosine(hw concat vs hf): 0.882403
```

**Both shards show real, comparable-magnitude degradation** (0.84 and
0.92) — not the pattern a shard-boundary/reconstruction bug would
produce (which would show one shard essentially correct and the other
catastrophically wrong, or a clean data-swap signature). This rules out
the `N=2`-specific hypothesis: the lm_head split itself is not where
this drift originates. Whatever's degrading these activations happens
*before* the split, upstream in the shared body, and shows up
proportionally on both shards' column ranges.

This makes it look much more like the same phenomenon as
[large-checkpoint-prefill-drift.md](large-checkpoint-prefill-drift.md)
(`Qwen2.5-0.5B-Instruct`, prefill cosine 0.858, also degraded but not
catastrophic) than a bug specific to this checkpoint's sharding case.
Both checkpoints are also the first two to go through the *new*
pre-quantization lm_head-splitting surgery
([large-vocab-lm-head-sharding.md](large-vocab-lm-head-sharding.md)) at
all — TinyStories and every other previously-validated checkpoint never
needed sharding (`LM_HEAD_SHARDS=1`, a no-op path). The surgery itself
(not specifically its `N=2` vs `N=5` shard count) is now the leading
suspect common to both new findings.

## Surgery isolation test: rules out the lm_head-splitting surgery entirely

Forced the surgery's code path on `Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA`
(the project's original, always-coherent baseline) by setting
`LM_HEAD_SHARDS=2` in its `run_config.json` (`VOCAB=32000` never
naturally needs sharding — this exercises `lm_head_split()`'s plumbing
for no real reason, purely to test the mechanism in isolation).
Confirmed the surgery actually ran (`s3`'s log: `lm_head split:
ts25mpipe/conv49 (32000 wide) -> 2 shards ..., chain [slice5,
mul_and_add13] duplicated per shard`), compiled a base-scope HEF
(8m44s, 55.70 MiB — the fast/small checkpoint this project started
with), and ran greedy base-scope generation on real hardware:

```
prompt:    Once upon a time there was a little girl who lived in a small house
generated: . She was very excited about her family
```

**Fully coherent, grammatically correct English.** This conclusively
rules out the lm_head-splitting surgery itself as the root cause of the
drift seen on both TinyMistral and Qwen2.5 — the exact same surgery
code, run on a checkpoint immune to whatever is affecting those two,
produces correct results. The drift's real cause remains open; leading
candidates now shift toward checkpoint scale, GQA ratio, or something
else specific to those two checkpoints that TinyStories doesn't share.

## Second confirmation: SmolLM2-135M shows the identical degenerate pattern

Ran the identical base-scope test on `HuggingFaceTB/SmolLM2-135M`
(30 layers, `hidden=576`, GQA 9/3, `NREP=3`, `VOCAB=49152` needing
`LM_HEAD_SHARDS=2`) — a real, standard `LlamaForCausalLM` checkpoint
notable for two things: it's the deepest checkpoint validated on this
branch so far, and it's a *second* independent architecture sharing a
non-power-of-2 GQA ratio with `Qwen2.5-0.5B-Instruct` (`NREP=7`) and
`TinyMistral-248M` (`NREP=4`). Compiled cleanly through step 6 (both
the standard 2-group HEF and this `--include-base-scope` variant,
289.95 MiB, ~4h16m — confirming the historical "30-layer memory wall"
noted in the wiki's Porting-Another-Model.md no longer applies with
this project's current recipe/memory discipline). Base-scope greedy
generation on real hardware:

```
'2162162162162162162162162162162162812161544221641262'
```

Degenerates to constant repetition of token `216`, the exact same
failure signature as TinyMistral — a third checkpoint, third
architecture family (standard LLaMA, not Mistral/Qwen2), showing the
same pattern. **Every degraded checkpoint so far (`Qwen2.5-0.5B`,
`TinyMistral-248M`, `SmolLM2-135M`) has `NREP` of 3, 4, or 7 — every
coherent one (`TinyStories`, and every smaller checkpoint validated
earlier in this branch) has `NREP` of 1 or 2.** This is now the
strongest correlation found; scale alone doesn't fully explain it either
(`TinyMistral` at 12 layers already degrades, `SmolLM2` at 30 layers
degrades the same way — no obvious scale gradient between them, both
just share `NREP ∉ {1, 2}`).

## GQA-ratio hypothesis refuted: `Felladrin/Llama-160M-Chat-v1` (NREP=1, MHA) degrades too

Direct isolation test: `Felladrin/Llama-160M-Chat-v1` — 12 layers
(matching TinyMistral exactly), `hidden=768`, **`NREP=1` (pure MHA, no
GQA at all)**, `VOCAB=32000` (`LM_HEAD_SHARDS=1`, no sharding surgery
involved either — this test isolates depth/hidden alone, with every
other variable this investigation has considered removed). If the GQA
ratio were the real driver, this checkpoint should behave like
TinyStories (coherent). It did not:

```
'263263263263263263278263263263263263263263263263'
```

**Identical degenerate constant-repetition pattern** (token `263` =
`"a"`), same signature as TinyMistral and SmolLM2. **This refutes the
GQA-ratio hypothesis** — `NREP=1` is exactly what TinyStories has as
its "safe" baseline family (`NREP∈{1,2}` was the dividing line
proposed earlier), yet it degrades anyway. The earlier NREP correlation
was confounded: every previously-tested non-power-of-2-`NREP`
checkpoint also happened to be deeper (12-30 layers) than TinyStories
(4 layers) — this test breaks that confound and points cleanly at
**depth/hidden scale**, not GQA ratio, as the real variable. Every
degraded checkpoint so far has `NLAYERS≥12`; the only coherent one has
`NLAYERS=4`.

## Not yet done

- **Checkpoint depth/hidden scale is now the leading hypothesis**,
  cleanly separated from GQA ratio by the Felladrin test above. Next:
  bisect on depth directly — find or construct a real checkpoint at an
  intermediate layer count (e.g. 6-8 layers) to locate where coherence
  breaks down, and/or vary `hidden` independently of `NLAYERS` to check
  whether hidden size (not layer count specifically) is the real driver
  (TinyStories: `hidden=256`; every degraded checkpoint so far:
  `hidden≥768`).
- Cross-reference with the already-documented, separate `SDK_NATIVE`
  cosine degradation with model scale
  ([sdk-native-cosine-drift.md](sdk-native-cosine-drift.md)) — that
  finding is about the *SDK emulator specifically*, confirmed **not**
  present on real hardware for at least one case (TinyStories). Whether
  there's a related but distinct *hardware*-side scale effect (this
  finding) sharing a root cause with the emulator-side one, or is fully
  independent, is unconfirmed.
- The old "Not yet done" items below (GQA-ratio-specific tests) are now
  superseded by the result above; kept struck through for the record
  rather than deleted, since the investigation trail matters as much as
  the conclusion.

~~`NREP` (GQA ratio) is now the leading, most-correlated hypothesis~~ —
refuted, see above.

## Impact

Does **not** invalidate [sliding-window-attention.md](sliding-window-attention.md)'s
core claim (the exported graph is structurally mask-content-agnostic —
that's a fact about the ONNX/HN graph shape, unaffected by this).
It does mean this specific checkpoint's actual generation quality is
unverified/likely broken beyond the already-known `__tbt` cache-read
bug, and the lm_head `N=2` sharding case needs its own correctness
check independent of `N≥3` cases already validated elsewhere.
