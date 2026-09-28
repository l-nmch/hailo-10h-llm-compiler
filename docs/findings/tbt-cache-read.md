# Finding 9 — FIXED: KV-cache memory layout must be token-major in both groups

**Status: fixed.** Multi-token generation through the KV-cache path
(`__tbt`, i.e. everything genai and hailo-ollama serve) used to degrade into
real words in incoherent order. The cause was a memory-layout mismatch of
the on-chip KV-cache between the two network groups, not a model, recipe or
runtime defect. With the fix, KV-cache greedy generation on hardware matches
the float32 Hugging Face model token for token, and hailo-ollama serves long,
coherent text from a checkpoint compiled by this pipeline.

## Symptom

- Prefill on hardware was exact (per-position cosine ≈ 1.0 vs float32).
- Base-scope greedy generation (no KV-cache) was coherent.
- Every token produced by `__tbt` — i.e. every token after the first in
  genai/hailo-ollama — was degraded: real words, no grammar, no continuity.

## Investigation

The decisive measurements came from
[kv_greedy.py](../../runtime/diagnostics/kv_greedy.py), which drives
`__prefill` + `__tbt` exactly like the genai server (cache offset advanced
before each run, ring-buffer mask convention), and from single-key
attention probes. TinyStories-25M, 16-token prompt, cache of 24:

| Measurement (cosine vs float32 HF) | Before fix |
|---|---|
| Prefill logits | 0.998 – 0.9996 |
| tbt step, cache fully masked (token attends only to itself) vs HF single-token forward | **0.9975** |
| tbt step reading the cache (first generated token after prefill) | **0.84 – 0.86** |

So the `__tbt` graph itself computes correctly; only what it **reads from
the cache** is wrong. Further steps:

- **Cache quantization parameters** (scale/zero-point/limits of every cache
  input and output) are identical between `__prefill` and `__tbt` — not a
  scale mismatch.
- **Mask convention** confirmed against the public HailoRT source
  (`libhailort/src/core_op/resource_manager/cache_manager.cpp`,
  `hailort_server/genai/llm/llm_server.cpp`): the cache is a ring; each run
  reads `input_length` entries from `read_offset` and writes its outputs at
  `read_offset + input_length`; the server advances the offset *before*
  each run. The first tbt window therefore holds the not-yet-written
  entries first, then the prefill tokens — exactly what
  [runtime_inputs.py](../../runtime/diagnostics/runtime_inputs.py)'s
  `build_mask` assumes. An alternative "valid rows on the left" mask was
  tried and does not reproduce HF either.
- **Column-to-token map.** For each of the 23 cache entries of the first
  tbt window, a tbt step whose mask allows only that entry and itself was
  compared (after removing the self-only contribution) with HF runs whose
  4D mask allows only one prompt position and itself. Instead of a clean
  diagonal, individual cache entries matched **mixtures** of several prompt
  tokens — the signature of a layout mismatch, not of wrong values.
- **The historic "~30% zeroed columns".** The old tap showed 82/272
  (K) and 77/256 (V) columns of the tbt cache read always zero. The number
  of zeros (82 × 23 = 1886, 77 × 23 = 1771) matches the 7 not-yet-written
  entries of the window (7 × 272 = 1904, 7 × 256 = 1792): those were the
  empty ring entries seen through the same layout mismatch, not a
  truncation.
- **Un-expanded cache (not the fix, but closer to the official graphs).**
  Official LLM graphs cache K/V un-expanded (`num_key_value_heads × head_dim`
  wide) and let the attention matmul do the GQA repetition via
  `input_tiles`; this pipeline caches the GQA-expanded K/V. Reproducing the
  official topology (see "Optional: un-expanded cache" below) brought the
  structure in line but did **not** fix generation on its own.

## Root cause

HailoRT treats each KV-cache as a ring of one-token entries
(`entry_size = hw_shape.features`), shared by `cache_id` between
`__prefill` and `__tbt`, and rotates it one entry per generated token
(`cache_manager.cpp`). That only works if the cache sits in memory
**token-major** (one token's features contiguous per entry).

The compiler does not guarantee that on its own. On this pipeline's graphs
it kept the core's native layout for the prefill cache edges and inserted a
different transform on the tbt read side, so writer and reader disagreed
about the layout and every ring rotation mixed tokens. The official
compile scripts shipped with Hailo's LLM artifacts declare an explicit
`format_conversion` on every cache edge — `tf_rgb_to_hailo_rgb` after each
cache input, `hailo_rgb_to_tf_rgb` before each cache output — in both
groups.

## Fix

[s6_compile_hef.py](../../pipeline/s6_compile_hef.py) —
`cache_layout_conversions()` adds, for every layer with
`io_type == "cache"` in both `__prefill` and `__tbt`:

- `format_conversion(<cache input>, <its consumer>, tf_rgb_to_hailo_rgb)`
- `format_conversion(<cache output producer>, <cache output>, hailo_rgb_to_tf_rgb)`

(32 lines for a 4-layer model: K and V, input and output, two groups.)

Negative result worth keeping: adding the conversions to `__prefill` only
made generation *worse* (constant repetition) — both groups must agree.

## Verification

On hardware (Hailo-10H), TinyStories-25M:

| Build | KV-cache greedy continuation of "Once upon a time there was a little girl who lived in a small house" |
|---|---|
| float32 HF | `. She was always very curious and loved` |
| before fix | `. from sh Jane,gry and to` |
| fix + un-expanded cache | `. She was always very curious and loved` — **identical ids, 8/8** |
| fix, standard export | `. She loved to play with her to` — coherent |

End to end with the repository's default pipeline (no flags; standard
export; 128-token cache, 32-token prefill), 32-token prompt ending in
"…to play with her best friend,":

- `kv_greedy.py`: "a little puppy named Spot. They loved to play together
  and have fun." (float32 HF: "a big bear. One day, the little girl was
  playing in the woods" — same first token, coherent continuation).
- hailo-ollama (with the two server-side tokenizer fixes in
  [tokenizer-bos-mismatch.md](tokenizer-bos-mismatch.md)), greedy:
  "Once upon a time, there was a little dog named Max." → "Max loved to
  play with his toys and run around the yard. He would run and jump in the
  yard, making a big mess. One day, Max saw a big ball in the yard. He
  wanted to get it, but he was too small to reach it. So, he tried…"

Reproduce:

```bash
python runtime/diagnostics/kv_greedy.py --hef workdir/model.hef --wte workdir/wte.npy \
    --prompt-ids <exactly PREFILL ids, BOS included>
python runtime/diagnostics/kv_greedy.py ... --mask-mode self   # cache-free tbt control
```

## Optional: un-expanded cache (closer to the official topology, not integrated)

`pre_quantization_optimization(llm_modifications, policy=enabled)` contains
a pass (`_handle_tiling_conv`, in DFC's `post_fuser/algorithms/llm_modifications.py`)
that removes a GQA-repetition conv and moves the repetition into the
attention matmul's `input_tiles`, which un-expands the cache. It never fired
on this pipeline because it only matches the exact chain
*tiling conv → scalar normalization (zero bias) → matmul*, with a kernel of
concatenated identity blocks. Two subtleties found the hard way:

- matmul `input_tiles` **repeats each group** (DFC's emulator comment:
  6 groups, tile 3, input `[1,2,3,4]` → `[1,2,1,2,1,2,3,4,3,4,3,4]`) —
  i.e. Hugging Face `repeat_kv` semantics — while the pass *detects* a
  block-tiling kernel. Permuting query heads to make the conv a true block
  tiling therefore breaks the model; the working export keeps HF head order
  and emits the block-tiling conv on purpose (so the float32 graph is
  intentionally off before the pass, cosine ≈ 0.990) and lets the pass
  restore the math;
- the attention scale must move onto K right after the tiling conv (and a
  scalar multiply onto V, compensated in `o_proj`), otherwise the chain
  doesn't exist; a `× 1.0` is folded away by the parser.

It gave the exact 8/8 match above, but requires disabling the float32
fidelity gates of steps 1-3, so it stays an experiment for now.
