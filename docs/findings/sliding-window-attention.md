# Finding 15 — sliding-window attention architectures (Mistral-style) compile with zero code changes

**Status: validated, compiles/serves end to end.** Priority 4 of the
generalization branch's scope
([feedback_generalize-branch-scope]) — the pipeline already supports
sliding-window/alternating-attention architectures, because this
project's attention-mask design was already mask-content-agnostic.

## Why this needed no exporter changes

This pipeline's attention mask (`input_layer2`) is entirely a **host/
runtime-computed additive tensor**, fed identically to every layer —
the exported graph itself only ever does `Q @ K^T + mask`, softmax,
`@ V`. Whether `mask` encodes plain causal masking, a sliding window, or
any other per-position visibility pattern is purely a question of what
*values* are in that tensor, computed outside the graph. The exported
ONNX/HN graph has zero structural dependency on the masking policy.

This is different from tied embeddings, QK-Norm, `head_dim`, or the
large-vocab lm_head — all of which needed real code changes because
they touch the graph's *shape* or *op structure*. Sliding-window
attention doesn't.

## Verification

`Locutusque/TinyMistral-248M` — a real, small (12 layers, `hidden=1024`,
GQA 32/8, `VOCAB=32005`), non-gated `MistralConfig` checkpoint with
`sliding_window=32` declared in its HF config, chosen specifically
because Mistral applies the window **uniformly to every layer**
(simpler than Gemma-2/3's per-layer alternating local/global pattern,
which additionally bundles unrelated new features — logit softcapping,
a custom attention scalar, a different MLP activation — not attempted
here; see "What's not yet covered" below).

Ran through the pipeline **completely unmodified**:

- Step 1 (export): cosine 1.000000 vs float32 HF (both the PyTorch
  reimplementation and the ONNX/onnxruntime round-trip) — architecture
  is structurally identical to every already-supported LLaMA-shaped
  checkpoint, since `sliding_window` never enters the exported graph at
  all.
- Step 2 (parse): structural parse succeeds; `SDK_NATIVE` cosine gate
  needed `--cosine-min 0` (0.9406 at the default 0.99 bar) — the
  already-documented `SDK_NATIVE` softmax-emulation drift
  ([sdk-native-cosine-drift.md](sdk-native-cosine-drift.md)), not
  specific to this checkpoint.
- Step 3 (surgery + resources): cosine recovers to 1.000000
  post-surgery.
- Step 4 (quantize), step 5 (conv repair): both pass cleanly.
- Step 6 (HEF compile): 310.91 MiB HEF, ~1h9m (dominated by the
  32-head/12-layer body's placement search, not anything
  sliding-window-specific).

Deployed and tested through the full runtime contract: registered with
hailo-ollama, served a live `genai` generation response without error.
Output text was incoherent but with recognizable English word fragments
— the same signature as the already-documented `__tbt` cache-read bug
([open-tbt-cache-read.md](open-tbt-cache-read.md)) present on every
checkpoint in this project, not a new issue introduced by this
architecture.

**Base-scope (no-cache) generation, however, does show a new problem on
this checkpoint** — unlike `__tbt` above, this was expected to be
coherent (it's this project's control test proving the compiled model
itself is sound, no cache involved). It degenerates to repetition
instead, while the same prompt on float32 HF is coherent. Not caused by
this finding's subject (sliding-window support needs no graph changes,
so it can't be the mask itself) — most likely connected to this
checkpoint being the first to land on the lm_head `N=2` shard-count
case. See [tinymistral-base-scope-degenerate.md](tinymistral-base-scope-degenerate.md)
for the full investigation; this doesn't change sliding-window
attention's status above.

## What's not yet covered

- **This run's `CACHE_SIZE` (24, the pipeline's default) never exceeded
  `sliding_window` (32)** — so the window constraint itself was never
  actually exercised; a position beyond the window would never have
  occurred at this geometry. A real test of windowing *correctness*
  (not just compilability) needs `SEQ`/`CACHE_SIZE` chosen larger than
  the checkpoint's `sliding_window`, plus a mask-construction path aware
  of the window (this project's own diagnostic tooling,
  `runtime/diagnostics/runtime_inputs.py`'s `build_mask()`, currently
  only knows plain causal masking — extending it to build a windowed
  mask is a small, separate task from what this finding covers).
- **Whether `genai`'s own closed-source runtime (`pre_process.cpp`)
  builds a sliding-window-aware mask at all is unknown.** If it always
  builds a plain causal mask regardless of the checkpoint's declared
  `sliding_window`, then serving through `hailo-ollama`/`genai` would
  silently behave as if the window were disabled once context exceeds
  it — a correctness gap in the *runtime*, not in this pipeline's
  compile step. Not investigated; Hailo's official model catalog has no
  Mistral/Gemma entries to compare against (see the wiki's
  `hef-charter` chapter for the full official-catalog inventory).
- **Gemma-2/3's per-layer alternating local/global pattern** — a
  genuinely different case from Mistral's uniform window, since
  different layers would need *different* mask content
  simultaneously, and this pipeline's single shared `input_layer2`
  input is currently broadcast identically to every layer. Whether
  that's even expressible within the fixed 6-input `genai` contract
  (one shared mask input, not one per layer or per layer-type) is an
  open design question, not attempted here. Gemma additionally bundles
  logit softcapping and a non-default attention scalar, both unrelated
  to attention masking and untested.

## Next steps (not started)

1. Extend `runtime_inputs.py`'s `build_mask()` with a sliding-window
   variant, and re-run the isolated low-level prefill test (as used for
   [large-checkpoint-prefill-drift.md](large-checkpoint-prefill-drift.md))
   at `SEQ > sliding_window` to confirm the model's own numerics are
   correct once genuinely windowed.
2. Investigate whether `genai`'s server-side mask construction has any
   sliding-window awareness at all — if not, this is a runtime-side gap
   this pipeline cannot fix from the compile side.
3. Gemma-2/3's alternating per-layer masking remains unattempted and
   may need a different mechanism (e.g. exposing two separate mask
   inputs and wiring different layer groups to each) rather than an
   extension of the current single-shared-mask design.
