# Changelog

Every change merged into `main` is a release: `VERSION` is bumped, a
section is added here, and the merge automatically creates the tag
`v<VERSION>-dfc<DFC_VERSION>` and a GitHub release whose notes are the
section below (see [CONTRIBUTING.md](CONTRIBUTING.md#releases)).

The `-dfcX.Y.Z` suffix names the Dataflow Compiler version the release was
validated against.

## [0.3.0] — Loadable quantization recipes, faster deep-model compiles (DFC 5.3.0)

### Added
- Step 4 loads its quantization recipe from a `.alls` file:
  `--recipe recipes/<name>.alls`, with `--print-recipe` to see the resolved
  script without quantizing. Recipes may use placeholders that step 4
  resolves against the graph (layer lists by role: residual adds, SwiGLU
  products, down/up/o projections, QKᵀ matmuls, lm_head, final slice), so
  one recipe applies to any checkpoint.
- `recipes/`: the default recipe, Hailo's own Qwen2 LLM recipe unmodified
  (publicly downloadable, kept as the reference), and `hailo-llm.alls`, that
  recipe with role placeholders. On TinyStories-25M a reduced form of it
  matches HF better on the chip than the default (top-1 15/16 vs 13/16);
  the full form currently scores lower (7/16) and is under investigation.
- Step 4: `--calib-batch-size` and `--no-saitama` (keeps calibration and
  bias correction on TensorFlow, avoiding TensorFlow/PyTorch GPU-memory
  contention on large models).

### Changed
- Step 6 skips the compiler's automatic spatial-reshape search
  (`--auto-spatial-reshapes` restores it). On deep checkpoints it cost about
  an hour per network group and inserted nothing: SmolLM2-135M `__tbt`
  placement drops from 1 h 22 to 15 min with identical placement decisions;
  TinyStories-25M HEFs produce bit-identical logits. See Finding 17.
- `bias_correction` stays on by default, but an on-chip A/B on
  TinyStories-25M measured no gain from it; `--no-bias-correction` is the
  fallback when it runs out of GPU memory.

## [0.2.1] — Release process (DFC 5.3.0)

### Changed
- `main` is now protected: changes land through pull requests only.
- Every merge into `main` publishes a release automatically: tag
  `v<VERSION>-dfc<DFC_VERSION>` plus GitHub release notes taken from this
  changelog.
- New `VERSION` and `DFC_VERSION` files; a pull-request check enforces the
  version bump and the changelog entry.
- README gains a compatibility table (release ↔ DFC ↔ HailoRT ↔ validated
  models).

No pipeline change — HEFs built with 0.2.0 and 0.2.1 are identical.

## [0.2.0] — First coherent generation in hailo-ollama (DFC 5.3.0)

A model compiled by this pipeline now generates coherent text through
hailo-ollama on a real Hailo-10H.

### Highlights
- **KV-cache layout fix.** Multi-token generation no longer degrades: the
  `__prefill` and `__tbt` groups now share a token-major KV-cache layout
  ([Finding 9](docs/findings/tbt-cache-read.md), fixed).
- **hailo-ollama tokenizer fixes.** BOS is inserted through the chat
  template, and spaces are no longer stripped from each generated word
  ([Finding 6](docs/findings/tokenizer-bos-mismatch.md)).
- **128-token context by default** (32-token prefill).
- **New diagnostic:** `runtime/diagnostics/kv_greedy.py` drives greedy
  generation through the real KV-cache path.

### Example (TinyStories-25M)
*"Once upon a time, there was a little dog named Max."* → *"Max loved to
play with his toys and run around the yard. He would run and jump in the
yard, making a big mess…"*

### Validated with
DFC 5.3.0 · HailoRT 5.3.0 · Hailo-10H firmware 5.3.0 · TinyStories-25M

### Known limitations
Larger checkpoints (SmolLM2-135M and up) are not coherent yet
([Finding 16](docs/findings/tinymistral-base-scope-degenerate.md); compile
fixes in progress). Notebooks are stale.

## [0.1.0] — Compile any eligible Hugging Face checkpoint (DFC 5.3.0)

### Highlights
- `--model <hf-id>`: every architecture constant is derived from the
  checkpoint's config; validated on LLaMA2, Qwen2/Qwen2.5, Qwen3 (QK-Norm)
  and Mistral (sliding-window attention) checkpoints from 15M to 500M
  parameters.
- Large-vocabulary `lm_head` sharding (e.g. Qwen's ~152K tokens,
  [Finding 12](docs/findings/large-vocab-lm-head-sharding.md)).
- q/k/v projection biases (Qwen2-style), tied embeddings, explicit
  `head_dim`.
- Self-contained HEFs register and serve in hailo-ollama; prefill is
  numerically exact on hardware.

### Known limitations
Multi-token generation through the KV-cache is degraded (fixed in 0.2.0).
Larger checkpoints show a separate fidelity gap (Finding 16).
