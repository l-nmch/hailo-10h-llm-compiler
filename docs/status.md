# Project status

An honest snapshot of what works, what does not, and what is unknown. Read
this before investing time, and update it when you learn something new.

## The one-paragraph summary

The full compile pipeline works end to end: a Hugging Face checkpoint goes
through ONNX → HAR → surgery → quantization → HEF, and the resulting
self-contained HEF loads in genai and hailo-ollama. **But a structural bug
in the mask/softmax wiring, present since the project's first commit, was
just found and is now the top-priority open issue**
([findings/sdk-native-cosine-drift.md](findings/sdk-native-cosine-drift.md)):
DFC's compiled softmax normalizes across all attention heads combined
instead of computing each head's softmax independently — confirmed
bit-exact by reproducing it from raw QK^T scores, and confirmed present on
every checkpoint tested, including the original "validated" TinyStories
default (it just never surfaced there because final-logits cosine happens
to still land near 1.0 despite the structurally wrong intermediate
computation). This is very likely the actual root cause behind the
multi-token KV-cache incoherence below, not a separate issue — the two
findings should be read together.

**Generalization in progress.** The pipeline no longer hardcodes one
checkpoint: `pipeline/s1_export_onnx.py --model <hf-id>` derives every
architecture constant from `transformers.AutoConfig` and steps 2-6 pick it
up automatically. Validated end to end (step 1 + step 2's `SDK_NATIVE`
fidelity gate) on 4 additional checkpoints spanning GQA and MHA, 4-22
layers, 64-768 hidden — all pass step 1; step 2's cosine degrades with
model scale on larger checkpoints, a second open issue
([findings/sdk-native-cosine-drift.md](findings/sdk-native-cosine-drift.md)).

## Stage-by-stage

| Stage | State | Notes |
|---|---|---|
| HF → ONNX export | ✅ solid | cosine 1.000000 vs float32; matmul-trick graph; lm_head + last-position slice; the PyTorch source model applies attention masking correctly per-head (real tensor dim, not flattened) |
| ONNX → HAR parse (hailo10h) | ⚠️ final-logits cosine 1.000000, but this metric is now known misleading | see the softmax finding — a perfect final-output cosine coexists with a structurally wrong intermediate softmax on every checkpoint tested so far |
| Graph surgery + resources | ⚠️ final cosine 1.000000, same caveat | `mask_surgery()`'s rewiring of `input_layer2` into each layer's mask-add is the current top suspect for where the per-head boundary gets lost — present since the project's first commit |
| Quantization (KV-cache) | ✅ runs (~30 s GPU) | recipe validated by comparison with official `.alls`; no emulator check possible (see below) |
| Conv repair pass | ✅ kept as safety net | finds 0 issues with the final recipe — its historical cause was ew_add_fusing |
| HAR → HEF compile | ✅ works (~5–8 min) | both network groups emitted; monolithic lm_head places at optimization_level=0 thanks to the last-position slice |
| Notebooks ([../notebooks/](../notebooks/)) | ✅ tested headless + on hardware | `walkthrough.ipynb` executes the full chain green (HEF ≈ 44 MiB); its HEF was probed through the raw `InferModel` API: prefill logits cosine 0.998 with exact argmax, tbt degraded identically to pipeline HEFs |
| genai.LLM load | ✅ works | HEF passes the full runtime contract (six inputs, embedded resources, config keys) |
| hailo-ollama serving | ✅ registration + serving work | content-addressed blob store + manifest procedure documented |
| Prefill numerics | ✅ exact | per-position cosines ≈ 1.0 vs float32 reference on hardware |
| Base-scope greedy generation (no cache) | ✅ coherent | real English text; proves weights/RoPE/GQA/lm_head are sound |
| **tbt generation via KV-cache** | ❌ degraded | words are real but incoherent across steps — see the open finding |

## Known SDK/toolchain behaviors worth knowing

Documented in detail in [findings/sdk-behavior-notes.md](findings/sdk-behavior-notes.md);
short version:

- the quantized emulator (`SDK_QUANTIZED`) is structurally broken on
  KV-cache graphs — hardware is the only judge after step 4;
- `optimization_level > 0` silently re-enables adaround/bias_correction;
- Keras deserialization of optimized HARs needs the acceleras layer
  registration preamble ([../pipeline/config.py](../pipeline/config.py));
- a host-side EINTR/ioctl failure can interrupt long-lived LLM sessions —
  [genai_generate.py](../runtime/genai_generate.py) isolates attempts in
  subprocesses to absorb it.

## What would move the needle next

Ranked guesses, informed by everything eliminated so far (full list inside
the open findings):

1. **Fix the shared-across-heads softmax bug**
   ([findings/sdk-native-cosine-drift.md](findings/sdk-native-cosine-drift.md)).
   Top priority — plausibly the single fix that resolves both the
   cosine-drift finding and the KV-cache incoherence below at once. Next
   concrete step: locate exactly where `mask_surgery()`'s `input_layer2`
   rewiring (or DFC's `SoftmaxLayer` construction) loses the per-head
   reduction boundary, and check whether the compiled/quantized HEF
   (not just the `SDK_NATIVE` float32 simulation) shares the same
   mechanism.
2. Instrument the tbt cache read path more deeply (which columns come back
   zeroed, for which scopes/contexts) — informed by (1), since a broken
   softmax could itself produce a pattern that looks like cache
   truncation without the cache mechanism itself being at fault.
3. Compare against a second official KV-cache model's HEF structure beyond
   the recipe level (context descriptors, cache layout declarations,
   specifically how the official mask/softmax wiring differs from this
   project's `causal_mask_tiled()` scheme).
4. Try `cache_size` / `prefill_size` combinations other than the
   SEQ==CACHE_SIZE assumption.
5. Reproduce on a second Hailo-10H unit to rule out the specific device.
6. Run DFC's **Layer Noise Analysis** checker
   (`model_optimization_config(checker_cfg, layer_noise_analysis, ...)`,
   or the algorithm's own name inside the SDK) during step 4. It's a
   read-only diagnostic — it infers the model both native and quantized
   and reports per-layer SNR, without altering the resulting `.Q.HAR` —
   never run on this project so far (the directive name tried,
   `layer_noise_analysis`, is confirmed wrong — `'layer_noise_analysis'
   is not a valid PostQuantizationFeature`; correct SDK syntax still
   unknown).

## Provenance note

Every claim above was verified on real hardware with reproducible tests;
the diagnostics under
[../runtime/diagnostics/](../runtime/diagnostics/) re-derive the key ones.
