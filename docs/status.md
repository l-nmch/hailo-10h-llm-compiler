# Project status

An honest snapshot of what works, what does not, and what is unknown. Read
this before investing time, and update it when you learn something new.

## The one-paragraph summary

The full compile pipeline works end to end: a Hugging Face checkpoint goes
through ONNX → HAR → surgery → quantization → HEF, and the resulting
self-contained HEF loads in genai and hailo-ollama. Prefill inference on
hardware is numerically faithful (cosine ≈ 1.0 against float32). Greedy
generation with the KV-cache disabled is coherent. **Multi-token
generation through the KV-cache path produces degraded text** — one open
issue remains ([findings/open-tbt-cache-read.md](findings/open-tbt-cache-read.md)).

**A separate SDK (not hardware) bug was found and closed this session**
([findings/sdk-native-cosine-drift.md](findings/sdk-native-cosine-drift.md)):
DFC's `SDK_NATIVE`/`SDK_BIT_EXACT` emulation normalizes softmax across all
attention heads combined instead of per-head — confirmed bit-exact by
reproducing it from raw QK^T scores, and confirmed present on every
checkpoint tested including the original TinyStories default (masked
there by a final-logits cosine that happens to still land near 1.0). It
looked at first like it might explain the KV-cache incoherence above, but
that connection is ruled out: TinyStories' real-hardware base-scope
generation is coherent despite carrying this exact SDK-emulation defect,
so real silicon does not share it. Net effect: don't trust
`SDK_NATIVE`/`SDK_BIT_EXACT` cosine as an attention-fidelity signal on
multi-head checkpoints; the KV-cache incoherence above remains a fully
separate, still-open question.

**Generalization in progress.** The pipeline no longer hardcodes one
checkpoint: `pipeline/s1_export_onnx.py --model <hf-id>` derives every
architecture constant from `transformers.AutoConfig` and steps 2-6 pick it
up automatically. Validated end to end (step 1 + step 2's `SDK_NATIVE`
fidelity gate) on 4 additional checkpoints spanning GQA and MHA, 4-22
layers, 64-768 hidden — all pass step 1; step 2's cosine degrades with
model scale on larger checkpoints, a second open issue
([findings/sdk-native-cosine-drift.md](findings/sdk-native-cosine-drift.md)).
Tied embeddings, QK-Norm (Qwen3-style), and explicit `head_dim` are now
supported and verified through quantization on real checkpoints
(`nickypro/tinyllama-15M`, `Qwen/Qwen3-0.6B`,
`tabularisai/Qwen3-0.3B-distil`). Full HEF compilation of a real Qwen3
checkpoint is currently blocked by a separate, unrelated wall: any
checkpoint sharing Qwen's ~152K-token vocabulary fails `lm_head`
placement as this pipeline currently exports it as one monolithic matmul
— official Hailo HEFs shard it into multiple output convs instead; fix
identified, not yet implemented
([findings/large-vocab-lm-head-sharding.md](findings/large-vocab-lm-head-sharding.md)).

## Stage-by-stage

| Stage | State | Notes |
|---|---|---|
| HF → ONNX export | ✅ solid | cosine 1.000000 vs float32; matmul-trick graph; lm_head + last-position slice |
| ONNX → HAR parse (hailo10h) | ✅ solid on hardware; ⚠️ `SDK_NATIVE` cosine numbers alone are not trustworthy | see the softmax finding — DFC's SDK emulation (`SDK_NATIVE`/`SDK_BIT_EXACT`) has a confirmed defect that makes attention-layer cosine look worse than it is; confirmed NOT present on real hardware (TinyStories base-scope generation is coherent despite carrying the defect in emulation) |
| Graph surgery + resources | ✅ solid on hardware, same emulation caveat | `mask_surgery()`'s `input_layer2` rewiring is correct — it was the prime suspect until hardware evidence ruled it out |
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
- `SDK_NATIVE`/`SDK_BIT_EXACT` softmax emulation ignores the HN's own
  `groups` metadata and normalizes across all attention heads combined —
  confirmed **not** present on real hardware (see
  [findings/sdk-native-cosine-drift.md](findings/sdk-native-cosine-drift.md)),
  but don't trust either context's cosine as an attention-fidelity signal.

## What would move the needle next

Ranked guesses, informed by everything eliminated so far (full list inside
the open findings):

1. Instrument the tbt cache read path more deeply (which columns come back
   zeroed, for which scopes/contexts).
2. Compare against a second official KV-cache model's HEF structure beyond
   the recipe level (context descriptors, cache layout declarations).
3. Try `cache_size` / `prefill_size` combinations other than the
   SEQ==CACHE_SIZE assumption.
4. Reproduce on a second Hailo-10H unit to rule out the specific device.
5. Continue the scale/quantization-precision investigation for why larger
   checkpoints (Felladrin, `hidden=768`) produce incoherent base-scope
   text on hardware while TinyStories doesn't — INT8 measurably better
   than INT4, `calibset_size` increase made it worse not better; root
   cause still open (see the "Downstream symptom" section of
   [findings/sdk-native-cosine-drift.md](findings/sdk-native-cosine-drift.md)).
6. Run DFC's **Layer Noise Analysis** checker (`hailo analyze-noise <har>
   --data-path <data>`, confirmed from the official user guide). Blocked
   by the `Cache`/`SDK_QUANTIZED` bug on this project's standard
   KV-cache-duplicated `quantized.har` — but confirmed working (reaches
   real per-layer noise computation) on a HAR quantized *without*
   `set_kv_cache_global_params`, before hitting a separate, real
   shape-mismatch error (`conv11`, 256 vs 272 — plausibly GQA/`repeat_kv`
   related, not yet investigated). See
   [findings/quantization-recipe.md](findings/quantization-recipe.md)'s
   "Layer Noise Analysis" section for the full evidence and the exact
   no-KV-cache recipe used. Next step: chase the `conv11` shape mismatch
   specifically — it's the one remaining blocker on the no-KV-cache path,
   and understanding it might also inform the KV-cache `Cache` bug itself.

## Provenance note

Every claim above was verified on real hardware with reproducible tests;
the diagnostics under
[../runtime/diagnostics/](../runtime/diagnostics/) re-derive the key ones.
