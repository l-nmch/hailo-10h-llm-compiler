# Project status

An honest snapshot of what works, what does not, and what is unknown. Read
this before investing time, and update it when you learn something new.

## The one-paragraph summary

The full compile pipeline works: a Hugging Face checkpoint goes through
ONNX → HAR → surgery → quantization → HEF, and the resulting self-contained
HEF loads in genai and hailo-ollama. Prefill inference on hardware is
numerically faithful (cosine ≈ 1.0 against float32). Greedy generation with
the KV-cache disabled is coherent. **Multi-token generation through the
KV-cache path produces degraded text** — one open issue remains
([findings/open-tbt-cache-read.md](findings/open-tbt-cache-read.md)).

## Stage-by-stage

| Stage | State | Notes |
|---|---|---|
| HF → ONNX export | ✅ solid | cosine 1.000000 vs float32; matmul-trick graph; lm_head + last-position slice |
| ONNX → HAR parse (hailo10h) | ✅ solid | SDK_NATIVE cosine 1.000000 exact |
| Graph surgery + resources | ✅ solid | post-surgery cosine 1.000000; RoPE widths + mask wiring match the runtime contract |
| Quantization (KV-cache) | ✅ runs (~30 s GPU) | recipe validated by comparison with official `.alls`; no emulator check possible (see below) |
| Conv repair pass | ✅ kept as safety net | finds 0 issues with the final recipe — its historical cause was ew_add_fusing |
| HAR → HEF compile | ✅ works (~5–8 min) | both network groups emitted; monolithic lm_head places at optimization_level=0 thanks to the last-position slice |
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
the open finding):

1. Instrument the tbt cache read path more deeply (which columns come back
   zeroed, for which scopes/contexts).
2. Compare against a second official KV-cache model's HEF structure beyond
   the recipe level (context descriptors, cache layout declarations).
3. Try `cache_size` / `prefill_size` combinations other than the
   SEQ==CACHE_SIZE assumption.
4. Reproduce on a second Hailo-10H unit to rule out the specific device.

## Provenance note

Every claim above was verified on real hardware with reproducible tests;
the diagnostics under
[../runtime/diagnostics/](../runtime/diagnostics/) re-derive the key ones.
