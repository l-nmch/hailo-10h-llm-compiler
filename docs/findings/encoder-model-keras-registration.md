# Finding 10 — compiling encoder-only models (BERT/MiniLM-style)

**Status: validated on hardware, out of this project's main scope.** This
pipeline otherwise targets causal LLaMA2-style decoders exclusively. This
page documents a working side-path for encoder-only (embedding) models,
in case it's ever worth extending the pipeline's scope.

## Source model and prior art

Base checkpoint:
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(Apache-2.0) — a 6-layer BERT-style sentence-embedding encoder, 384-dim
output.

The ONNX export cut point used here (`/embeddings/Add_1`, i.e. right
after the embedding sum, feeding the encoder stack directly) reproduces
the one used by
[`cstr/all-MiniLM-L6-v2-hailo10h`](https://huggingface.co/cstr/all-MiniLM-L6-v2-hailo10h)
on Hugging Face — a third-party Hailo-10H compile of this same model,
independent confirmation that DFC 5.3.0 is usable by third parties
outside Hailo for non-LLM compiles too. That repo's own export script
isn't published; the cut point was deduced from its docstring and
reproduced independently here by exporting the full HF model and slicing
the ONNX graph at the matching node.

Any similar encoder-only checkpoint should hit the same DFC bug and fix
below, since it's triggered by DFC's own optimizer code, not by anything
model-specific.

## Attribution

The core workaround below is adapted, under its MIT license, from
[ruvnet/RuVector](https://github.com/ruvnet/RuVector)
(`crates/ruvector-hailo-cluster/deploy/compile-encoder-hef.py`), which
first compiled `all-MiniLM-L6-v2` to a working Hailo-8 HEF. This project
reproduced the same recipe against DFC 5.3.0 targeting `hailo10h` — the
hardware-architecture string is the only difference the SDK codepath that
hits the bug below actually cares about.

## Symptom

Loading an all-MiniLM-L6-v2 (or similar BERT-style encoder) HAR through
DFC's optimizer crashes with a `KeyError` deep inside a Keras
JSON-serialize/deserialize round-trip triggered by
`_decompose_layer_norm`, on any custom `acceleras` layer class that isn't
registered as Keras-serializable (e.g. `ElementwiseAddDirectOp`).

## Fix

1. Before importing `ClientRunner`, walk every module under
   `hailo_model_optimization.acceleras` and register every
   `keras.layers.Layer` subclass found with
   `keras.saving.register_keras_serializable()`. This is a superset of
   the acceleras-registration preamble already used elsewhere in this
   pipeline ([sdk-behavior-notes.md](sdk-behavior-notes.md) "Keras
   deserialization" item) — that one only covers layers this project's
   own graphs use; the encoder path needs the whole package walked.
2. `model_optimization_config(globals, multiproc_policy=disabled)` is
   non-negotiable here: without it, the optimizer's calibration step runs
   in a spawned subprocess that never sees the monkey-patched
   registrations, and the same `KeyError` resurfaces there instead.
3. Drop the `attention_mask` input entirely rather than wiring it through
   the graph. The encoder runs full (unmasked) attention over padded
   positions on-chip; the host applies the real mask during mean-pooling
   of the output embeddings, after inference. Calibration data is shaped
   `[batch, 1, seq, hidden]` (NCHW) — DFC's HN treats non-image inputs as
   4D with an implicit channel dimension of 1.

## The recipe

```text
model_optimization_config(calibration, batch_size=8, calibset_size=48)
model_optimization_config(globals, multiproc_policy=disabled)
pre_quantization_optimization(equalization, policy=enabled)
pre_quantization_optimization(ew_add_fusing, policy=disabled)
model_optimization_flavor(optimization_level=0, compression_level=0)
pre_quantization_optimization(matmul_correction, layers={matmul*}, correction_type=zp_comp_block)
model_optimization_config(negative_exponent, layers={*}, rank=0)
quantization_param({ew_add*}, precision_mode=a16_w16)
```

Derived from Hailo's own generic BERT recipe
(`cfg/alls/generic/bert_base_uncased.alls` in
[hailo-ai/hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo),
MIT-licensed), minus the `set_input_mask_to_softmax()` directive — there
is no second (mask) input for it to refer to once the mask is dropped per
the fix above.

## Results

A HEF (`minilm-l6-ruvector.hef`, ~12 MiB) was produced successfully
end-to-end (translate → optimize → compile). An earlier attempt without
the recipe's specific `matmul_correction`/`negative_exponent` overrides
failed at placement (`Resources presolve failed`, insufficient LCUs) —
the recipe above is what got a working compile, not just a working
quantization. No INT8-vs-FP32 cosine number survives from this run; an
evaluation script (dequantize, mean-pool, L2-normalize, compare cosine
against an ONNX Runtime FP32 reference) exists as the intended
verification method but its output wasn't captured to a log — rerun it
before relying on this recipe for anything beyond "it compiles".

## Caution for higher-precision attempts on similar encoders

This recipe stays at the default INT4/INT8 precision path (only
`ew_add*` bumped to `a16_w16`). Fernando_Soria, attempting **full**
`a16`/`w16` precision on a similar sentence-transformer encoder
(`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, same
architecture family), hit a three-stage compiler crash cascade, reported
on the [public Hailo forum](https://community.hailo.ai/t/hailo-10h-dfc-v5-3-0-a16-w16-on-a-transformer-encoder-is-not-a-blanket-allocator-wall-its-a-3-stage-cascade-attention-crash-a16-conv-nan-exponent-needs-super-defuse-what-is-the-intended-16-bit-path/19530):
`BackendAllocatorException` on 16-bit attention softmax, a NaN-exponent
error from `a16`'s high/low convolution decomposition when residuals are
near-zero, and an `Assignment needs super-defuse` error on non-convolution
layers with no Hailo-side workaround at time of writing. Not hit here —
this recipe never asks for full 16-bit — but worth knowing before pushing
precision higher on this model family.

A [follow-up post by Fernando_Soria on the same thread](https://community.hailo.ai/t/hailo-10h-dfc-v5-3-0-a16-w16-on-a-transformer-encoder-is-not-a-blanket-allocator-wall-its-a-3-stage-cascade-attention-crash-a16-conv-nan-exponent-needs-super-defuse-what-is-the-intended-16-bit-path/19530/3)
reports a second, independent finding worth carrying over here: on a
12-layer transformer encoder, enabling the
accuracy stages (`equalization`, `finetune`, `bias_correction`) at
`optimization_level>0` **collapsed retrieval top-1 accuracy from 58.3% to
4.2%** — catastrophic, not a minor regression — with `optimization_level=0`
recommended as the only safe setting for encoder models. This recipe
already uses `optimization_level=0` and no accuracy stages, consistent
with that finding. Notable because this project separately found
`bias_correction` alone measurably *improves* cosine on a causal LLM
decoder (`quantization-recipe.md`) — the two results aren't in tension,
they suggest whether accuracy stages help or hurt is architecture-class
dependent (causal decoder vs. bidirectional encoder), not a universal
rule either way. Re-verify before assuming either direction on a new
architecture.

## Scope note

This is a materially different capability than the rest of this
repository — an encoder producing a fixed-size embedding, not an
autoregressive decoder producing tokens through a KV-cache. It shares
almost nothing with the `__prefill`/`__tbt` HEF contract (see the wiki's
"Anatomy of an LLM HEF" page) beyond both going through the same DFC.
Kept here as a validated reference in case the project's scope ever grows
to cover embedding models.
