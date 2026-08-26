# Finding 12 — OPEN: large-vocabulary `lm_head` needs sharded output convs

**Status: fix found and proven on hardware-bound HEF compilation
(TinyStories proof of concept, post-parse HN-level split — "Fix, attempt
2" below); not yet integrated into the pipeline or verified on a real
large-vocab checkpoint end to end.** The first fix attempt (ONNX-side
sharding) failed on a second, independent DFC parser limitation — see
"Fix, attempt 1". Currently blocks HEF compilation
(step 6) for any checkpoint with a large vocabulary — specifically, every
real Qwen3 checkpoint, since they all share the ~152K-token Qwen
tokenizer regardless of model size.

## Symptom

Step 6 (`s6_compile_hef.py --include-base-scope`) fails placement on the
final `lm_head` matmul, on checkpoints with `VOCAB` in the ~150K range,
independent of `hidden`/layer count:

```
No successful assignments: <scope>/conv215_dc errors:
	Agent infeasible
```

Confirmed on two real checkpoints during QK-Norm/`head_dim` validation
(see [sdk-native-cosine-drift.md](sdk-native-cosine-drift.md) and this
project's `feat(config): read explicit head_dim` commit): a 2-layer toy
Qwen3 model (`yujiepan/qwen3-tiny-random`, `hidden=64`) and a 14-layer real
one (`tabularisai/Qwen3-0.3B-distil`, `hidden=1024`) both fail at the exact
same op type with the exact same error, ruling out model depth/hidden size
as the driver. Inspecting the failing layer directly confirms it's the
`lm_head` matmul, already reduced to a single position by this project's
existing last-position-slice fix
([output-shape-last-position.md](output-shape-last-position.md)):
`input_shapes: [[-1, 1, 1, 1024]]`, `output_shapes: [[-1, 1, 1, 151936]]`.
The single-position slice alone isn't enough once `VOCAB` itself is large
— a `[1024] -> [151936]` matmul is still too wide to place as one op,
regardless of how few rows it's evaluated over.

## Investigation

Every checkpoint this pipeline has validated end to end until now used
`VOCAB` around 32000 (TinyStories, Felladrin, JackFram, `tinyllama-15M`).
Real Qwen3 checkpoints (and Qwen2/2.5) all use the shared ~152K-token
tokenizer regardless of model size — so this wasn't visible until testing
the Qwen3 family specifically.

Inspected an official Hailo-compiled `Qwen2.5-1.5B-Instruct.hef` directly
(`hailo_platform.HEF(...).get_output_vstream_infos()`) to see how Hailo's
own production pipeline handles a large vocabulary at an even larger
`hidden` (1536, vs. this project's largest tested checkpoint at 1024).
**It does not use a single monolithic `lm_head` output.** The `__tbt`
network group exposes four separate output vstreams:

```
OUT qwen2__tbt/block29__conv1 (1, 1, 37984)
OUT qwen2__tbt/block29__conv2 (1, 1, 37984)
OUT qwen2__tbt/block29__conv3 (1, 1, 37984)
OUT qwen2__tbt/block29__conv4 (1, 1, 37984)
```

`4 x 37984 = 151936` — the full vocabulary, split into four independently
placed conv/matmul ops, each within the range this pipeline already
compiles successfully today (~32-38K). The host presumably concatenates
all four before argmax/top-k, the same way it already reconstructs other
host-side-composed tensors in this project's runtime contract.

## Root cause

This pipeline's `lm_head` export (`ExportableModelWithHead` in
`s1_export_onnx.py`) always emits **one** matmul against the full `Wlm`
weight matrix (`[HIDDEN, VOCAB]`). At `VOCAB` around 32000 this places
fine. Official Hailo tooling never emits a single op that wide for a large
vocabulary — it shards the output weight matrix into several
independently-placed chunks. This project's exporter has no equivalent
sharding logic, so it hits the same placement wall official tooling
avoids by design.

## Fix, attempt 1 (ONNX-side sharding): tried, does not work

First attempt: shard `Wlm` into `N` column-blocks at export time (`N`
separate `x @ Wlm_i` matmuls in `ExportableModelWithHead.forward()`,
`N` separate ONNX/HN output nodes) — implemented in `s1_export_onnx.py`.
This broke step 4 (quantization) with a new, different error:

```
UnsupportedModelError: Unexpected input_shapes at normalization layer
<scope>/mul_and_add71 (translated from /Mul_1),
input_shapes=[[-1, 1, 1, HIDDEN]] * N
```

`mul_and_add71` is the final `rms_norm`'s weight-multiply, translated
from a single ONNX node named `/Mul_1`. **Confirmed this is not an ONNX
export artifact**: inspected `model.onnx` directly with the `onnx`
Python package — there is exactly one `/Mul_1` node, with exactly 2
inputs (`x`, `norm_w`) and exactly one direct consumer (the
last-position `Slice`). Even after inserting an explicit no-op
(`x_last + 0.0`) between the slice and the `N` shard matmuls specifically
to give the fan-out point its own distinct node, the resulting ONNX graph
is provably clean — one linear chain down to the inserted `Add`, which
then correctly fans out to `N` separate `MatMul` nodes, no duplication
anywhere — and DFC produced **the exact same error, byte-for-byte**,
including the `mul_and_add71` layer name. This rules out the ONNX graph
as the cause entirely: the duplication happens inside DFC's own
translator, while it builds the HN graph from `N` declared end nodes
(`logits_0`...`logits_{N-1}`) that share a deep common ancestor. Walking
backward from each end node independently appears to (re)traverse and
duplicate every shared upstream node — including ones far removed from
the fan-out point, like the final normalization — rather than recognizing
already-visited shared nodes. This looks like a genuine DFC parser
limitation with multi-output graphs sharing deep ancestry, not something
fixable from the ONNX-export side.

The `x_last + 0.0` insertion and the `if len(self.Wlm_shards) > 1`
conditional are still in `s1_export_onnx.py` as of this writing — reverted
attempt, kept a no-op for the (unsharded) `N == 1` common case pending a
decision on how to proceed. **Sharding is not currently functional.**

## Fix, attempt 2: shard after parse, at the HN level — WORKS

Rather than asking DFC's ONNX translator to build a multi-output graph
from scratch (attempt 1's failure mode), parse the model normally with
its original single monolithic `lm_head` matmul — a graph shape this
translator already handles correctly, single end node, no duplication —
then perform the split as a **post-parse HN edit**, the same technique
`s3_surgery_and_resources.py`'s `mask_surgery()` already uses successfully
for the attention-mask rewiring. This sidesteps DFC's ONNX-to-HN
translation path for the multi-output case entirely — the translator
never sees more than one end node.

**Verified working end to end on TinyStories** (artificially split into
2 shards as a proof of concept — `VOCAB=32000` doesn't need sharding for
real use, this was purely to validate the mechanism): extracted
`resources.har`'s tarball, edited `<scope>.hn` directly (JSON) and
`<scope>.hdf5` directly (the actual weights — `h5py`, datasets named
`<scope>/<layer>/<param>:0/value`), re-tarred, ran the result through
steps 4-6 unmodified. **Quantization (step 4) — the exact step attempt 1
crashed at — completed cleanly.** Compilation (step 6) succeeded on all
three network groups, HEF written (55.59 MiB, matching the unsharded
56.20 MiB baseline closely).

Concretely, per shard `i` of `N`:

1. Locate the final matmul/conv layer's HN dict entry (identified by
   being the sole predecessor of the graph's `output_layer`; original
   name `/MatMul_4` in this pipeline's export).
2. Copy its dict, override `output_shapes` to the shard's column width,
   `original_names` to `["logits_{i}"]`, and `params.kernel_shape`'s last
   dim.
3. Slice the real weights out of the `.hdf5` (`kernel[:, :, :, lo:hi]`,
   `bias[lo:hi]`) and write them under the new shard layer's name.
4. Add a corresponding `output_layer` dict per shard, wired to the new
   conv.
5. Rewire the *predecessor* of the original matmul (e.g. the
   last-position `slice`) to list all `N` new shard convs as its
   `output` — this step is easy to miss and produces a validation error
   (`InvalidHNError: output named ... is not found`) if skipped.
6. Delete the original matmul and `output_layer` dict entries.
7. Update `net_params.output_layers_order` — note this lists the *conv*
   layer names, not the `output_layer` wrapper names.
8. Re-tar.

## Integration (not yet done)

The proof-of-concept lives as an ad-hoc script, not yet integrated into
the pipeline proper. To land it: add the split as a new step in
`s3_surgery_and_resources.py` (alongside `mask_surgery()`, gated on
`config.LM_HEAD_SHARDS > 1` so it's a no-op for every checkpoint
validated so far), operating on the already-loaded HN dict/params rather
than a separate tar-extract pass. The ONNX-side sharding code added
during attempt 1 (`s1_export_onnx.py`'s `Wlm_shards`,
`LM_HEAD_MAX_SHARD_WIDTH`) should be reverted back to a single monolithic
`Wlm` — the ONNX/step-1/step-2 layers should go back to always producing
exactly one output; only the post-parse HN step needs to know about `N`.

## A simpler native alternative exists (`defuse`), but doesn't work out of the box either

The DFC user guide documents a built-in `defuse(layer, defuse_number)`
model-script command, specifically for this: *"Defusing splits a logical
layer into multiple physical layers... Feature defuse: Each physical
layer calculates part of the output features... Like most mechanisms,
the defuse mechanism happens automatically, so no user intervention is
required."* This is precisely our symptom — worth trying before
committing to the HN-edit approach, since it would need zero pipeline
surgery and zero runtime-script changes (the compiler re-concatenates the
physical layers back into one logical output automatically).

Tried on the same TinyStories `convfixed.har`, splitting `conv49` (the
lm_head, `VOCAB=32000`) into 2: `defuse1, defuse2, defuse_c =
defuse(ts25mpipe/conv49, 2)` (all three return values required — omitting
the auto-generated concat layer `defuse_c` throws
`AllocatorScriptParserException`). Loads and starts compiling, but fails
fast (2s) with a different error than attempt 1: `Auto defused failed
layer ts25mpipe/defuse1 required too many SCs` (subclusters) — each
16000-wide half apparently still doesn't fit as constructed, or is
missing an explicit `compilation_param(..., resources_allocation_
strategy=manual_scs_selection, number_of_subclusters=N)` override (the
guide's compilation-parameters section suggests defuse's automatic SC
allocation may need manual tuning for extreme cases like this). Not
pursued further given the working alternative below already exists;
worth revisiting since it would be the cleaner fix if made to work — no
HN/hdf5 surgery, no runtime-script updates.

## Verification plan (remaining)

Two paths forward, either untested end-to-end yet:

1. **`defuse()` native command** — try higher `defuse_number` (more,
   narrower physical layers) and/or explicit `compilation_param` SC
   overrides on the defused layers; if it can be made to work, prefer it
   over the HN-edit approach — it needs zero runtime-script changes since
   the compiler reconstructs one logical output automatically.
2. **HN-level split (attempt 2, proven)** — integrate into
   `s3_surgery_and_resources.py` as described above.

Whichever lands: recompile `tabularisai/Qwen3-0.3B-distil` (already
validated correct through step 4 quantization with the *unsharded*
`lm_head` — see the `head_dim` commit) through step 6, confirm the HEF
compiles at `VOCAB=151936`'s actual shard count (5, not the 2 used in the
TinyStories proof of concept), then test base-scope generation on
hardware for coherent output, same bar used throughout this project
(`docs/status.md`'s "real English, cosine ≈ 0.99"). If the HN-edit
approach is what lands, runtime scripts
(`runtime/diagnostics/generate_base_scope.py`, `runtime/genai_generate.py`,
hailo-ollama serving path) still need updating to gather `N` outputs and
concatenate before argmax/top-k — untested so far, the TinyStories proof
of concept was only verified through HEF compilation, not hardware
inference. The `defuse()` path would not need this runtime-script work.
