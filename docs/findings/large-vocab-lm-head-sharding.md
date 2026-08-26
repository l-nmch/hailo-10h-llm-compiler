# Finding 12 — OPEN: large-vocabulary `lm_head` needs sharded output convs

**Status: root cause identified from official HEF structure; first fix
attempt (ONNX-side sharding) failed on a second, independent DFC parser
limitation — see "Fix, attempt 1" below.** Blocks HEF compilation
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

## Fix, attempt 2 (not yet tried): shard after parse, at the HN level

Rather than asking DFC's ONNX translator to build a multi-output graph
from scratch (attempt 1's failure mode), parse the model normally with
its original single monolithic `lm_head` matmul — a graph shape this
translator already handles correctly, single end node, no duplication —
then perform the split as a **post-parse HN edit**, the same technique
`s3_surgery_and_resources.py`'s `mask_surgery()` already uses successfully
for the attention-mask rewiring: load the parsed HN, locate the final
matmul/conv layer, replace it with `N` smaller conv layers each holding a
column-slice of the original weight matrix, wire each as its own
`OutputLayer`, save. This sidesteps DFC's ONNX-to-HN translation path
for the multi-output case entirely — the translator never sees more than
one end node — at the cost of needing to understand and replicate the
DFC-internal `.hn` layer/weight format for a conv layer directly (more
invasive than `mask_surgery()`'s pure rewiring, which never had to
fabricate new layers from scratch).

## Verification plan

Once a working sharding mechanism exists (attempt 2 or otherwise):
recompile `tabularisai/Qwen3-0.3B-distil` (already validated correct
through step 4 quantization with the *unsharded* `lm_head` — see the
`head_dim` commit; step 4 with `N==1` still works, this finding only
blocks the actual large-`VOCAB` case) through step 6 and confirm the HEF
compiles; then test base-scope generation on hardware for coherent
output, same bar used throughout this project (`docs/status.md`'s "real
English, cosine ≈ 0.99"). Runtime scripts
(`runtime/diagnostics/generate_base_scope.py`, `runtime/genai_generate.py`,
hailo-ollama serving path) still need updating regardless of which fix
lands, to gather `N` outputs and concatenate before argmax/top-k.
