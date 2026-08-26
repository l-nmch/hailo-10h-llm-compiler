# Finding 12 — OPEN: large-vocabulary `lm_head` needs sharded output convs

**Status: root cause identified from official HEF structure; fix not yet
implemented.** Blocks HEF compilation (step 6) for any checkpoint with a
large vocabulary — specifically, every real Qwen3 checkpoint, since they
all share the ~152K-token Qwen tokenizer regardless of model size.

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

## Fix (not yet implemented)

Shard `Wlm` into `N` column-blocks (e.g. matching the official ~38K chunk
width, or whatever this hardware's placer reliably accepts — needs a
determined safe threshold, not just copying "4" blindly) at export time:
`N` separate `x @ Wlm_i` matmuls instead of one, each becoming its own
ONNX/HN output node. Concatenation back into a single logits vector
happens host-side in the runtime scripts
(`runtime/diagnostics/generate_base_scope.py`,
`runtime/genai_generate.py`, hailo-ollama serving path) — all of which
currently assume a single output tensor and need updating to gather `N`
outputs and concatenate before argmax/top-k.

## Verification plan

Once implemented: recompile `tabularisai/Qwen3-0.3B-distil` (already
validated correct through step 4 quantization — see the `head_dim`
commit) through step 6 and confirm the HEF compiles; then test base-scope
generation on hardware for coherent output, same bar used throughout this
project (`docs/status.md`'s "real English, cosine ≈ 0.99").
