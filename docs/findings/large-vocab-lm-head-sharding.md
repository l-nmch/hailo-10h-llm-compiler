# Finding 12 — large-vocabulary `lm_head` needs sharded output convs

**Status: fixed and integrated into the pipeline.** `s6_compile_hef.py`
now generates a native `defuse(<layer>, N)` model-script directive
automatically, for every network-group scope present in the HAR, whenever
`config.LM_HEAD_SHARDS > 1`. Proven on hardware-bound HEF compilation
(TinyStories, `VOCAB=32000`, artificially forced to `N=8` shards): compiles
cleanly through both `__prefill`/`__tbt` scopes, and the resulting HEF's
output vstream structure is byte-identical to the unsharded baseline (one
`conv49` output per scope, same shape — the compiler's auto-generated
concat makes the sharding fully transparent to genai/hailo-ollama). Not
yet re-verified on a real large-vocab checkpoint (`VOCAB=151936`) end to
end through step 6 — see "Remaining verification" below.

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

## Fix, attempt 3: the native `defuse` command — WORKS, this is what's shipped

The DFC user guide documents a built-in `defuse(layer, defuse_number)`
model-script command, specifically for this: *"Defusing splits a logical
layer into multiple physical layers... Feature defuse: Each physical
layer calculates part of the output features... Like most mechanisms,
the defuse mechanism happens automatically, so no user intervention is
required."* This is precisely our symptom, and needs zero pipeline
surgery and zero runtime-script changes — the compiler re-concatenates
the physical layers back into one logical output automatically, so it
supersedes attempt 2's HN-edit approach entirely.

First try, on the same TinyStories `convfixed.har`, splitting `conv49`
(the lm_head, `VOCAB=32000`) into 2 (`defuse1, defuse2, defuse_c =
defuse(ts25mpipe/conv49, 2)` — all three return values required, omitting
the auto-generated concat layer throws `AllocatorScriptParserException`):
fails fast (2s) with `Auto defused failed layer ts25mpipe/defuse1
required too many SCs` (subclusters) — each 16000-wide half still doesn't
fit as constructed. **Retried with `defuse_number=8`** (4000-wide
shards): compiles successfully (56.90 MiB HEF), and the resulting HEF was
confirmed byte-for-byte-equivalent on real hardware to the unsharded
baseline (identical generated text). The user guide's own hint —
*"num_splits might be overwritten by a larger number due to hw
limitations"* — matches what was observed: `N=2` too coarse, `N=8` fine.
No principled formula was derived for the minimum safe `N`; `N=8` is an
empirically-found value for `VOCAB=32000`, not a general threshold.

## Integration — done

`s6_compile_hef.py` generates the `defuse(...)` directives automatically
when loading the compiled HAR, gated on `config.LM_HEAD_SHARDS > 1`
(computed from `VOCAB`/`LM_HEAD_MAX_SHARD_WIDTH` in `config.py`, so it's
a no-op for every checkpoint validated so far at `VOCAB` around 32000):

1. Load the HAR's HN model (`runner.get_hn_model()`) before compiling.
2. For each active network-group scope (`__prefill`, `__tbt`, plus the
   base scope with `--include-base-scope`) — **the lm_head layer is
   duplicated once per scope** by `set_kv_cache_global_params` (confirmed:
   `ts25mpipe/conv49`, `ts25mpipe__prefill/conv49`,
   `ts25mpipe__tbt/conv49` all exist as independent HN nodes) — find that
   scope's output layer whose `original_names` starts with `"logits"`
   (there are several other output layers per scope, the KV-cache write
   taps, so filtering by name is required — `hn.get_output_layers()`
   alone is not enough), then take its sole predecessor
   (`out_layer.inputs[0]`, already a plain layer-name string) as the
   layer to defuse.
3. Emit one `defuse(<scope>/<layer>, N)` line per scope, capturing all
   `N+1` return values (`N` shards + the auto-generated concat) even
   though the pipeline never references them by name afterward — DFC
   requires every return value to be bound or the model script fails to
   parse.

The ONNX-side sharding code from attempt 1 (`s1_export_onnx.py`'s
`Wlm_shards`, multi-output `torch.onnx.export`) was reverted back to a
single monolithic `Wlm`/`"logits"` output — `s1`/`s2` always produce
exactly one lm_head output now; only `s6` needs to know about `N`, and
only at compile time via a model-script directive, not an export-time
graph shape.

**Verified through the full integration** (not just the standalone proof
above): ran `s6_compile_hef.py --workdir <tinystories-workdir>` with
`LM_HEAD_SHARDS` forced to 8 in `run_config.json` — the script correctly
found and defused `conv49` in both `__prefill` and `__tbt` scopes,
compiled cleanly (43.60 MiB, matching the unsharded 43.94 MiB baseline
closely), and `hailo_platform.HEF(...).get_output_vstream_infos()` on the
result confirmed each scope still exposes exactly one `conv49` output at
the full `(1, 1, 32000)` shape — the sharding is fully invisible past
compile time, exactly as the `defuse` mechanism promises. A default
(`LM_HEAD_SHARDS=1`) regression run on the same checkpoint was also
re-verified unaffected (identical 43.94 MiB HEF, no `defuse` lines
emitted).

## Remaining verification

Not yet done: recompile `tabularisai/Qwen3-0.3B-distil` (already
validated correct through step 4 quantization with the *unsharded*
`lm_head` — see the `head_dim` commit) through step 6 at its real
`VOCAB=151936` shard count, confirm the HEF compiles, then test base-scope
generation on hardware for coherent output, same bar used throughout this
project (`docs/status.md`'s "real English, cosine ≈ 0.99"). Also open:
whether `LM_HEAD_MAX_SHARD_WIDTH=32000` (implying `ceil(151936/32000) =
5` shards for Qwen3) is actually wide enough — the empirical evidence so
far only confirms `N=2` fails and `N=8` works at `VOCAB=32000` (4000-wide
shards); `N=5` at `VOCAB=151936` implies ~30K-wide shards, close to the
`N=2` failure's 16000-wide shards in absolute terms but a different
vocabulary, so this needs its own empirical check rather than assuming
the TinyStories result transfers directly.
