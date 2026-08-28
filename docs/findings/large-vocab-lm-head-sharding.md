# Finding 12 — OPEN (likely unfixable on this DFC build): large-vocabulary `lm_head` needs sharded output convs

**Status: two independent, structurally different sharding mechanisms
were built and proven at small scale, then both hit hard DFC-internal
limits before reaching a real ~152K-vocabulary checkpoint (Qwen3
family). Current conclusion: no configuration was found that compiles a
real Qwen-scale vocabulary on DFC 5.3.0.** This is the most
thoroughly-investigated open finding in this project; read all three
attempts below before trying a fourth.

## Why this matters

Every real Qwen2/Qwen2.5/Qwen3 checkpoint shares the ~152K-token Qwen
tokenizer regardless of model size, so this blocks the entire Qwen
family end-to-end on this pipeline, independent of every other fix
(QK-Norm, `head_dim`, tied embeddings — all separately proven working).
It does **not** block smaller-vocabulary checkpoints (TinyStories,
Felladrin, `tinyllama-15M`, all `VOCAB` ≈ 32000), which this pipeline
compiles and runs today with a single monolithic lm_head matmul.

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

**Superseded** by attempt 4 below on real Qwen-scale checkpoints: this
mechanism works at small `N` (proven at `N=2` on the tiny 2-layer Qwen3
toy checkpoint, `VOCAB=151936`) but crashes at `N≥3` with the exact same
DFC-internal fuser bug attempt 1 hit — see attempt 4's "Root cause,
finally pinned down" for the full story; this is not a bug in the HN
surgery itself, it reproduces identically no matter how the multi-output
graph is constructed.

## Fix, attempt 3: the native `defuse` command — WORKS at small scale, fails on real Qwen-scale checkpoints

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

**Fails again on a real large-vocab checkpoint at any workable `N`.**
Retried on the real `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, `VOCAB=
151936`, after separately fixing a real q/k/v-bias export bug this
checkpoint surfaced — see the commit history). `__prefill` always
compiled without issue; `__tbt` alone failed every time:

- `N=5` (the config-computed shard count, `ceil(151936/32000)`) and
  `N=10`: `required too many SCs` (the same too-coarse-a-shard failure
  as TinyStories' `N=2`).
- `N=24`: passes the SC wall, but fails with a **new, deterministic**
  error — `Context-Partition topology error`: a handful of the first
  1-2 layers' RoPE/mask-consuming ops placed in an earlier partition
  bucket than the shared inputs they read from. Reproduced
  byte-for-byte across `performance_param(compiler_optimization_level=1)`,
  `context_switch_param(allow_auto_merge_in_multicontext=True)`, and
  `context_switch_param(mode=disabled)` — none of the three documented
  model-script levers for exactly this symptom class changed the
  outcome at all. ~32 min per attempt. Full detail:
  [large-body-multicontext-topology.md](large-body-multicontext-topology.md).
- `N=32`: both the SC wall (on some shards) and `dc`'s memory-capacity
  overflow simultaneously.
- `N=48`: `dc`'s auto-generated concat tree alone overflows on-chip
  memory (`Memory units capacity exceeded`) — `defuse`'s
  `concat_f_from_concat_f_from_...` naming confirms it nests pairwise
  concats in a linear (not balanced) chain, so memory cost keeps growing
  with `N` past some point regardless of per-shard width.

No `N` was found that avoids all three failure modes simultaneously for
this checkpoint. This is what motivated abandoning `defuse` for attempt
4 below.

## Fix, attempt 4: pre-quantization HN/npz split (no on-chip concat) — WORKS at small N, ALSO fails at real Qwen-scale N

Motivated by inspecting the official `Qwen2.5-1.5B-Instruct.hef` again:
its 4 lm_head output convs are **genuinely independent output vstreams,
concatenated host-side** — there is no on-chip concat at all in the
official recipe. `defuse`'s mandatory auto-generated concat (attempt 3)
is therefore not what official tooling does, and is exactly the layer
implicated in attempt 3's `__tbt`-only failures (SC starvation on the
shards, then a topology bug, then concat memory overflow — all either
directly the concat layer or immediately downstream of it).

Implementation: same technique as attempt 2 (post-parse HN/weights
surgery, no `defuse`), but moved **before** `optimize()` instead of
after quantization+compile-time-adjacent — i.e. it edits
`resources.har` (the output of step 3, still pre-`set_kv_cache_global_params`
duplication, so there is exactly one scope to edit, not three). DFC's
own `DuplicateLLMToNetworkGroups` pass then replicates the N independent
output layers into `__prefill`/`__tbt` automatically, the same as every
other layer — no per-scope handling needed, unlike attempt 3. One format
difference from the original attempt-2 POC: at this pipeline stage
weights live in a `.npz` (plain dict of numpy arrays keyed
`"<layer>/<param>:0"`), not the `.hdf5` the original POC script (written
against a later ClientRunner-processed HAR) assumed.

Implemented as `lm_head_split()` in `s3_surgery_and_resources.py`,
called right after `mask_surgery()`, gated on `config.LM_HEAD_SHARDS > 1`
(no-op for every checkpoint validated at `VOCAB` around 32000). The
downstream `isinstance(out, list)` handling already present in
`s2_parse_har.py`/`s3_surgery_and_resources.py`'s cosine checks (kept
from attempt 1, see below) picks up the multi-output case for free.

**Root cause, finally pinned down.** Retested on a 2-layer toy Qwen3
checkpoint (`yujiepan/qwen3-tiny-random`, `VOCAB=151936` — same
tokenizer as every real Qwen checkpoint, but a body cheap enough to
iterate on in seconds instead of ~30 minutes) to bisect fast:

| `N` | step 4 (quantize) |
|---|---|
| 2 | ✅ works |
| 3 | ❌ fails |
| 5 | ❌ fails |

All three failures are **the exact same error attempt 1 hit at the ONNX
level**: DFC's post-fuser `normalization_optimizer.py`'s
`_move_normalization_layers_after_unfuseable_layers()` →
`fuser_helper.swap_layers_order()` sets `input_shapes` on the shared
ancestor normalization layer to a list with one entry **per shard**
(`[[-1,1,1,64]] * N`) instead of one, then `Layer.set_input_shapes()`
rejects it: `UnsupportedModelError: Unexpected input_shapes at
normalization layer ...`. This is now confirmed, across three
structurally different ways of building the multi-output graph (ONNX
export, `defuse`, direct HN/npz surgery), to be **a genuine DFC-internal
bug/limitation in the post-fuser's handling of any layer whose output
fans out to ≥3 independently-tracked successors sharing deep ancestry**
— not an artifact of any one construction method. `N=2` is a special
case the fuser's swap logic happens to handle (likely because
`swap_layers_order()` is written for a binary swap, `first_degree_succs[0]`,
and just happens not to crash when there are only 2 successors total).

**This closes off both viable sharding mechanisms for real Qwen-scale
vocabularies**: `N=2` avoids the fuser bug but produces ~76K-wide shards
— far past the ~32K-38K width every other data point in this
investigation (TinyStories, the official Hailo HEF's own 4×37984-wide
convs) shows is the actual placement ceiling. `N≥3` avoids the
width-placement wall but crashes the fuser unconditionally. **No value
of `N` satisfies both constraints at once for `VOCAB≈152000`.**

## Current status: open, likely unfixable on DFC 5.3.0

Three independent construction methods (ONNX multi-output export,
`defuse()`, direct HN/npz surgery) all hit the identical post-fuser
`normalization_optimizer` crash once fan-out reaches 3, and a fourth,
independent placement-width ceiling (~32K-38K, corroborated by the
official Hailo HEF's own output-conv width) makes `N=2` unusable for a
152K-token vocabulary. Together these bracket out every value of `N`.
Real Qwen2/Qwen2.5/Qwen3 checkpoints remain blocked on this DFC version
regardless of which sharding mechanism is used; every other fix on this
generalization branch (tied embeddings, QK-Norm, explicit `head_dim`,
q/k/v projection biases) is proven correct independent of this wall.

**What would actually move this forward**, none attempted:

1. `pre_quantization_optimization(defuse, layers=X, num_splits=N,
   defuse_type=MHA)` — a different, attention-block-specific defuse
   mechanism documented separately from the compilation-time `defuse()`
   used in attempt 3 (see the DFC user guide's Model Optimization
   section, not the Model Compilation section). Untested against this
   symptom — it targets the attention block's first matmul, not
   lm_head, so may not even apply, but was found only after this
   investigation's attempts were already exhausted.
2. Report the post-fuser bug upstream — it is reproducible, minimal
   (any conv with ≥3 independently-tracked successors sharing an
   ancestor normalization layer), and version-specific to DFC 5.3.0;
   worth checking against a newer DFC release if one becomes available.
3. A fundamentally different approach to the vocabulary problem
   entirely — e.g. quantizing lm_head to a narrower effective width
   somehow, or restructuring the export so the shared ancestor
   normalization layer itself isn't shared (duplicate the final RMSNorm
   N times before the split, so the fuser never sees a fan-out ≥3 point)
   — untested, plausible given the bug is specifically about the fuser's
   handling of the *shared ancestor*, not the split itself.
