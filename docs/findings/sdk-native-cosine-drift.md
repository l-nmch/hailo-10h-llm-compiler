# Finding 11 — ROOT CAUSE FOUND: DFC's softmax normalizes across heads instead of per-head

**Status: root cause confirmed bit-exact (cosine 0.9999999999996048) in
`SDK_NATIVE`.** What started as an unexplained cosine-drift correlation
with model scale turned out to be one specific, reproducible bug: DFC's
compiled softmax for this project's head-tiled causal-mask scheme reduces
over the entire `NHEAD*SEQ` axis as a single softmax instead of `NHEAD`
independent per-head softmaxes — see "Root cause confirmed" below for the
proof. Still open: whether the compiled/quantized HEF (not just the
`SDK_NATIVE` float32 simulation measured here) shares the same mechanism,
and exactly where in the graph the per-head boundary is lost.

## Symptom

Step 2's fidelity gate (cosine of DFC's `SDK_NATIVE` simulation vs
float32 HF, normally > 0.999) degrades on checkpoints other than the
validated 4-layer/256-hidden default — while step 1's gate (pure PyTorch
reimplementation + ONNX export vs HF, same threshold) stays at or near
1.0 on every checkpoint tried except the most extreme one. The divergence
is introduced specifically inside DFC's own simulation, not in this
project's exporter.

## Investigation

Five checkpoints compiled end to end through step 1/2 (see
`Porting-Another-Model.md`'s eligibility screen — all pass it: untied
embeddings, no attention/MLP bias, standard RoPE):

| Checkpoint | hidden | heads (Q/KV) | layers | step 1 cosine | step 2 (`SDK_NATIVE`) cosine |
|---|---|---|---|---|---|
| TinyStories-LLaMA2-25M (validated default) | 256 | 16/8 (GQA) | 4 | 1.000000 | 1.000000 |
| Maykeye/TinyLLama-v0 | 64 | 16/16 (MHA) | 8 | 1.000000 | 0.995502 |
| Felladrin/Smol-Llama-101M-Chat-v1 | 768 | 24/8 (GQA) | 6 | 0.999994 | 0.978675 |
| JackFram/llama-160m | 768 | 12/12 (MHA) | 12 | 1.000000 | 0.972556 |
| TinyLlama/TinyLlama_v1.1 | 2048 | 32/4 (GQA) | 22 | 0.768992 | *(never reached step 2 — failed at step 1)* |

**GQA vs MHA is not the driver**: Felladrin (GQA) and JackFram (MHA) land
at nearly the same cosine (0.979 vs 0.973) despite different attention
mechanisms. **Layer count alone doesn't explain it either**: Felladrin (6
layers) degrades *more* than Maykeye (8 layers). The one clean pattern:
Felladrin and JackFram both have `hidden=768` and land within 0.006 cosine
of each other, while the only checkpoint with `hidden=256` is exact.
Working hypothesis: float32 accumulation drift between `SDK_NATIVE` and
PyTorch/ONNX Runtime grows with model **scale** (hidden size, plausibly
compounded by depth) — not with the attention architecture choice.

TinyLlama_v1.1 (hidden=2048, 22 layers — the largest and deepest
checkpoint tried) is a different, more severe case: it fails **step 1**
(the pure-PyTorch reimplementation check, cosine 0.77), before DFC is
even involved. Its config rules out the obvious suspects (no attention or
MLP bias, standard `rope_theta=10000.0`, no `rope_scaling`,
`rms_norm_eps=1e-5` correctly read).

**Hypothesis tested and refuted**: the checkpoint's loaded model uses the
`LlamaSdpaAttention` class (a fused-kernel attention path with a
different accumulation order than this project's manual
matmul+softmax+matmul reimplementation) despite `_attn_implementation`
reporting `"eager"` in its config — a plausible source of exactly this
kind of scale-amplified divergence. Forcing
`AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")`
in step 1 (now the default for all checkpoints, harmless elsewhere)
produced the **exact same cosine to six decimal places** — no effect
whatsoever. On CPU (no CUDA available in this project's toolchain
images), PyTorch's SDPA falls back to the same reference math as eager,
so the two paths were never actually numerically different here. Ruled
out with evidence, not assumption.

Root cause still open. **Deliberately not investigated further at this
pass** — tracked as the concrete next step below, alongside isolating the
scale-drift operation.

## Mitigation shipped

`config.COSINE_MIN` (default `0.999`, the validated bar) is now a
first-class, explicit setting — `--cosine-min` on step 1, threaded to
steps 1-3 via `run_config.json`. It is never silently below default:
every step that reads a relaxed value prints a warning naming it. This
lets a checkpoint whose drift is understood and judged benign proceed
past step 2, without moving the bar for the validated default path or
hiding the fact that a non-standard bar was used.

**Do not reach for a lower `COSINE_MIN` as a first response to a step 2
failure.** Check whether the failing checkpoint fits the scale pattern
above first; a drop that doesn't fit the pattern (e.g. a small,
shallow model still failing) is more likely a real bug than benign drift.

## Downstream symptom: base-scope generation stays incoherent regardless of calibration size

Tested on hardware whether the `SDK_NATIVE` drift above translates into a
quantization/calibration problem that more calibration data could paper
over. On Felladrin (`hidden=768`, the same checkpoint from the table
above), base-scope greedy generation was compared across three configs,
all INT8 conv precision + `bias_correction`:

| `calibset_size` | Base-scope greedy output |
|---|---|
| 32 | real content words + correct punctuation, not fully coherent |
| 128 | `<s> < " ⏎ - . ( : " ⏎` — almost entirely punctuation/formatting tokens, no content words |

Quadrupling the calibration set made the output **worse**, not better —
rules out "not enough calibration samples" as the driver of Felladrin's
incoherence. Consistent with the scale-drift hypothesis above: the defect
tracks `hidden` size and is visible even in `SDK_NATIVE` (pre-quantization)
simulation, so no purely quantization-side knob (calibration size,
precision, bias_correction) is expected to fix it on its own. Next
candidate to isolate: the layer-by-layer probe described below, run once
before spending more hardware time on calibration-size sweeps.

## Attempted: tapping post-quantization activations to test causality, not just correlation

To distinguish whether the base-scope incoherence above is *caused* by the
same operation driving `SDK_NATIVE` drift, or is an independent symptom
that merely shares `hidden` size as a confound, the plan was: compare
intermediate activations at matching layers in `SDK_NATIVE` (pre-quant)
vs. the actual post-quantization/on-chip domain. The hardware side is not
reachable — the compiled HEF exposes only the final logits output
(`conv71`), no intermediate taps. The software fallback,
`InferenceContext.SDK_QUANTIZED`, was retried here specifically on the
non-KV-cache-duplicated base scope in case that scope sidesteps the known
cache bugs above — it does not: `Cache.__init__` still requires a
`lora_adapter_name` matching one of `__prefill`/`__tbt` (raises
`UnsupporteLoraAdapterException` otherwise, base scope isn't a valid
adapter name at all), and even forcing `__prefill` specifically hits the
exact same structural concat-shape bug already on record above (a
`[1,1,8,792]` cache-write tensor concatenated against a `[4096,1,16,792]`
conv output — batch/shape mismatch, independent of input data). Confirms
finding #8's assessment is complete and general — this isn't a narrower
bug that only affects `__tbt`; there is no working post-quantization
emulation path for this project's graphs, full stop. Causality vs.
correlation between the `SDK_NATIVE` drift and on-chip incoherence remains
open; only a pre-quantization layer-by-layer probe (comparing `SDK_NATIVE`
against the PyTorch reimplementation) is actually executable in software.

## `SDK_BIT_EXACT` hits the identical structural bug, not a workaround

Fernando_Soria, on the same [public forum thread](https://community.hailo.ai/t/hailo-10h-dfc-v5-3-0-a16-w16-on-a-transformer-encoder-is-not-a-blanket-allocator-wall-its-a-3-stage-cascade-attention-crash-a16-conv-nan-exponent-needs-super-defuse-what-is-the-intended-16-bit-path/19530)
already cited above, reports that `InferenceContext.SDK_BIT_EXACT` gives
the same quantized result 3x faster than `SDK_QUANTIZED`, with two
caveats: hide the GPU from TensorFlow
(`tf.config.set_visible_devices([], "GPU")`) *before* importing the SDK, and
run evaluation with TF on CPU since bit-exact emulation is TF-only. Both
applied here (plus `CUDA_VISIBLE_DEVICES=''`) and retested against
`smol_llama_101m_chat_v1__prefill`. No change: `SDK_BIT_EXACT` reaches
further into the call stack than `SDK_QUANTIZED` did
(`_bit_exact_run` -> `call_bit_exact` -> `call_hw_sim`) but fails on the
exact same `cache_concat_matmul1` concat-shape mismatch
(`[1,1,8,792]` vs `[4096,1,16,792]`), with the identical `Cache` object
passed through `kwargs` in both cases. Conclusion: the bug lives in the
shared `Cache`/cache-concat subsystem itself, not in a particular
`InferenceContext`'s execution path — switching emulation modes cannot
route around it. Fernando_Soria's tip is presumably sound for whatever
non-KV-cache-duplicated graph he measured it on; it doesn't apply to this
project's architecture.

## Next step (not yet done)

Isolate which specific operation accumulates the drift — likely
candidates given the shape of the pattern: softmax normalization,
RMSNorm's variance reduction, or the order of accumulation in the
attention/MLP matmuls at larger hidden widths. A layer-by-layer cosine
probe (compare `SDK_NATIVE` intermediate activations against the PyTorch
reimplementation's, not just the final logits) on `Felladrin/Smol-Llama-101M-Chat-v1`
would localize it without needing TinyLlama_v1.1's expense. Diagnosing
TinyLlama_v1.1's step-1 failure separately is worthwhile once the
scale-drift question is settled — right now it's a confound, not
evidence either way.

**Attempted, blocked on tooling, not yet done.** `HailoNN.update_output_layers_order()`
looked like the way to tap an arbitrary internal layer (`layer_normalization1`,
`softmax1`, etc. — confirmed reachable by name and correctly mapped to
`layers.0`/`layers.3`'s ops via `original_names`), but it only *selects
among already-declared* graph outputs (it calls
`get_real_output_layers_by_recipe()` internally, which is driven by the
existing `output_layers_order`) — it cannot promote an arbitrary internal
node into a new output. Doing that needs either (a) graph surgery that
inserts a real `OutputLayer` node after the target layer, the same
mechanism DFC itself uses internally to expose the `--include-base-scope`
network group (no existing project code to copy — that insertion happens
inside the SDK, not in this pipeline's own surgery step), or (b) dropping
below `ClientRunner.infer()` entirely and pulling the intermediate tensor
directly off the underlying Keras model DFC builds
(`sdk_backend.build_acceleras_model()` returns a `keras.Model`; a
functional-style rebuild with the target layer's output as a new model
output is the standard Keras technique, untried here). Neither attempted
yet — next session should start with (b), it's the smaller lift.

## Resolved: (b) worked — the drift is not gradual, it's a sharp break inside layer 0

`HailoModel.set_output_interal_layers(names)` (note the SDK's own typo,
"interal") is the intended public API for exactly this: call it on the
model returned by `SDKBackend.build_acceleras_model(InferenceContext.SDK_NATIVE)`
(reached via `ClientRunner._sdk_backend`, bypassing `.infer()` entirely),
then `model.predict(inputs)` returns `(final_output, [internal_outputs...])`
in the order requested. Internal layer names are the same HN layer names
used elsewhere (`layer_normalization1`, `softmax1`, ...) — no graph
surgery needed. RoPE inputs on the post-surgery HAR need the tiled widths
from `s3_surgery_and_resources.py`'s `tile_groups()` helper (`NKVHEAD`/`NHEAD`
groups of `HD`), not the untiled `[SEQ, HD]` shape step 2 uses on the
pre-surgery HAR — the two steps' inputs are not interchangeable.

Tapped Felladrin's `input_layernorm` output (pre-attention RMSNorm) at
layers 0/1/3/5 and the final norm, in both `SDK_NATIVE` and a hooked
PyTorch reference (same prompt, same `attn_implementation="eager"` model
as step 1 uses). Note the HN decomposes RMSNorm into a normalize-only
`layer_normalization*` op followed by a separate weight-multiply
`mul_and_add*` op — tap the `mul_and_add*` one to match HF's full RMSNorm
output (confirmed by matching mean/std before trusting the cosine numbers).

| Probe point | cosine (`SDK_NATIVE` vs PyTorch) |
|---|---|
| Layer 0 input_layernorm | **0.999999999998** — bit-exact, not just "close" |
| Layer 1 input_layernorm | 0.676 |
| Layer 3 input_layernorm | 0.366 |
| Layer 5 input_layernorm | 0.333 |
| Final norm | 0.163 |

This rules out gradual float32 accumulation drift as the mechanism — a
gradual-drift hypothesis predicts a smooth cosine decay across depth, not
a cliff between layer 0's input and layer 1's input while layer 0's own
input is essentially perfect. Something inside layer 0's
attention/RoPE/MLP block (between the input_layernorm tap and the next
layer's input_layernorm tap) introduces the bulk of the error in one
step, and every layer after that just carries it forward.

Layer 0's attention softmax was also tapped
(`smol_llama_101m_chat_v1/softmax1`) and compared against HF's
`output_attentions=True` post-softmax weights, but the DFC tensor's
head-tiling layout in the flattened last axis (576 = `NHEAD`×`SEQ`) isn't
confirmed — cosine came out ~0.48 under the best-guess `(seq_q, heads,
seq_k)` reshape, well below layer 0's near-perfect RMSNorm match, which is
at least directionally consistent with the divergence originating inside
layer 0's attention path (RoPE application or the QK/mask/softmax
sequence) rather than in RMSNorm itself — but don't treat 0.48 as a
trustworthy number until the tiling layout is verified against a known
input pattern (e.g. an identity-like calibration input whose expected
attention pattern is analytically known).

## Next step (not yet done), revised

Narrow inside layer 0's block: tap the Q/K/V projection outputs and the
RoPE-rotated Q/K (before the QK matmul) against the same PyTorch hooks,
to find exactly which operation between "RMSNorm output" (verified exact)
and "attention softmax" (already degraded) introduces the divergence.
Given this project's RoPE is reimplemented as an explicit matmul-trick
(rotate-half via constant matrices, not a native op DFC might mishandle
differently), the QK matmul itself and/or the RoPE constant-matrix
multiplication are the sharpest remaining suspects.

## Narrowed further: the break is between raw QK^T scores and the softmax output — exact mechanism still open

Tapped `conv5`/`conv6`/`conv7` (layer 0's raw Q/K/V projections, pre-RoPE),
`ew_mult1-4` (the RoPE `x*cos` / `rotate_half(x)*sin` components), and
`matmul1` (raw QK^T scores, pre-mask, pre-softmax), each compared against
a from-scratch NumPy recomputation seeded from the same HF `q_proj`/
`k_proj` hook outputs (same RoPE math, same GQA `repeat_kv` — confirmed
`repeat_interleave` semantics, not block-concat, by testing both and
taking the one that matches: interleave gives 0.9999754, block-concat
gives 0.178):

| Probe point | cosine vs. from-scratch reference |
|---|---|
| Q/K/V raw projections (`conv5/6/7`) | 0.999988 (all three) |
| RoPE `x*cos` / `rotate_half(x)*sin` (`ew_mult1-4`) | 0.999989 (all four) |
| Raw QK^T scores, pre-mask (`matmul1`) | 0.999975 |
| **`softmax1`, reshaped `(seq_q, NHEAD, seq_k)`-then-transposed and compared per-head to HF's real attention weights** | **0.4768** |

Every stage through raw QK^T scores is bit-exact against a from-scratch
reproduction. `softmax1` is not — but pin down what "not" means precisely,
because an earlier pass through this analysis got it wrong and the
correction matters:

**Retracted claim, corrected here**: an earlier version of this section
claimed DFC's `softmax1` output "doesn't sum to 1 across the key axis"
citing row sums like `[0.037, 0.019, ...]`. That was an artifact of
reshaping `softmax1` to `(seq_q, NHEAD, seq_k)` *before* summing — summing
the raw, un-reshaped `(24, 576)` tensor along its actual last axis gives
row sums of `~1.0` everywhere (`0.9999999`–`1.0000001`), a completely
normal softmax. **DFC's softmax layer does produce a valid, correctly
normalized probability distribution.** The bug (if there is one, see
below) is not "the softmax fails to normalize."

What's still real: reshaping `softmax1` to `(seq_q, NHEAD, seq_k)` and
comparing per-head slices against HF's real per-head attention weights
gives cosine 0.4768, well below the near-1.0 match every earlier stage
gets. Two competing explanations were tested:

1. **Per-head softmax, `(seq_q, NHEAD, seq_k)` layout** — doesn't match
   (that's the 0.4768 above).
2. **A single softmax spanning the full 576-wide tiled axis** (masking
   `matmul1` with `config.causal_mask_tiled()`, exact same tensor
   `input_layer2` receives, then one softmax over all 576 elements per
   row) — closer but still not exact: cosine 0.7703, not ~1.0.

Neither simple hypothesis fully reproduces `softmax1`. Since `matmul1` is
independently confirmed correct (0.999975 vs. from-scratch QK^T) and a
correctly-masked, correctly-per-head softmax of those same scores
reproduces HF's real attention at cosine 0.9999999999999838 (bit-exact,
computed directly, not via DFC's graph), the discrepancy has to be in
either (a) the exact axis/layout DFC's compiled softmax actually reduces
over — possibly a permutation `(seq_q, seq_k, NHEAD)` or similar rather
than `(seq_q, NHEAD, seq_k)` assumed above, which would also explain why
neither simple hypothesis lined up — or (b) a scale/temperature difference
between what `matmul1` stores and what actually feeds the compiled
softmax. Root cause narrowed to this one operation but not yet pinned
down to the exact mechanism.

This still reframes the whole `SDK_NATIVE` drift table at the top of this
document, just less definitively than the retracted version claimed: the
divergence is real, isolated to the masking/softmax stage of layer 0's
attention (not gradual float32 accumulation across depth — everything
before this point is bit-exact), and interacts with each checkpoint's
`NHEAD`/`NKVHEAD`/mask-tiling geometry, which is consistent with
TinyStories' simple layout surviving it while others don't. Whether it's
a genuine DFC bug or a layout-assumption error in this analysis is not
yet settled.

## Root cause confirmed, bit-exact: DFC's softmax is shared across all heads instead of per-head

Rather than guess at the axis layout, read it directly off real data that
was already collected: at query position `q=0`, the causal mask permits
attending *only* to key position `k=0` — true independently of any axis
ordering ambiguity. Inspecting `softmax1`'s raw `(24, 576)` tensor at row
`q=0` directly: exactly 24 nonzero columns, at indices
`0, 24, 48, 72, ..., 552` — i.e. `head * SEQ + 0`, confirming the
`(seq_q, NHEAD, seq_k)` block-per-head layout used throughout this
document is correct after all. But their **values are not uniform** —
`[0.037, 0.206, 0.038, 0.028, 0.001, ..., 0.107]`, spread unevenly across
the 24 heads rather than each head independently landing on exactly
`1.0` (the only mathematically valid value for a per-head softmax with
exactly one unmasked key). **This is the proof, not an inference**: a
correct per-head softmax has no free parameter to produce anything but
`1.0` at `q=0`'s single valid key per head; DFC produces a distribution
across heads instead, meaning the 24 head-slots are competing for shared
probability mass.

Confirmed exactly by reproducing it: scale `matmul1`'s raw QK^T scores by
`1/sqrt(HD)` (`HD=32`, so `1/sqrt(32)`), add `config.causal_mask_tiled()`
(the literal tensor `input_layer2` receives) unchanged, and take **one
softmax over the full 576-wide row** (not per-head) — this reproduces
`softmax1` at **cosine 0.9999999999996048**, bit-exact. (The earlier
"0.7703" result above used the same recipe without the `1/sqrt(HD)`
scale — the missing scale factor, not the axis layout, was why that
attempt fell short.)

**Root cause, stated precisely**: DFC's compiled softmax for this
project's head-tiled attention-mask scheme reduces over the entire
tiled `NHEAD*SEQ` axis as a single softmax, instead of `NHEAD`
independent softmaxes each over `SEQ` keys. Every head's attention
weights are computed by competing against every other head's raw scores
for a shared normalization budget — mathematically wrong multi-head
attention, structurally, not a numerical precision artifact. This fully
explains the `SDK_NATIVE` cosine-drift table at the top of this document
(worse with more heads / larger tiled mask width, not with `hidden` size
per se — `NHEAD` and `hidden` happened to covary across the checkpoints
tested here), why depth compounds it (each layer's attention output is
wrong, corrupting every downstream layer), and is a strong candidate for
the underlying mechanism behind both the base-scope incoherence
documented earlier in this file and
[open-tbt-cache-read.md](open-tbt-cache-read.md)'s KV-cache incoherence —
"real words, wrong order/weighting" is exactly the failure mode a
shared-normalization-across-heads bug produces, and it explains why no
quantization-recipe or calibration-size change (bias_correction,
`calibset_size`) ever helped: the bug is upstream of quantization
entirely, in the float32 `SDK_NATIVE` graph itself.

## Next step (not yet done)

This is now a masking/graph-construction question, not a numerics
question. Locate exactly where in this pipeline's own surgery
(`s3_surgery_and_resources.py`'s `mask_surgery()`, which rewires
`input_layer2` directly into each layer's attention-mask consumer) or in
DFC's own softmax-layer construction the per-head boundary gets lost —
compare against how Hailo's own official multi-head LLM `.alls`/HN
structures per-head softmax reduction (if inspectable) to see whether
this project's mask-tiling convention (`causal_mask_tiled()`, tiling the
same `[SEQ,SEQ]` block `NHEAD` times into the last axis) is fundamentally
incompatible with how DFC's `SoftmaxLayer` infers its reduction axis, or
whether there's a per-layer/per-op config (an axis or `group_size`
parameter) that tells the softmax layer to reduce in blocks of `SEQ`
rather than over the full input width. Also verify whether this affects
the compiled (quantized, on-chip) graph identically — everything measured
in this document is `SDK_NATIVE` (float32 simulation); the compiled HEF's
softmax could plausibly use a different reduction mechanism at the
hardware level, worth checking via `hef_audit.py`'s structural inspection
before assuming the hardware inherits this exact bug.
