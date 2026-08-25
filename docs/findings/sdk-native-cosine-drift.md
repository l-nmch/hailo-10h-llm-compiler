# Finding 11 — OPEN: `SDK_NATIVE` cosine drifts with model scale, not architecture

**Status: open, root cause not yet isolated.** Documented now so the
`COSINE_MIN` setting it motivates isn't mistaken for an unexplained
tolerance bump, and so the next person doesn't have to re-derive this
table from scratch.

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

A public tip (unrelated third party, general DFC performance advice) claims
`InferenceContext.SDK_BIT_EXACT` gives the same quantized result 3x faster
than `SDK_QUANTIZED`, with two caveats: hide the GPU from TensorFlow
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
route around it. The tip is presumably sound for non-KV-cache-duplicated
graphs (ordinary CNN/vision models); it doesn't apply to this project's
architecture.

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
