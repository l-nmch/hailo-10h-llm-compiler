# Finding 8 — SDK and toolchain behavior notes

**Status: reference page. Each item cost real debugging time; collected
here so it costs you less.**

## Quantized emulator is structurally broken on KV-cache graphs

`InferenceContext.SDK_QUANTIZED` inference — the standard way to measure
post-quantization cosine — fails in two stacked, independent ways:

1. `acceleras/utils/cache.py::_get_prefill_size` raises `TypeError` when
   `lora_adapter_name=None`. Universal: reproduces on official KV-cache
   HARs too.
2. Patching past that, a structural shape inconsistency appears inside
   `__prefill/matmul1` (256 vs 384), independent of supplied data. Deeper
   emulator limitation; never resolved.

**Consequence:** after step 4 there is no software fidelity check.
Hardware is the only judge. Plan iteration cycles accordingly.

## `optimization_level > 0` silently re-enables accuracy stages

Setting a non-zero optimization level implicitly enables adaround,
bias_correction and finetune. If your recipe deliberately excludes them,
you must pin `optimization_level=0` or they come back without warning.

## Keras deserialization needs the acceleras registration preamble

Loading an optimized HAR with `ClientRunner(har=...)` fails with "Unknown
layer" unless every internal acceleras layer class is registered as
Keras-serializable first. [pipeline/config.py](../../pipeline/config.py)'s
`register_acceleras_layers()` does the walk-and-register; import config
before touching HARs.

## SDKPaths patch outside release installs

In non-release DFC installs, the paths-manager singleton can point at a
nonexistent build directory and break compiles. Forcing release mode plus a
fresh temp build dir (see `patch_sdk_paths()` in
[pipeline/s6_compile_hef.py](../../pipeline/s6_compile_hef.py)) works
around it host-side.

## Host-side EINTR interruptions of LLM sessions

Long-lived processes driving genai can hit an interrupt/ioctl failure
around session accept (`HAILO_PCI_EP_ACCEPT`). It is environmental, not
model-related, and retrying in a fresh process succeeds. This is why
[runtime/genai_generate.py](../../runtime/genai_generate.py) runs every
attempt in its own short-lived subprocess with a hard timeout instead of
keeping one session open.

**Confirmed root cause**, from the public MIT-licensed
[hailort_driver.cpp](https://github.com/hailo-ai/hailort) (`HailoRTDriver::pci_ep_accept`):
the accept path retries its ioctl call up to 100 times, but only when the
status is `HAILO_DEVICE_TEMPORARILY_UNAVAILABLE`:

```cpp
for (size_t i = 0; i < MAX_CONNECT_RETRIES; i++) {
    status = RUN_IOCTL(HAILO_PCI_EP_ACCEPT, &params);
    if (HAILO_DEVICE_TEMPORARILY_UNAVAILABLE != status) {
        break;   // any other status, including HAILO_DRIVER_INTERRUPTED, exits immediately
    }
    std::this_thread::sleep_for(RETRY_INTERVAL);
}
CHECK_SUCCESS(status, "Failed pci_ep accept");
```

A POSIX signal arriving while the kernel driver blocks on the accept
ioctl makes that syscall return `EINTR`, which the host's errno-mapping
converts to `HAILO_DRIVER_INTERRUPTED` — a *different* status than
`HAILO_DEVICE_TEMPORARILY_UNAVAILABLE`, so the retry loop's condition is
false and it breaks immediately instead of retrying. The failure then
propagates to Python as `HAILO_COMMUNICATION_CLOSED`. This is a real gap
in the public retry loop's status check, not a hardware fault — which is
exactly why a fresh process (a fresh, non-interrupted syscall) succeeds
every time.

## NumPy ≥ 2 promotion semantics break quantization

The quantization stage relies on float32/float64 arrays staying distinct;
NumPy 2's default promotion merges them and `create_hw_params()` fails.
`NPY_PROMOTION_STATE=legacy` must be set **before numpy is imported**
([docker images bake it in](../../docker/README.md); config.py also sets
it defensively).

## GPU selection

If `CUDA_VISIBLE_DEVICES` is unset at run time, the toolchain sets it to
`"99"` internally — hiding every GPU from PyTorch/TensorFlow. Always pass
it explicitly (`-e CUDA_VISIBLE_DEVICES=0`); on ROCm builds the same
variable name applies.

## Opaque client-side errors (timeouts, generic failure codes)

When `genai`/HailoRT reports a generic client-side error (a timeout, a
vague failure code) that doesn't explain itself: the device runs its own
firmware with a mini-Linux, and its own logs (`hailortcli logs
runtime/system_control/nnc`) frequently contain the specific, immediately
actionable error that the client-side log never surfaces. Check there
before hypothesizing from client-visible symptoms alone — see the
[`device-debug`](../../.claude/skills/device-debug/SKILL.md) skill for the
full playbook.

## `Memory units capacity exceeded` is a fixed physical limit, not a tunable

On a larger GQA model (30 layers), HAR→HEF compilation hit `Memory units
capacity exceeded (available: 128, required: 138)` on a single node,
identically across every compiler parameter tried (`max_memory_utilization`
at several levels, `optimization_level`, alternate partitioner
configurations). The `128` figure matches `CLUSTER_UNITS::MEMORY` in the
SDK's own (plain, non-obfuscated) hardware-constants file for this chip
family — confirming it's a fixed per-cluster memory budget on silicon, not
a compiler setting. No model-script parameter changes it.

**What actually worked**: removing the base (non-duplicated) network group
from the compile entirely — keeping only `__prefill`/`__tbt` — let the
multi-context partitioner find a working split where it previously
couldn't. If you hit this exact error, stop tuning compiler flags and
check whether a base-scope group is present in your compile.

**Untested lever**: `llm_modifications` — a model-script directive
referenced in official recipes but never otherwise documented in this
project — is policy-gated (off by default) rather than chip-gated. One of
its sub-passes, `_handle_tiling_conv`, is designed to detect the
GQA-repeat-via-matmul pattern (a conv whose kernel is concatenated
identity matrices) and rewrite it as a native `input_tiles` parameter on
the matmul instead of a real conv — which could reduce the memory
footprint that trips this wall. Never enabled in this project's pipeline.

## Scope collapse in some optimizer backends

Certain optimizer backends collapse the duplicated `__prefill`/`__tbt`
scopes onto shared base weights during optimization. If you observe
scope-dependent layers mysteriously sharing tensors after optimize,
re-check with the recipe from [quantization-recipe.md](quantization-recipe.md)
before blaming your graph.
