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

## Scope collapse in some optimizer backends

Certain optimizer backends collapse the duplicated `__prefill`/`__tbt`
scopes onto shared base weights during optimization. If you observe
scope-dependent layers mysteriously sharing tensors after optimize,
re-check with the recipe from [quantization-recipe.md](quantization-recipe.md)
before blaming your graph.
