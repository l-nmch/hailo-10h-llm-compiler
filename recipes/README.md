# Quantization recipes

Step 4 loads one of these with `python s4_optimize_kvcache.py --recipe ../recipes/<name>.alls`
(`--print-recipe` prints the resolved script and exits without quantizing).
Without `--recipe`, step 4 applies its built-in script, identical to `default.alls`
when no recipe flag is passed.

| File | What it is |
|---|---|
| `default.alls` | This pipeline's default: everything a8 except the embedding input, `bias_correction`, `compression_level=4`. |
| `hailo-llm.alls` | Hailo's Qwen2 LLM recipe with role placeholders, so it applies to any checkpoint. |
| `hailo-qwen2-1.5b-instruct.alls` | Hailo's original file, unmodified (reference; names only fit Hailo's own Qwen2 graph). |

A recipe is a DFC model script (`.alls`). It may contain placeholders, which step 4
replaces before loading it. Other braces (DFC globs such as `{*}`) are left alone. A line whose
layer-list placeholder matches no layer in the graph is dropped with a warning (for
example `weight_group_size=128` on a model whose conv input widths are not multiples
of 128).

| Placeholder | Resolves to |
|---|---|
| `{scope}` | network scope of the run |
| `{prefill_size}`, `{cache_size}`, `{calibset_size}` | values of the run |
| `{norm_groups}` | largest divisor of `hidden` that is at most 16 |
| `{compressible_convs}` | the convs the default recipe quantizes to INT4 (lm_head included) |
| `{block_convs}` | the same without the lm_head |
| `{block_convs_g128}` | block convs whose input width is a multiple of 128 |
| `{lm_head}` | lm_head conv(s); `<conv>_d<i>` when the recipe enables `llm_modifications` |
| `{final_slice}` | last-position slice; `moved_<slice>` when the recipe enables `llm_modifications` |
| `{residual_adds}` | `ew_add` layers feeding a `layer_normalization` |
| `{swiglu_mults}` | `ew_mult` layers feeding a conv (SwiGLU product) |
| `{down_proj}` | convs fed by a SwiGLU product |
| `{up_proj}` | linear convs feeding a SwiGLU product (the gate conv carries the silu) |
| `{o_proj}` | convs fed by a matmul |
| `{qk_matmuls}` | matmuls not fed by a softmax (QKᵀ) |

Step 4 always overrides `set_kv_cache_global_params` with the run's prefill and cache
sizes, and calibrates with a multiplicative 1/0 mask when the recipe contains
`set_input_mask_to_softmax()` (that command turns the additive mask into a
multiplicative one inside the softmax; the uint8 255/0 wire format is unchanged).

`llm_modifications` splits only the conv feeding the first output, so a recipe that
enables it needs a single lm_head conv: it refuses to resolve on a run whose step 1
sharded the lm_head.

Measured on TinyStories-25M on the chip (Pearson vs HF, HF top-1 over prefill + 15
decode steps): `default.alls` 0.982, 13/16; `hailo-llm.alls` 0.936, 7/16;
`hailo-llm.alls` without its four advanced features (fused softmax mask, `quarot`,
norm decomposition, `smart_softmax_stats`) 0.985, 15/16. Adding those back one at a
time, only `set_input_mask_to_softmax()` costs accuracy (0.945, 9/16); the other
three are neutral on their own.
