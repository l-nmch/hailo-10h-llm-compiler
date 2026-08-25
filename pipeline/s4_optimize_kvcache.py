#!/usr/bin/env python3
"""Step 4 — quantization with KV-cache duplication (HAR → quantized HAR).

Applies the project's validated quantization recipe and lets
``set_kv_cache_global_params`` duplicate the graph into the ``__prefill``
and ``__tbt`` scopes that the genai runtime drives.

The recipe was derived by direct comparison with Hailo's official Qwen2-1.5B
`.alls` recipe and validated on hardware:

- ``pre_quantization_optimization(ew_add_fusing, policy=disabled)``
  The official recipe disables this; leaving it at its default silently
  fuses residual adds and misaligns conv inputs in the duplicated scopes.
  This single line was the root cause of both the historical conv
  misalignments AND wrong argmax outputs.
- ``bias_correction`` **enabled**, with ``use_saitama=True, device=cuda``
  set directly on its own directive (not just the global calibration one —
  otherwise it silently falls back to CPU). This is the project's de-facto
  standard as of the GPU/saitama quantization path: bias_correction alone
  measurably improves cosine. ``adaround`` combined with it is a mild net
  negative (not broken, just not worth it); ``finetune`` (QAT) combined
  with it measured catastrophic — not a bug in finetune itself, a real
  measured incompatibility between the two (see
  docs/findings/quantization-recipe.md's ablation table). Both stay
  explicitly disabled here.
- NO ``weight_group_size`` — incompatible with bias_correction+saitama
  (SDK bug: `FusedQWGModule` has no `mac` attribute in
  `bias_accumulator.py`, doesn't handle grouped-weight convs).
- ``model_optimization_flavor(compression_level=4, optimization_level=0)``
  optimization_level=0 matters regardless: any higher level implicitly
  re-enables ``adaround``/``finetune`` too, which we want to stay off.
- ``quantization_param(input_layer1, precision_mode=a16_w16)``
  Embeddings stay 16-bit (they are read host-side as uint16 codes).
- convs at ``a8_w4`` (INT8 activations / INT4 weights), excluding sparse or
  ultra-narrow convs.
- Calibration feeds RAW INTEGER POSITIONS to the RoPE inputs — not
  precomputed cos/sin. DFC's conversion_type mechanism computes cos/sin
  itself for calibration; that software emulation never runs on-chip.

The SDK_QUANTIZED emulator is structurally broken on KV-cache graphs (see
docs/findings/sdk-behavior-notes.md), so no cosine is available after this
step — hardware is the only judge until step 6's compile.

Usage:
    python s4_optimize_kvcache.py
"""
import argparse

import config  # must precede numpy imports — sets NPY_PROMOTION_STATE et al.


# Generic-domain default calibration pool: deliberately varied in topic,
# sentence length, and register (narrative, technical, dialogue,
# instruction, question) rather than matched to any one model's training
# domain. Calibration quality depends on covering a broad activation
# range, not on topical similarity to the target model — a model trained
# on a narrow domain (e.g. children's stories) may still calibrate fine
# on generic text, and a narrow/repetitive pool undercalibrates a
# general-purpose model regardless of topic match. Override with
# --calib-text-file for a checkpoint where domain genuinely matters (e.g.
# code, medical, a specific language register).
SENTENCE_POOL = [
    "Once upon a time there was a little girl named Lily who loved to play in the garden every day.",
    "The quarterly report showed a marginal increase in revenue despite rising operational costs.",
    "Can you explain how photosynthesis converts sunlight into chemical energy within plant cells?",
    "Turn left at the second intersection, then continue straight for about three hundred meters.",
    "The old wizard walked slowly through the forest, looking for herbs to make his special potion.",
    "\"I don't think that's a good idea,\" she said, crossing her arms and shaking her head.",
    "Researchers at the university published a study linking sleep quality to long-term memory retention.",
    "First, preheat the oven to 200 degrees, then whisk the eggs and sugar until pale and fluffy.",
    "The stock market fluctuated wildly after the central bank announced an unexpected rate change.",
    "A kind old man lived in a small cottage at the edge of the village near the river.",
    "What time does the next train to the city center leave, and how much does a ticket cost?",
    "The algorithm sorts the array in place, achieving O(n log n) time complexity in the average case.",
]


def discover_compressible_convs(runner):
    """Select conv layers for INT4: exclude near-empty kernels and convs whose
    narrowest input is at most one head wide (RoPE-scale helpers)."""
    import numpy as np

    hn = runner.get_hn()
    params = runner.get_params()

    def sparsity(name):
        inner = params.get(name)
        if inner is None:
            return 0.0
        kernel = inner.get("kernel:0")
        if kernel is None:
            return 0.0
        return float(np.mean(np.asarray(kernel) == 0))

    def min_in_width(layer):
        shapes = layer.get("input_shapes") or []
        return min((s[-1] for s in shapes), default=10**9)

    all_convs = [(n, l) for n, l in hn["layers"].items() if l.get("type") == "conv"]
    keep, excluded = [], []
    for name, layer in all_convs:
        if sparsity(name) > 0.9 or min_in_width(layer) <= config.HD:
            excluded.append(name)
        else:
            keep.append(name)
    print(f"  {len(all_convs)} convs found -> {len(keep)} INT4, {len(excluded)} excluded")
    return sorted(keep)


def load_sentence_pool(path) -> list:
    """One calibration sentence per non-empty, non-comment line."""
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    assert lines, f"{path} has no usable calibration sentences"
    return lines


def build_calibration(tokenizer, wte, pad_id, sentence_pool):
    """calibset_size samples: permuted sentence concatenations padded to SEQ."""
    import numpy as np

    rng = np.random.default_rng(0)
    token_rows = []
    for _ in range(config.CALIBSET_SIZE):
        order = rng.permutation(len(sentence_pool))
        ids = []
        for idx in order:
            ids.extend(tokenizer(sentence_pool[idx])["input_ids"])
            if len(ids) >= config.SEQ:
                break
        ids = (ids + [pad_id] * config.SEQ)[: config.SEQ]
        token_rows.append(np.array(ids, dtype=np.int64))
    calib_token_ids = np.stack(token_rows, axis=0)

    calib_embeds = wte[calib_token_ids][:, np.newaxis, :, :].astype(np.float32)
    mask = config.causal_mask_tiled(config.CALIBSET_SIZE, config.SEQ)
    # Raw positions for RoPE inputs — DFC derives cos/sin internally.
    raw_positions = np.tile(
        np.arange(config.SEQ, dtype=np.float32)[np.newaxis, :], (config.CALIBSET_SIZE, 1)
    ).astype(np.float32)

    scope = config.NET_SCOPE
    return {
        f"{scope}/input_layer1": calib_embeds,
        f"{scope}/input_layer2": mask,
        f"{scope}/input_layer3": raw_positions,
        f"{scope}/input_layer4": raw_positions,
        f"{scope}/input_layer5": raw_positions,
        f"{scope}/input_layer6": raw_positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    parser.add_argument("--calib-text-file", default=None,
                        help="one calibration sentence per line, overriding "
                             "the built-in generic-domain pool — use this "
                             "when the target checkpoint's domain genuinely "
                             "differs from general English (code, medical, "
                             "another language, etc.)")
    parser.add_argument("--calibset-size", type=int, default=None,
                        help="override config.CALIBSET_SIZE for this run only "
                             "(default: whatever step 1 resolved, usually 32)")
    parser.add_argument("--bias-correction", dest="bias_correction", action="store_true", default=True,
                        help="enable bias_correction on saitama/GPU (default: "
                             "on — this project's validated standard; measurably "
                             "improves cosine alone, see quantization-recipe.md)")
    parser.add_argument("--no-bias-correction", dest="bias_correction", action="store_false",
                        help="disable bias_correction (matches the official "
                             "Qwen2-1.5B recipe this project started from)")
    parser.add_argument("--adaround", action="store_true", default=False,
                        help="enable adaround (default: off — a mild net "
                             "negative combined with bias_correction in our "
                             "measurements, not recommended, see "
                             "quantization-recipe.md's ablation table)")
    parser.add_argument("--finetune", action="store_true", default=False,
                        help="enable finetune/QAT (default: off — measured "
                             "CATASTROPHIC combined with bias_correction, "
                             "cosine -0.72; only enable this if you are "
                             "specifically re-investigating that finding)")
    parser.add_argument("--layer-noise-analysis", action="store_true", default=False,
                        help="run the Layer Noise Analysis checker (default: "
                             "off — read-only diagnostic, never changes the "
                             ".Q.HAR; see docs/status.md 'what would move the "
                             "needle next')")
    parser.add_argument("--conv-precision", default="a8_w4",
                        help="precision_mode for compressible convs (default: "
                             "a8_w4 — INT8 activations / INT4 weights)")
    parser.add_argument("--compression-level", type=int, default=4,
                        help="model_optimization_flavor compression_level (default: 4)")
    parser.add_argument("--optimization-level", type=int, default=0,
                        help="model_optimization_flavor optimization_level "
                             "(default: 0 — REQUIRED to keep adaround/finetune "
                             "off by default; any higher value silently "
                             "re-enables both regardless of the flags above, "
                             "see sdk-behavior-notes.md)")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    config.load()  # picks up run_config.json written by step 1, if any
    if args.calibset_size is not None:
        config.CALIBSET_SIZE = args.calibset_size
    sentence_pool = (
        load_sentence_pool(args.calib_text_file) if args.calib_text_file else SENTENCE_POOL
    )
    print(f"==> calibration pool: {len(sentence_pool)} sentences "
          f"({'from ' + args.calib_text_file if args.calib_text_file else 'built-in generic default'})")
    print(f"==> recipe: bias_correction={args.bias_correction} adaround={args.adaround} "
          f"finetune={args.finetune} layer_noise_analysis={args.layer_noise_analysis} "
          f"conv_precision={args.conv_precision} compression_level={args.compression_level} "
          f"optimization_level={args.optimization_level} calibset_size={config.CALIBSET_SIZE}")
    if args.optimization_level > 0 and not (args.adaround or args.finetune):
        print("!! optimization_level>0 silently re-enables adaround/finetune "
              "regardless of --adaround/--finetune — see sdk-behavior-notes.md !!")
    if args.layer_noise_analysis:
        print("!! --layer-noise-analysis uses an UNVERIFIED directive name/syntax "
              "(never exercised in this project before — see docs/status.md) — "
              "if this crashes, that's why; check the SDK's actual API before filing a bug !!")
    P = config.paths()

    n_registered = config.register_acceleras_layers()
    print(f"registered {n_registered} acceleras/Keras layer classes")

    import numpy as np
    from transformers import AutoTokenizer
    from hailo_sdk_client import ClientRunner

    tokenizer = AutoTokenizer.from_pretrained(str(P.tokenizer_dir))
    wte = np.load(P.wte)
    pad_id = config.PAD_TOKEN_ID

    print("==> discovering compressible convs")
    runner = ClientRunner(har=str(P.har_resources))
    conv_names = discover_compressible_convs(runner)

    scope = config.NET_SCOPE
    conv_list_str = ", ".join(conv_names)

    def _policy(flag: bool, use_saitama: bool = False) -> str:
        if not flag:
            return "policy=disabled"
        return "policy=enabled, use_saitama=True, device=cuda" if use_saitama else "policy=enabled"

    # layer_noise_analysis's directive name is unverified in this SDK version
    # (confirmed wrong once: "'layer_noise_analysis' is not a valid
    # PostQuantizationFeature") — only emit the line at all when explicitly
    # requested, so a bad/guessed directive name can't break every run.
    layer_noise_line = (
        f"post_quantization_optimization(layer_noise_analysis, {_policy(True)})"
        if args.layer_noise_analysis else ""
    )
    model_script = f"""
pre_quantization_optimization(ew_add_fusing, policy=disabled)
set_kv_cache_global_params({config.PREFILL_SIZE}, {config.CACHE_SIZE})
model_optimization_config(globals, multiproc_policy=disabled)
model_optimization_config(calibration, batch_size=1, calibset_size={config.CALIBSET_SIZE}, use_saitama=True, device=cuda)
model_optimization_flavor(compression_level={args.compression_level}, optimization_level={args.optimization_level})
post_quantization_optimization(bias_correction, {_policy(args.bias_correction, use_saitama=True)})
post_quantization_optimization(adaround, {_policy(args.adaround)})
post_quantization_optimization(finetune, {_policy(args.finetune)})
{layer_noise_line}
quantization_param([{scope}/input_layer1], precision_mode=a16_w16)
quantization_param([{conv_list_str}], precision_mode={args.conv_precision})
"""
    print("=== model script ===")
    print("\n".join(model_script.strip().splitlines()[:6]) + "\n    ... conv list omitted ...")

    print("==> building calibration set")
    calib_data = build_calibration(tokenizer, wte, pad_id, sentence_pool)
    print({k: v.shape for k, v in sorted(calib_data.items())})

    print("==> optimizing (GPU expected; ~30s on a modern GPU)")
    runner.load_model_script(model_script)
    runner.optimize(calib_data)
    runner.save_har(str(P.har_quantized))
    print(f"quantized HAR saved -> {P.har_quantized}")
    print("[OK] step 4 complete (no cosine available here — see module docstring)")


if __name__ == "__main__":
    main()
