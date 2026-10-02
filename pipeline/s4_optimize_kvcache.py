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
  otherwise it silently falls back to CPU). It stays the default, but an
  on-chip A/B on TinyStories-25M measured no gain from it, and it is the
  stage that runs out of GPU memory first on larger models
  (``--no-bias-correction`` drops it; see quantization-recipe.md's
  on-device A/B). ``adaround`` combined with it is a mild net
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
    python s4_optimize_kvcache.py [--recipe ../recipes/<name>.alls] [--print-recipe]

``--recipe`` loads a quantization recipe from a ``.alls`` file instead of the
script above (recipes/README.md lists the shipped recipes and the
placeholders step 4 resolves against the graph).
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


def find_roles(runner, compressible) -> dict:
    """Layer lists by role in the pre-quantization graph, for recipe placeholders (see recipes/README.md)."""
    layers = runner.get_hn_dict()["layers"]
    succ = lambda n: [o for o in layers[n].get("output", []) if o in layers]
    pred = lambda n: [i for i in layers[n].get("input", []) if i in layers]
    typ = lambda n: layers[n]["type"]

    swiglu = sorted(n for n in layers if typ(n) == "ew_mult" and any(typ(o) == "conv" for o in succ(n)))
    lm_head = sorted(n for n in compressible if any(typ(o) == "output_layer" for o in succ(n)))
    block = [n for n in compressible if n not in lm_head]
    hidden = layers[f"{config.NET_SCOPE}/input_layer1"]["output_shapes"][0][-1]
    return {
        "compressible_convs": list(compressible),
        "block_convs": block,
        "block_convs_g128": [n for n in block if layers[n]["input_shapes"][0][-1] % 128 == 0],
        "lm_head": lm_head,
        "final_slice": sorted(n for n in layers if typ(n) == "slice"),
        "residual_adds": sorted(n for n in layers if typ(n) == "ew_add"
                                and any(typ(o) == "layer_normalization" for o in succ(n))),
        "swiglu_mults": swiglu,
        "down_proj": sorted({o for m in swiglu for o in succ(m) if typ(o) == "conv"}),
        "up_proj": sorted({i for m in swiglu for i in pred(m)
                           if typ(i) == "conv" and layers[i].get("params", {}).get("activation") == "linear"}),
        "o_proj": sorted(n for n in layers if typ(n) == "conv" and any(typ(i) == "matmul" for i in pred(n))),
        "qk_matmuls": sorted(n for n in layers if typ(n) == "matmul" and not any(typ(i) == "softmax" for i in pred(n))),
        "norm_groups": max(g for g in range(1, 17) if hidden % g == 0),
    }


def resolve_recipe(text: str, roles: dict) -> tuple:
    """Turn a recipe .alls into the model script to load, plus whether the mask gets fused into the softmax.

    Replaces the placeholders listed in recipes/README.md (other braces, e.g. DFC globs like {*}, are kept),
    forces set_kv_cache_global_params to the run's sizes, and follows the renames llm_modifications applies.
    """
    import re

    active = "\n".join(ln.split("#", 1)[0] for ln in text.splitlines())
    llm = re.search(r"pre_quantization_optimization\(\s*llm_modifications\b[^)]*policy\s*=\s*enabled", active)
    roles = dict(roles)
    if llm:
        # llm_modifications (hailo_sdk_client/post_fuser/algorithms/llm_modifications.py) moves the final slice
        # before the last norm as moved_<slice>, and splits the conv feeding the first output into conv_splits
        # convs <conv>_d<i> (default 4) -- precision lines have to name what exists after that rewrite.
        m = re.search(r"llm_modifications\b[^)]*conv_splits\s*=\s*(\d+)", active)
        splits = int(m.group(1)) if m else 4
        roles["final_slice"] = [f"{n.split('/', 1)[0]}/moved_{n.split('/', 1)[1]}" for n in roles["final_slice"]]
        if splits > 1:
            if len(roles["lm_head"]) != 1:
                raise SystemExit(f"recipe enables llm_modifications, which splits a single lm_head conv, but step 1 "
                                 f"sharded the lm_head into {len(roles['lm_head'])} convs ({roles['lm_head']})")
            roles["lm_head"] = [f"{roles['lm_head'][0]}_d{i}" for i in range(splits)]
    values = {
        "scope": config.NET_SCOPE,
        "prefill_size": config.PREFILL_SIZE,
        "cache_size": config.CACHE_SIZE,
        "calibset_size": config.CALIBSET_SIZE,
        **{k: (", ".join(v) if isinstance(v, list) else v) for k, v in roles.items()},
    }
    empty = {k for k, v in roles.items() if isinstance(v, list) and not v}
    fill = lambda code: re.sub(r"\{(\w+)\}", lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), code)
    lines = []
    for ln in text.splitlines():
        code, sep, comment = ln.partition("#")
        unresolved = sorted(set(re.findall(r"\{(\w+)\}", code)) & empty)
        if unresolved:
            # A role absent from this graph (e.g. no conv input width divisible by 128 for {block_convs_g128}):
            # the line has nothing to apply to.
            print(f"!! recipe line dropped, {unresolved} match no layer in this graph: {code.strip()[:100]}")
            continue
        lines.append(fill(code) + sep + comment)
    script = "\n".join(lines)
    kv = f"set_kv_cache_global_params({config.PREFILL_SIZE}, {config.CACHE_SIZE})"
    if "set_kv_cache_global_params" in active:
        script = re.sub(r"set_kv_cache_global_params\([^)]*\)", kv, script)
    else:
        script = kv + "\n" + script
    return script, "set_input_mask_to_softmax" in active


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
    parser.add_argument("--recipe", default=None,
                        help="quantization recipe (.alls, see recipes/README.md) replacing the built-in script; "
                             "the recipe flags below are then ignored")
    parser.add_argument("--print-recipe", action="store_true",
                        help="print the resolved model script and exit without quantizing")
    parser.add_argument("--calib-text-file", default=None,
                        help="one calibration sentence per line, overriding "
                             "the built-in generic-domain pool — use this "
                             "when the target checkpoint's domain genuinely "
                             "differs from general English (code, medical, "
                             "another language, etc.)")
    parser.add_argument("--calibset-size", type=int, default=None,
                        help="override config.CALIBSET_SIZE for this run only "
                             "(default: whatever step 1 resolved, usually 32)")
    parser.add_argument("--calib-batch-size", type=int, default=1,
                        help="calibration inference batch size (default 1, the minimum: "
                             "the lowest GPU memory use; raise it only to speed up "
                             "calibration on a GPU with spare memory)")
    parser.add_argument("--no-saitama", action="store_true", default=False,
                        help="run calibration and bias_correction on DFC's TensorFlow engine instead of "
                             "saitama (PyTorch); avoids TF/PyTorch VRAM contention on large models")
    parser.add_argument("--bias-correction", dest="bias_correction", action="store_true", default=True,
                        help="enable bias_correction on saitama/GPU (default: on; an "
                             "on-chip A/B on TinyStories-25M measured no gain from it, "
                             "see quantization-recipe.md)")
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
          f"optimization_level={args.optimization_level} calibset_size={config.CALIBSET_SIZE} "
          f"calib_batch_size={args.calib_batch_size}")
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
    # saitama = DFC's PyTorch optimization engine. With it, TensorFlow and PyTorch share the GPU in one process and
    # TensorFlow keeps the VRAM it grew into, which can starve PyTorch on large models; without it everything stays
    # in TensorFlow (also on GPU on NVIDIA). See docs/findings/quantization-recipe.md.
    _no_saitama = args.no_saitama
    _saitama_cal = "" if _no_saitama else ", use_saitama=True, device=cuda"
    model_script = f"""
pre_quantization_optimization(ew_add_fusing, policy=disabled)
set_kv_cache_global_params({config.PREFILL_SIZE}, {config.CACHE_SIZE})
model_optimization_config(globals, multiproc_policy=disabled)
model_optimization_config(calibration, batch_size={args.calib_batch_size}, calibset_size={config.CALIBSET_SIZE}{_saitama_cal})
model_optimization_flavor(compression_level={args.compression_level}, optimization_level={args.optimization_level})
post_quantization_optimization(bias_correction, {_policy(args.bias_correction, use_saitama=not _no_saitama)})
post_quantization_optimization(adaround, {_policy(args.adaround)})
post_quantization_optimization(finetune, {_policy(args.finetune)})
{layer_noise_line}
quantization_param([{scope}/input_layer1], precision_mode=a16_w16)
quantization_param([{conv_list_str}], precision_mode={args.conv_precision})
"""
    fused_mask = False
    if args.recipe:
        with open(args.recipe) as f:
            model_script, fused_mask = resolve_recipe(f.read(), find_roles(runner, conv_names))
        print(f"==> recipe {args.recipe} (recipe flags ignored)")
    if args.print_recipe:
        print(model_script.strip())
        return
    print("=== model script ===")
    print("\n".join(model_script.strip().splitlines()[:6]) + "\n    ... conv list omitted ...")

    print("==> building calibration set")
    calib_data = build_calibration(tokenizer, wte, pad_id, sentence_pool)
    if fused_mask:
        # set_input_mask_to_softmax() turns the additive mask into a multiplicative one inside the softmax
        # (hailo_sdk_client/post_fuser/algorithms/softmax_mapping.py multiplies the scores and the exp by it), so
        # it is calibrated as 1 (allowed) / 0 (blocked). The 0/-100 additive mask would multiply scores by -100 and
        # zero the softmax sums (NaN statistics). The uint8 255/0 wire format is unchanged.
        key = f"{config.NET_SCOPE}/input_layer2"
        calib_data[key] = (calib_data[key] == 0).astype(np.float32)
    print({k: v.shape for k, v in sorted(calib_data.items())})

    print("==> optimizing (GPU expected; ~30s on a modern GPU)")
    runner.load_model_script(model_script)
    runner.optimize(calib_data)
    runner.save_har(str(P.har_quantized))
    print(f"quantized HAR saved -> {P.har_quantized}")
    print("[OK] step 4 complete (no cosine available here — see module docstring)")


if __name__ == "__main__":
    main()
