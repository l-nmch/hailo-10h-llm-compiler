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
- NO ``bias_correction``, NO ``adaround``, NO ``weight_group_size``
  The official LLM recipes use none of them; ``weight_group_size`` is also
  incompatible with the PyTorch optimizer (SDK bug in bias_accumulator).
- ``model_optimization_flavor(compression_level=4, optimization_level=0)``
  optimization_level=0 matters: any higher level implicitly re-enables
  adaround/bias_correction/finetune.
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


# Calibration sentences drawn from the TinyStories domain of MODEL_ID.
SENTENCE_POOL = [
    "Once upon a time there was a little girl named Lily who loved to play in the garden every day.",
    "The old wizard walked slowly through the forest, looking for herbs to make his special potion.",
    "Tom and his dog Max went for a walk in the park and found a shiny red ball under a tree.",
    "The little mouse was scared of the big cat, so it hid inside a small hole in the wall.",
    "Every morning the farmer fed his chickens and collected fresh eggs for breakfast.",
    "The children built a sandcastle on the beach and watched the waves crash against the shore.",
    "A kind old man lived in a small cottage at the edge of the village near the river.",
    "The bright yellow sun rose over the mountains as the birds began to sing their morning song.",
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


def build_calibration(tokenizer, wte, pad_id):
    """calibset_size samples: permuted sentence concatenations padded to SEQ."""
    import numpy as np

    rng = np.random.default_rng(0)
    token_rows = []
    for _ in range(config.CALIBSET_SIZE):
        order = rng.permutation(len(SENTENCE_POOL))
        ids = []
        for idx in order:
            ids.extend(tokenizer(SENTENCE_POOL[idx])["input_ids"])
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
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
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
    model_script = f"""
pre_quantization_optimization(ew_add_fusing, policy=disabled)
set_kv_cache_global_params({config.PREFILL_SIZE}, {config.CACHE_SIZE})
model_optimization_config(globals, multiproc_policy=disabled)
model_optimization_config(calibration, batch_size=1, calibset_size={config.CALIBSET_SIZE}, use_saitama=True, device=cuda)
model_optimization_flavor(compression_level=4, optimization_level=0)
quantization_param([{scope}/input_layer1], precision_mode=a16_w16)
quantization_param([{conv_list_str}], precision_mode=a8_w4)
"""
    print("=== model script ===")
    print("\n".join(model_script.strip().splitlines()[:6]) + "\n    ... conv list omitted ...")

    print("==> building calibration set")
    calib_data = build_calibration(tokenizer, wte, pad_id)
    print({k: v.shape for k, v in sorted(calib_data.items())})

    print("==> optimizing (GPU expected; ~30s on a modern GPU)")
    runner.load_model_script(model_script)
    runner.optimize(calib_data)
    runner.save_har(str(P.har_quantized))
    print(f"quantized HAR saved -> {P.har_quantized}")
    print("[OK] step 4 complete (no cosine available here — see module docstring)")


if __name__ == "__main__":
    main()
