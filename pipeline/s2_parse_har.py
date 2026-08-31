#!/usr/bin/env python3
"""Step 2 — ONNX → native (float32) HAR.

Translates the ONNX graph with the DFC parser targeting Hailo-10H and
validates it in ``SDK_NATIVE`` context against the float32 reference saved
by step 1.

Input declarations matter for the runtime contract:

- ``inputs_embeds``  [1, SEQ, HIDDEN], NHWC-style (BATCH/WIDTH/CHANNELS)
- ``attention_mask`` [1, 1, SEQ, NHEAD*SEQ] (BATCH/HEIGHT/WIDTH/CHANNELS) —
  already head-tiled host-side, matching what the genai runtime writes
- RoPE inputs        [1, SEQ, HD] each, untiled at parse time; step 3
  rewires them to their final asymmetric widths

At this stage the graph is still mathematically identical to the model —
the runtime-facing quirks that step 3 fixes do not affect DFC's software
simulation, only real on-chip behavior. The validation here is a sanity
checkpoint before surgery.

Usage:
    python s2_parse_har.py
"""
import argparse

import config  # must precede numpy imports — sets NPY_PROMOTION_STATE et al.

import numpy as np
from hailo_sdk_client import ClientRunner
from hailo_sdk_client.exposed_definitions import Dims, DistributionStrategy, InferenceContext


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    config.load()  # picks up run_config.json written by step 1, if any
    if config.COSINE_MIN < 0.999:
        print(f"!! COSINE_MIN overridden to {config.COSINE_MIN} (validated default: 0.999) !!")
    P = config.paths()

    refs = np.load(P.hf_refs)
    token_embeds = refs["token_embeds"]
    hf_logits_last = refs["hf_logits_last"]

    print(f"==> parsing {P.onnx} -> HAR ({config.NET_SCOPE})")
    runner = ClientRunner(hw_arch="hailo10h")
    runner.translate_onnx_model(
        str(P.onnx),
        config.NET_SCOPE,
        disable_onnx_simplifier=True,
        net_input_shapes={
            "inputs_embeds": [1, config.SEQ, config.HIDDEN],
            "attention_mask": [1, 1, config.SEQ, config.NHEAD * config.SEQ],
            "pe_k_cos": [1, config.SEQ, config.HD],
            "pe_q_cos": [1, config.SEQ, config.HD],
            "pe_k_sin": [1, config.SEQ, config.HD],
            "pe_q_sin": [1, config.SEQ, config.HD],
        },
        net_input_format={
            "inputs_embeds": [Dims.BATCH, Dims.WIDTH, Dims.CHANNELS],
            "attention_mask": [Dims.BATCH, Dims.HEIGHT, Dims.WIDTH, Dims.CHANNELS],
            "pe_k_cos": [Dims.BATCH, Dims.WIDTH, Dims.CHANNELS],
            "pe_q_cos": [Dims.BATCH, Dims.WIDTH, Dims.CHANNELS],
            "pe_k_sin": [Dims.BATCH, Dims.WIDTH, Dims.CHANNELS],
            "pe_q_sin": [Dims.BATCH, Dims.WIDTH, Dims.CHANNELS],
        },
    )
    runner.save_har(str(P.har_parsed))
    print(f"[OK] parsed -> {P.har_parsed}")

    # --- Sanity check: SDK_NATIVE inference on the untouched graph ---
    scope = config.NET_SCOPE
    theta = config.head_dim_frequencies()
    angles = np.outer(np.arange(config.SEQ), theta.astype(np.float64))
    cos_full = np.cos(angles).astype(np.float32)[np.newaxis]
    sin_full = np.sin(angles).astype(np.float32)[np.newaxis]

    calib = {
        f"{scope}/input_layer1": token_embeds[:, np.newaxis, :, :].astype(np.float32),
        f"{scope}/input_layer2": config.causal_mask_tiled(1, config.SEQ),
        f"{scope}/input_layer3": cos_full[:, np.newaxis, :, :],
        f"{scope}/input_layer4": cos_full[:, np.newaxis, :, :],
        f"{scope}/input_layer5": sin_full[:, np.newaxis, :, :],
        f"{scope}/input_layer6": sin_full[:, np.newaxis, :, :],
    }
    with runner.infer_context(InferenceContext.SDK_NATIVE, gpu_policy=DistributionStrategy.SINGLE) as ctx:
        out = runner.infer(ctx, dataset=calib, data_type="np_array", batch_size=1)
    # Multiple outputs when config.LM_HEAD_SHARDS > 1 (large-vocab lm_head
    # sharding — see docs/findings/large-vocab-lm-head-sharding.md);
    # concatenate in declaration order to reconstruct the full logits.
    shards = out if isinstance(out, list) else [out]
    native_logits = np.concatenate([np.array(s).reshape(1, 1, -1) for s in shards], axis=-1)
    sim = config.cosine(hf_logits_last, native_logits)
    print(f"cosine(HF last position, HAR/SDK_NATIVE, pre-surgery): {sim:.6f}")
    assert sim > config.COSINE_MIN, "native HAR diverged from HF"
    print("[OK] step 2 complete")


if __name__ == "__main__":
    main()
