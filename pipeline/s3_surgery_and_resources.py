#!/usr/bin/env python3
"""Step 3 — HAR graph surgery + genai external resources.

Two structural rewrites of the parsed HAR, both fixing mismatches between
what the graph declares and what the genai runtime actually writes into the
HEF inputs at run time (see docs/findings/ for the full evidence):

**RoPE surgery.** The parser declares `input_layer3-6` at uniform width HD,
but the runtime feeds asymmetrically tiled buffers: K = theta_size *
num_key_value_heads (=128 here), Q = theta_size * num_attention_heads
(=256). The parser compensates by inserting per-input duplication convs —
which tile a SECOND time on top of the host-side tiling. Fix: delete the
four convs, widen the input layers to their final widths and reconnect them
straight to their consumers.

**Attention-mask surgery.** The mask path went through a slice then an
element-wise add carrying `input_repeats`. DFC's `input_repeats` uses NumPy
*repeat* semantics (AABBCC), while broadcasting a head-tiled mask requires
*tile* semantics (ABCABC) — confirmed in DFC source
(`element_wise_add_op.py::repeat_inputs`) and empirically (cosine 0.951 ->
1.000000 after fix). The correct alternative (`input_tiles`) is rejected by
the PyTorch optimizer ("Input tiles must be trivial for MAC EWAdd"). Since
the host already tiles the mask identically across heads (as
genai::prepare_attention_mask_input does), the broadcast is unnecessary:
reconnect the adds directly to `input_layer2`.

After surgery the graph is re-validated in SDK_NATIVE context, then the
genai external resources are attached:

- on-chip embedding table (`embed`) bound to input_layer1
- RoPE theta table bound to the four cos/sin inputs (on-chip cos/sin
  computation via conversion_type)
- tokenizer.json and hailo-config.json as named external files

Usage:
    python s3_surgery_and_resources.py
"""
import argparse
import json
import os
import tarfile
import tempfile

import config  # must precede numpy imports — sets NPY_PROMOTION_STATE et al.

import numpy as np

# Surgery tables: (input layer, duplication conv to remove, final head count)
ROPE_LAYERS = None  # built in main(), depends on NET_SCOPE
MASK_EW_ADDS = ["ew_add3", "ew_add8", "ew_add13", "ew_add18"]  # one per layer


def load_hn_from_har(har_path, tmpdir):
    """Extract the HAR tarball and return the path + dict of its main .hn."""
    with tarfile.open(har_path) as t:
        t.extractall(tmpdir)
    hn_files = [
        f for f in os.listdir(tmpdir)
        if f.endswith(".hn") and not f.endswith((".fp.hn", ".native.hn"))
    ]
    assert len(hn_files) == 1, f"expected exactly one main .hn, got {hn_files}"
    hn_path = os.path.join(tmpdir, hn_files[0])
    with open(hn_path) as f:
        return hn_path, json.load(f)


def save_har_from_hn(tmpdir, har_out_path):
    with tarfile.open(har_out_path, "w") as t:
        for f in os.listdir(tmpdir):
            t.add(os.path.join(tmpdir, f), arcname=f)


def rope_surgery(layers: dict, scope: str) -> None:
    for input_name, conv_name, groups in ROPE_LAYERS:
        conv = layers[conv_name]
        consumers = list(conv["output"])
        new_width = conv["output_shapes"][0][-1]
        expected = config.HD * groups
        assert new_width == expected, f"{conv_name}: width {new_width} != {expected}"

        inp = layers[input_name]
        old_shape = inp["output_shapes"][0]
        new_shape = old_shape[:-1] + [new_width]
        inp["input_shapes"] = [new_shape]
        inp["output_shapes"] = [new_shape]
        inp["output"] = consumers
        for cname in consumers:
            c = layers[cname]
            c["input"] = [input_name if x == conv_name else x for x in c["input"]]
            c["input_shapes"] = [
                new_shape if x == input_name else s
                for x, s in zip(c["input"], c["input_shapes"])
            ]
        del layers[conv_name]
        print(f"  {input_name}: {old_shape} -> {new_shape}; removed {conv_name}")


def mask_surgery(layers: dict, scope: str) -> None:
    full_width = config.NHEAD * config.SEQ
    il2_name = f"{scope}/input_layer2"
    il2 = layers[il2_name]
    for i, ew_name in enumerate(MASK_EW_ADDS, start=1):
        slice_name = f"{scope}/slice{i}"
        ew_full = f"{scope}/{ew_name}"
        ew = layers[ew_full]
        # Reconnect directly to input_layer2 (already at full tiled width)
        # and neutralize any repeat/tile expansion params.
        ew["input"] = [il2_name if x == slice_name else x for x in ew["input"]]
        ew["input_shapes"] = [[-1, 1, config.SEQ, full_width]] * 2
        ew["params"]["input_repeats"] = [[1, 1, 1], [1, 1, 1]]
        ew["params"].pop("input_tiles", None)
        il2["output"] = [ew_full if x == slice_name else x for x in il2["output"]]
        del layers[slice_name]
        print(f"  {ew_name}: rewired to {il2_name}; removed {slice_name}")


def build_hailo_config() -> dict:
    """Generation-side configuration embedded into the HEF.

    Key naming is contractual: the LLM server reads
    ``pre_process_params.prefill_input_tokens_count`` (NOT
    ``..._size`` — an earlier draft used `_size`, which is silently ignored
    and falls back to a hardcoded default of 96).

    Note there is deliberately no "date" field: official Hailo configs carry
    a build-date string, but nothing in the public server source reads it.
    """
    return {
        "model_name": config.NET_SCOPE,
        "stop_token_id": [config.EOS_TOKEN_ID],
        "eos_token_id": config.EOS_TOKEN_ID,
        "default_generation_params": {
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        # Plain-text concatenation: TinyStories has no chat/role format.
        "chat_template": (
            "{% for message in messages %}"
            "{% for item in message['content'] %}"
            "{{ item['text'] }}"
            "{% endfor %}"
            "{% if not loop.last %}\n{% endif %}"
            "{% endfor %}"
        ),
        "pre_process_params": {
            "num_attention_heads": config.NHEAD,
            "num_key_value_heads": config.NKVHEAD,
            "kv_cache_size": config.CACHE_SIZE,
            "prefill_input_tokens_count": config.PREFILL_SIZE,
        },
        "input_layers_names_suffixes": {
            "embeddings": "input_layer1",
            "attention_mask": "input_layer2",
            "pe_k_cos": "input_layer3",
            "pe_q_cos": "input_layer4",
            "pe_k_sin": "input_layer5",
            "pe_q_sin": "input_layer6",
        },
    }


def attach_resources(runner, wte, theta_tiled) -> None:
    scope = config.NET_SCOPE
    runner.add_external_resources({
        "input_layers_mapping": {
            f"{scope}/input_layer1": "embedding",
            f"{scope}/input_layer3": "cos",
            f"{scope}/input_layer4": "cos",
            f"{scope}/input_layer5": "sin",
            f"{scope}/input_layer6": "sin",
        },
        "weights": {
            f"{scope}/input_layer1": {"embed": wte},
            # theta + per-group tile counts: the device computes cos/sin
            # itself from raw positions (conversion_type mechanism).
            f"{scope}/input_layer3": {
                "theta": theta_tiled, "tile": np.array([1, 1, config.NKVHEAD], dtype=np.int32),
                "factor": np.array([1.0], dtype=np.float32),
            },
            f"{scope}/input_layer4": {
                "theta": theta_tiled, "tile": np.array([1, 1, config.NHEAD], dtype=np.int32),
                "factor": np.array([1.0], dtype=np.float32),
            },
            f"{scope}/input_layer5": {
                "theta": theta_tiled, "tile": np.array([1, 1, config.NKVHEAD], dtype=np.int32),
                "factor": np.array([1.0], dtype=np.float32),
            },
            f"{scope}/input_layer6": {
                "theta": theta_tiled, "tile": np.array([1, 1, config.NHEAD], dtype=np.int32),
                "factor": np.array([1.0], dtype=np.float32),
            },
        },
    })


def main() -> None:
    global ROPE_LAYERS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    P = config.paths()

    scope = config.NET_SCOPE
    ROPE_LAYERS = [
        (f"{scope}/input_layer3", f"{scope}/conv1", config.NKVHEAD),  # pe_k_cos
        (f"{scope}/input_layer4", f"{scope}/conv2", config.NHEAD),    # pe_q_cos
        (f"{scope}/input_layer5", f"{scope}/conv3", config.NKVHEAD),  # pe_k_sin
        (f"{scope}/input_layer6", f"{scope}/conv4", config.NHEAD),    # pe_q_sin
    ]

    refs = np.load(P.hf_refs)
    wte = refs["wte"]
    hf_logits_last = refs["hf_logits_last"]
    token_embeds = refs["token_embeds"]

    print("=== surgery 1/2: RoPE (remove redundant duplication convs) ===")
    with tempfile.TemporaryDirectory() as d:
        hn_path, hn = load_hn_from_har(P.har_parsed, d)
        layers = hn["layers"]
        rope_surgery(layers, scope)

        print("=== surgery 2/2: attention mask (direct wiring to input_layer2) ===")
        mask_surgery(layers, scope)

        with open(hn_path, "w") as f:
            json.dump(hn, f)
        save_har_from_hn(d, P.har_surgery)
    print(f"surgery HAR saved -> {P.har_surgery}")

    # --- Validate post-surgery fidelity in SDK_NATIVE context ---
    print("=== SDK validation after surgery ===")
    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.exposed_definitions import DistributionStrategy, InferenceContext

    runner = ClientRunner(har=str(P.har_surgery))
    hn_dict = runner.get_hn_dict()["layers"]
    for _, conv_name, _ in ROPE_LAYERS:
        assert conv_name not in hn_dict, f"{conv_name} still present"
    for i in range(1, len(MASK_EW_ADDS) + 1):
        assert f"{scope}/slice{i}" not in hn_dict, f"slice{i} still present"

    theta = config.head_dim_frequencies()
    angles = np.outer(np.arange(config.SEQ), theta.astype(np.float64))
    cos_base = np.cos(angles).astype(np.float32)
    sin_base = np.sin(angles).astype(np.float32)

    def tile_groups(base, groups):
        return np.tile(base, (1, groups)).reshape(1, 1, config.SEQ, groups * config.HD).astype(np.float32)

    # Pre-tile widths, as the host would supply them at run time.
    calib = {
        f"{scope}/input_layer1": token_embeds[:, np.newaxis, :, :].astype(np.float32),
        f"{scope}/input_layer2": config.causal_mask_tiled(1, config.SEQ),
        f"{scope}/input_layer3": tile_groups(cos_base, config.NKVHEAD),
        f"{scope}/input_layer4": tile_groups(cos_base, config.NHEAD),
        f"{scope}/input_layer5": tile_groups(sin_base, config.NKVHEAD),
        f"{scope}/input_layer6": tile_groups(sin_base, config.NHEAD),
    }
    with runner.infer_context(InferenceContext.SDK_NATIVE, gpu_policy=DistributionStrategy.SINGLE) as ctx:
        out = runner.infer(ctx, dataset=calib, data_type="np_array", batch_size=1)
    logits = np.array(out[0] if isinstance(out, list) else out).reshape(1, 1, -1)
    sim = config.cosine(hf_logits_last, logits)
    print(f"cosine(HF last position, HAR/SDK_NATIVE, post-surgery): {sim:.6f}")
    assert sim > 0.999, "surgery broke model fidelity"

    # --- Attach genai external resources ---
    print("=== attaching external resources ===")
    attach_resources(runner, wte, theta)  # theta is already the doubled layout

    hailo_config = build_hailo_config()
    with open(P.hailo_config, "w") as f:
        json.dump(hailo_config, f, indent=2)

    tokenizer_json = P.tokenizer_dir / "tokenizer.json"
    assert tokenizer_json.exists(), (
        f"{tokenizer_json} missing — run s1_export_onnx.py first"
    )
    runner.add_external_file("tokenizer.json", str(tokenizer_json))
    runner.add_external_file("hailo-config.json", str(P.hailo_config))

    runner.save_har(str(P.har_resources))
    print(f"resources HAR saved -> {P.har_resources}")
    print("[OK] step 3 complete")


if __name__ == "__main__":
    main()
