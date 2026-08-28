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

# Surgery table: (input layer, duplication conv to remove, final head count)
ROPE_LAYERS = None  # built in main(), depends on NET_SCOPE


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


def mask_surgery(layers: dict, scope: str) -> list:
    """Rewire every mask-consuming element-wise add directly to input_layer2.

    One such add exists per transformer layer, so the count scales with
    NLAYERS. Discovered from input_layer2's own consumer list rather than a
    hardcoded per-layer name pattern (e.g. ew_add3/8/13/18) — a fixed-length
    list silently produces an inconsistent graph on any checkpoint with a
    different layer count (see docs/findings/ for the finding this fixed).

    Returns the list of removed slice layer names, for the caller to verify
    none remain post-surgery.
    """
    full_width = config.NHEAD * config.SEQ
    il2_name = f"{scope}/input_layer2"
    il2 = layers[il2_name]
    slice_names = [n for n in il2["output"] if layers[n].get("type") == "slice"]
    assert slice_names, f"no slice layers found consuming {il2_name}"
    assert len(slice_names) == config.NLAYERS, (
        f"found {len(slice_names)} mask slices but NLAYERS={config.NLAYERS} "
        "— one mask-consuming add is expected per transformer layer"
    )

    new_il2_output = list(il2["output"])
    for slice_name in slice_names:
        consumers = list(layers[slice_name]["output"])
        assert len(consumers) == 1, (
            f"{slice_name} has {len(consumers)} consumers, expected exactly 1"
        )
        ew_full = consumers[0]
        ew = layers[ew_full]
        # Reconnect directly to input_layer2 (already at full tiled width)
        # and neutralize any repeat/tile expansion params.
        ew["input"] = [il2_name if x == slice_name else x for x in ew["input"]]
        ew["input_shapes"] = [[-1, 1, config.SEQ, full_width]] * 2
        ew["params"]["input_repeats"] = [[1, 1, 1], [1, 1, 1]]
        ew["params"].pop("input_tiles", None)
        new_il2_output = [ew_full if x == slice_name else x for x in new_il2_output]
        del layers[slice_name]
        print(f"  {ew_full}: rewired to {il2_name}; removed {slice_name}")
    il2["output"] = new_il2_output
    return slice_names


def lm_head_split(layers: dict, scope: str, params: dict, n_shards: int) -> tuple[str, list[str]] | None:
    # NOTE: `params` is the network's raw weights dict as stored in the
    # (pre-optimize) HAR's .npz -- key layout "<layer>/<param>:0", plain
    # numpy arrays. Mutated in place; caller re-serializes to .npz.
    """Split the monolithic lm_head matmul into N independently-placed
    output convs, pre-quantization -- the same technique official Hailo
    HEFs use for large vocabularies (confirmed by inspecting an official
    Qwen2.5-1.5B HEF's output vstreams: 4 separate ~38K-wide outputs, not
    one 152K-wide one). No-op when a single shard covers the whole VOCAB.

    Deliberately NOT built on the `defuse()` model-script command (tried
    first, see docs/findings/large-vocab-lm-head-sharding.md): defuse's
    mandatory auto-generated on-chip concat is itself the root of two
    separate, severe compile-time failures at scale (subcluster
    starvation and a deterministic multi-context topology error) --
    see docs/findings/large-body-multicontext-topology.md. This function
    instead produces genuinely independent output layers with no on-chip
    merge at all, exactly mirroring the official recipe; the host
    concatenates the N logits arrays after inference (already handled by
    every downstream cosine check in this pipeline via the
    `isinstance(out, list)` branch).

    Runs on `resources.har` (pre-`optimize()`) deliberately: this is
    still the single, undeplicated base scope --
    `set_kv_cache_global_params` only duplicates into `__prefill`/`__tbt`
    scopes *inside* `optimize()` (confirmed elsewhere in this project).
    DFC's own duplication pass then replicates these N output layers into
    all three scopes automatically, exactly like every other layer --
    no per-scope handling needed here, unlike the abandoned defuse-based
    approach which had to target three scopes explicitly.
    """
    if n_shards <= 1:
        return None

    out_layer_name = next(
        name for name, layer in layers.items()
        if layer.get("type") == "output_layer" and name.startswith(f"{scope}/")
        and any(o.startswith("logits") for o in layer.get("original_names", []))
    )
    out_layer = layers[out_layer_name]
    old_name = out_layer["input"][0]
    old_layer = layers[old_name]
    vocab = old_layer["output_shapes"][0][-1]
    bounds = np.linspace(0, vocab, n_shards + 1, dtype=int)

    kernel = params[f"{old_name}/kernel:0"]
    bias = params[f"{old_name}/bias:0"]
    pad_const = params[f"{old_name}/padding_const_value:0"]

    shard_names = []
    for i in range(n_shards):
        lo, hi = int(bounds[i]), int(bounds[i + 1])
        shard_name = f"{old_name}_shard{i}"
        shard_out_name = f"{out_layer_name}_shard{i}"

        shard_layer = dict(old_layer)
        shard_layer["output_shapes"] = [old_layer["output_shapes"][0][:-1] + [hi - lo]]
        shard_layer["output"] = [shard_out_name]
        shard_layer["original_names"] = [f"logits_{i}"]
        shard_layer["params"] = dict(old_layer["params"])
        shard_layer["params"]["kernel_shape"] = [1, 1, kernel.shape[2], hi - lo]
        layers[shard_name] = shard_layer

        shard_out_layer = dict(out_layer)
        shard_out_layer["input"] = [shard_name]
        shard_out_layer["input_shapes"] = [shard_layer["output_shapes"][0]]
        shard_out_layer["output_shapes"] = [shard_layer["output_shapes"][0]]
        shard_out_layer["original_names"] = [f"logits_{i}"]
        layers[shard_out_name] = shard_out_layer
        shard_names.append(shard_name)

        params[f"{shard_name}/kernel:0"] = kernel[:, :, :, lo:hi]
        params[f"{shard_name}/bias:0"] = bias[lo:hi]
        params[f"{shard_name}/padding_const_value:0"] = pad_const

    for pred_name in old_layer["input"]:
        pred = layers[pred_name]
        pred["output"] = [n for n in pred["output"] if n != old_name] + shard_names

    del layers[old_name]
    del layers[out_layer_name]
    del params[f"{old_name}/kernel:0"]
    del params[f"{old_name}/bias:0"]
    del params[f"{old_name}/padding_const_value:0"]

    print(f"  lm_head split: {old_name} ({vocab} wide) -> {n_shards} shards ({shard_names[0]}..{shard_names[-1]})")
    return old_name, shard_names


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
    config.load()  # picks up run_config.json written by step 1, if any
    if config.COSINE_MIN < 0.999:
        print(f"!! COSINE_MIN overridden to {config.COSINE_MIN} (validated default: 0.999) !!")
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
        removed_slices = mask_surgery(layers, scope)

        if config.LM_HEAD_SHARDS > 1:
            print(f"=== surgery 3/3: lm_head split ({config.LM_HEAD_SHARDS} shards) ===")
            npz_path = hn_path[: -len(".hn")] + ".npz"
            params = dict(np.load(npz_path))
            split_info = lm_head_split(layers, scope, params, config.LM_HEAD_SHARDS)
            old_name, shard_names = split_info
            np.savez(npz_path, **params)
            order = hn["net_params"]["output_layers_order"]
            idx = order.index(old_name)
            hn["net_params"]["output_layers_order"] = order[:idx] + shard_names + order[idx + 1:]

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
    for slice_name in removed_slices:
        assert slice_name not in hn_dict, f"{slice_name} still present"

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
    # Multiple outputs when config.LM_HEAD_SHARDS > 1 — see
    # docs/findings/large-vocab-lm-head-sharding.md.
    shards = out if isinstance(out, list) else [out]
    logits = np.concatenate([np.array(s).reshape(1, 1, -1) for s in shards], axis=-1)
    sim = config.cosine(hf_logits_last, logits)
    print(f"cosine(HF last position, HAR/SDK_NATIVE, post-surgery): {sim:.6f}")
    assert sim > config.COSINE_MIN, "surgery broke model fidelity"

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
