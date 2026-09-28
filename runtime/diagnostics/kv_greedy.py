#!/usr/bin/env python3
"""Greedy multi-token generation through the real KV-cache path (__prefill + __tbt).

Bypasses genai/hailo-ollama and drives the two network groups exactly like
the genai server does (cache offset advanced before each run, ring-buffer
mask convention): one prefill over PREFILL prompt tokens, then one __tbt
step per generated token, each reading the on-chip KV-cache. Compare the
printed ids with a float32 Hugging Face greedy run on the same prompt — this
is the end-to-end check for docs/findings/tbt-cache-read.md.

`--mask-mode self` lets each tbt step attend only to itself (the cache is
fully masked): its logits must match HF's single-token forward, which
separates "tbt graph is wrong" from "what tbt reads from the cache is wrong".

Usage (on the device host):
    python kv_greedy.py --hef model.hef --wte wte.npy --prompt-ids <PREFILL ids> \
        [--dump-logits steps.npy]
"""
import argparse

import numpy as np

import runtime_inputs as ri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hef", required=True)
    parser.add_argument("--net-scope", default="ts25mpipe")
    parser.add_argument("--wte", required=True, help="fp32 embedding table .npy")
    parser.add_argument("--prompt-ids", type=int, nargs="+", required=True,
                        help="exactly PREFILL token ids, BOS included")
    parser.add_argument("--seq", type=int, default=128, help="total KV-cache size")
    parser.add_argument("--prefill", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--mask-mode", choices=("cache", "self"), default="cache")
    parser.add_argument("--dump-logits", default=None, help="save every step's logits (.npy)")
    args = parser.parse_args()
    assert len(args.prompt_ids) == args.prefill, "prompt must be exactly PREFILL tokens"
    assert args.prefill + args.new_tokens <= args.seq, "prompt + new tokens exceed the cache"

    import hailo_platform as hpf
    from hailo_platform.pyhailort import _pyhailort

    wte = np.load(args.wte).astype(np.float32)
    theta = ri.rope_frequencies(args.head_dim, args.rope_theta)
    hef = hpf.HEF(args.hef)
    group_p, group_t = f"{args.net_scope}__prefill", f"{args.net_scope}__tbt"
    qi = next(v.quant_info for v in hef.get_input_vstream_infos(group_p) if v.name.endswith("input_layer1"))

    vdevice = hpf.VDevice()
    models = {}
    for g in (group_p, group_t):
        m = vdevice.create_infer_model(args.hef, g)
        for name in m.input_names:
            if any(s in name for s in ("input_layer3", "input_layer4", "input_layer5", "input_layer6")):
                m.input(name).set_format_type(_pyhailort.FormatType.FLOAT32)
        for name in m.output_names:
            m.output(name).set_format_type(_pyhailort.FormatType.FLOAT32)
        m._infer_model.set_enable_kv_cache(True)
        models[g] = m
    configured = {g: models[g].configure() for g in models}

    def suffix(names, s):
        return next(n for n in names if n.endswith(s))

    def run(group, embeds_f, mask, positions, offset_delta):
        m = models[group]
        cos_k, sin_k = ri.build_rope(positions, args.n_kv_heads, theta)
        cos_q, sin_q = ri.build_rope(positions, args.n_heads, theta)
        inputs = {
            suffix(m.input_names, "input_layer1"): ri.encode_embeddings_uint16(
                embeds_f[np.newaxis], qi.qp_scale, qi.qp_zp),
            suffix(m.input_names, "input_layer2"): mask,
            suffix(m.input_names, "input_layer3"): cos_k,
            suffix(m.input_names, "input_layer4"): cos_q,
            suffix(m.input_names, "input_layer5"): sin_k,
            suffix(m.input_names, "input_layer6"): sin_q,
        }
        # lm_head may be split into several shards; they come back in order
        infos = hef.get_output_vstream_infos(group)
        outputs = {v.name: np.zeros((1, *v.shape), dtype=np.float32) for v in infos}
        bindings = configured[group].create_bindings(input_buffers=inputs, output_buffers=outputs)
        # the server advances the cache offset BEFORE each run
        configured[group]._configured_infer_model.update_cache_offset(offset_delta)
        configured[group].run([bindings], 10000)
        return np.concatenate([bindings.output(v.name).get_buffer().ravel() for v in infos])

    tokens = list(args.prompt_ids)
    mask_p = ri.encode_mask_uint8(ri.build_mask(args.prefill, args.prefill, args.seq, args.n_heads))
    logits = run(group_p, wte[tokens], mask_p, range(args.prefill), args.prefill)
    assert logits.size == args.vocab, f"logits size {logits.size} != vocab {args.vocab}"
    generated, all_logits = [int(np.argmax(logits))], [logits]
    for step in range(args.new_tokens - 1):
        pos = args.prefill + step
        if args.mask_mode == "cache":
            mask_t = ri.build_mask(1, pos + 1, args.seq, args.n_heads)
        else:
            row = np.full((1, args.seq), -100.0, dtype=np.float32)
            row[0, args.seq - 1] = 0.0
            mask_t = np.tile(row[np.newaxis], (1, 1, args.n_heads))
        logits = run(group_t, wte[[generated[-1]]], ri.encode_mask_uint8(mask_t), [pos], 1)
        generated.append(int(np.argmax(logits)))
        all_logits.append(logits)
    print("GENERATED_IDS", generated)
    if args.dump_logits:
        np.save(args.dump_logits, np.stack(all_logits))


if __name__ == "__main__":
    main()
