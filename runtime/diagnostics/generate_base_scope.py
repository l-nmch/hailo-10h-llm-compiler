#!/usr/bin/env python3
"""Greedy generation through the BASE scope only — no KV-cache involved.

The base network scope (`<scope>` without the `__prefill`/`__tbt` suffix)
runs the whole sequence in one shot. Driving it in a greedy loop (recompute
the full prefix each step) validated that the compiled model itself is
sound: cosine 0.99 vs float32 HF on hardware AND genuinely coherent greedy
text ("...a small house near a park. The little girl loved"). That control
test is what isolated the open issue to the KV-cache mechanism
(docs/findings/open-tbt-cache-read.md).

Inefficient by design (O(n^2) recomputation, capped at SEQ total positions)
— use genai_generate.py for real generation; use this for debugging.

Usage (on the device host):
    python generate_base_scope.py --hef model.hef --wte wte.npy \
        [--prompt-ids 1 9038 2501 ...] [--vocab-json id_to_str.json]
"""
import argparse
import json

import numpy as np

import runtime_inputs as ri


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hef", required=True)
    parser.add_argument("--net-scope", default="ts25mpipe")
    parser.add_argument("--wte", required=True, help="fp32 embedding table .npy")
    parser.add_argument("--prompt-ids", type=int, nargs="+",
                        help="BOS-terminated prompt token ids (must fit in SEQ)")
    parser.add_argument("--vocab-json", default=None,
                        help="optional {id: piece} map for readable output")
    parser.add_argument("--seq", type=int, default=24, help="base scope length / cap")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--emb-qp-scale", type=float, default=1.440075175196398e-05)
    parser.add_argument("--emb-qp-zp", type=float, default=15923.0)
    parser.add_argument("--logits-output", default="conv49")
    parser.add_argument("--eos-token-id", type=int, default=2)
    args = parser.parse_args()

    import hailo_platform as hpf
    from hailo_platform.pyhailort import _pyhailort

    wte = np.load(args.wte).astype(np.float32)
    id_to_str = {}
    if args.vocab_json:
        with open(args.vocab_json) as f:
            id_to_str = json.load(f)

    def detok(token_id: int) -> str:
        return repr(id_to_str.get(str(token_id), token_id))

    prompt_ids = args.prompt_ids or [
        1, 9038, 2501, 263, 931, 727, 471, 263, 2217, 7826, 1058, 10600, 297, 263, 2319, 3699,
    ]
    assert len(prompt_ids) <= args.seq, "prompt longer than the base scope"
    print("prompt ids:", prompt_ids)

    vdevice = hpf.VDevice()
    model = vdevice.create_infer_model(args.hef, args.net_scope)
    # RoPE inputs stay float32 on this path.
    for name in model.input_names:
        if any(s in name for s in ("input_layer3", "input_layer4", "input_layer5", "input_layer6")):
            model.input(name).set_format_type(_pyhailort.FormatType.FLOAT32)
    for name in model.output_names:
        model.output(name).set_format_type(_pyhailort.FormatType.FLOAT32)
    configured = model.configure()

    def suffix(names, s):
        return next(n for n in names if n.endswith(s))

    logits_out = suffix(model.output_names, args.logits_output)

    theta = ri.rope_frequencies(args.head_dim, args.rope_theta)
    tokens = list(prompt_ids)
    generated = []
    max_new = args.seq - len(tokens)
    print(f"generating up to {max_new} new tokens (SEQ={args.seq} cap)")

    for step in range(max_new):
        n_real = len(tokens)
        pad_rows = args.seq - n_real

        embeds_f = np.zeros((args.seq, args.hidden), dtype=np.float32)
        embeds_f[pad_rows:] = wte[tokens]
        embeds = ri.encode_embeddings_uint16(
            embeds_f[np.newaxis], args.emb_qp_scale, args.emb_qp_zp
        )
        # Right-aligned sequence: cache usage == number of real tokens, so
        # every padded row lands in "block 1" (all allowed) and the causal
        # triangle covers exactly the real positions.
        mask = ri.encode_mask_uint8(ri.build_mask(args.seq, n_real, args.seq, args.n_heads))
        positions = [0] * pad_rows + list(range(n_real))
        cos_k, sin_k = ri.build_rope(positions, args.n_kv_heads, theta)
        cos_q, sin_q = ri.build_rope(positions, args.n_heads, theta)

        in_buffers = {
            suffix(model.input_names, "input_layer1"): embeds,
            suffix(model.input_names, "input_layer2"): mask,
            suffix(model.input_names, "input_layer3"): cos_k,
            suffix(model.input_names, "input_layer4"): cos_q,
            suffix(model.input_names, "input_layer5"): sin_k,
            suffix(model.input_names, "input_layer6"): sin_q,
        }
        out_buffers = {logits_out: np.zeros((1, 1, args.vocab), dtype=np.float32)}
        bindings = configured.create_bindings(input_buffers=in_buffers, output_buffers=out_buffers)
        configured.run([bindings], 10000)

        logits = bindings.output(logits_out).get_buffer().flatten()
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        tokens.append(next_token)
        print(f"  step {step}: token={next_token} {detok(next_token)}", flush=True)
        if next_token == args.eos_token_id:
            print("  EOS reached.")
            break

    print("\n=== generated text (greedy, base scope) ===")
    print(repr("".join(id_to_str.get(str(t), str(t)) for t in generated)))


if __name__ == "__main__":
    main()
