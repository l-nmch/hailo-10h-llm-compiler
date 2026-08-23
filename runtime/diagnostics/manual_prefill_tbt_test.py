#!/usr/bin/env python3
"""Low-level prefill + token-by-token probe of a compiled KV-cache HEF.

Drives the ``__prefill`` and ``__tbt`` network groups directly through the
low-level InferModel API — bypassing genai entirely — to compare on-chip
activations against a float32 Hugging Face reference. This is the tool that
isolated the open __tbt cache-read issue (docs/findings/open-tbt-cache-read.md)
and validated prefill as numerically exact.

You need a reference NPZ produced from the float32 model with keys:
    token_ids          int64 [SEQ]           prompt tokens
    next_token_id      int                   HF greedy next token after prompt
    hf_hidden_prefill  fp32 [HIDDEN]         last-position hidden state after prefill
    hf_hidden_tbt      fp32 [HIDDEN]         hidden state for the follow-up tbt step
    embed_rows_prompt  fp32 [SEQ, HIDDEN]    embedding rows of the prompt
    embed_row_next     fp32 [HIDDEN]         embedding row of the follow-up token
Optional (logits-level comparison when no hidden output exists):
    hf_logits_last     fp32 [VOCAB]          HF last-position logits after prefill
    hf_logits_tbt      fp32 [VOCAB]          HF logits for the follow-up tbt step
    next_token_id_tbt  int                   HF greedy token at the tbt step
Optional (deeper layer comparison):
    conv12_prefill_ref_full / conv12_tbt_ref / conv12 output taps

If the HEF exposes no hidden-state output (self-contained HEFs often expose
only the lm_head logits), pass ``--hidden-output ""`` and provide
``hf_logits_last`` / ``hf_logits_tbt`` in the NPZ instead: the comparison
falls back to cosine + argmax on the logits themselves.

Output vstream names are compile-specific (e.g. conv49 = lm_head logits,
slice5 = last-position hidden state); pass them explicitly if your HEF
differs from the reference pipeline.

Usage (on the device host):
    python manual_prefill_tbt_test.py --hef model.hef --reference refs.npz
"""
import argparse

import numpy as np

import runtime_inputs as ri

# Wire-format quantization parameters of input_layer1 (embeddings).
# Read the actual values from your HAR (input_layer1 qp) or via hef_audit.py.
EMB_QP_SCALE = 1.440075175196398e-05
EMB_QP_ZP = 15923.0


def name_with_suffix(names, suffix: str) -> str:
    matches = [n for n in names if n.endswith(suffix)]
    assert len(matches) >= 1, f"no output ending in {suffix!r} among {list(names)}"
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hef", required=True)
    parser.add_argument("--net-scope", default="ts25mpipe")
    parser.add_argument("--reference", required=True, help="HF reference .npz")
    parser.add_argument("--seq", type=int, default=24, help="total cache size")
    parser.add_argument("--prefill", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--emb-qp-scale", type=float, default=EMB_QP_SCALE)
    parser.add_argument("--emb-qp-zp", type=float, default=EMB_QP_ZP)
    parser.add_argument("--logits-output", default="conv49")
    parser.add_argument("--hidden-output", default="slice5")
    parser.add_argument("--extra-output", default="precision_change1",
                        help="optional early-layer tap (empty string to skip)")
    args = parser.parse_args()

    import hailo_platform as hpf
    from hailo_platform.pyhailort import _pyhailort

    ref = np.load(args.reference)
    theta = ri.rope_frequencies(args.head_dim, args.rope_theta)

    print("==> opening VDevice + InferModel (__prefill and __tbt)")
    vdevice = hpf.VDevice()
    model_prefill = vdevice.create_infer_model(args.hef, f"{args.net_scope}__prefill")
    model_tbt = vdevice.create_infer_model(args.hef, f"{args.net_scope}__tbt")

    for m in (model_prefill, model_tbt):
        # RoPE inputs stay float32; embeddings/mask are consumed raw uint16/uint8.
        for name in m.input_names:
            if any(s in name for s in ("input_layer3", "input_layer4", "input_layer5", "input_layer6")):
                m.input(name).set_format_type(_pyhailort.FormatType.FLOAT32)
        for name in m.output_names:
            m.output(name).set_format_type(_pyhailort.FormatType.FLOAT32)
        m._infer_model.set_enable_kv_cache(True)

    configured_prefill = model_prefill.configure()
    configured_tbt = model_tbt.configure()

    def in_name(m, suffix):
        return name_with_suffix(m.input_names, suffix)

    logits_out = name_with_suffix(model_prefill.output_names, args.logits_output)

    def find_output(names, suffix):
        matches = [n for n in names if n.endswith(suffix)]
        return matches[0] if matches else None

    # Hidden-state comparison is optional: HEFs without such an output fall
    # back to logits-level comparison (needs hf_logits_last/hf_logits_tbt).
    hidden_out = find_output(model_prefill.output_names, args.hidden_output) if args.hidden_output else None
    if hidden_out is None:
        assert "hf_logits_last" in ref, (
            "no hidden-state output in this HEF and no hf_logits_last in the "
            "reference -- nothing to compare against"
        )
        print(f"[i] no {args.hidden_output!r} output in this HEF -- comparing logits instead")
    extra_out = find_output(model_prefill.output_names, args.extra_output) if args.extra_output else None

    # ============================ PREFILL ============================
    # NOTE the ordering gotcha: update_cache_from_embeddings() runs BEFORE
    # prepare_attention_mask_input() inside the server's pre_process — so at
    # mask-build time the cache usage ALREADY equals the prefill length.
    print(f"==> prefill ({args.prefill} tokens)")
    embeds = ri.encode_embeddings_uint16(
        ref["embed_rows_prompt"][np.newaxis].astype(np.float32), args.emb_qp_scale, args.emb_qp_zp
    )
    mask = ri.encode_mask_uint8(ri.build_mask(args.prefill, args.prefill, args.seq, args.n_heads))
    cos_k, sin_k = ri.build_rope(range(args.prefill), args.n_kv_heads, theta)
    cos_q, sin_q = ri.build_rope(range(args.prefill), args.n_heads, theta)

    in_buffers = {
        in_name(model_prefill, "input_layer1"): embeds,
        in_name(model_prefill, "input_layer2"): mask,
        in_name(model_prefill, "input_layer3"): cos_k,
        in_name(model_prefill, "input_layer4"): cos_q,
        in_name(model_prefill, "input_layer5"): sin_k,
        in_name(model_prefill, "input_layer6"): sin_q,
    }
    out_buffers = {logits_out: np.zeros((1, 1, args.vocab), dtype=np.float32)}
    if extra_out:
        out_buffers[extra_out] = np.zeros((1, args.prefill, args.hidden), dtype=np.float32)
    if hidden_out:
        out_buffers[hidden_out] = np.zeros((1, 1, args.hidden), dtype=np.float32)

    bindings = configured_prefill.create_bindings(
        input_buffers=in_buffers, output_buffers=out_buffers
    )
    configured_prefill._configured_infer_model.update_cache_offset(args.prefill)
    configured_prefill.run([bindings], 10000)

    hef_logits = bindings.output(logits_out).get_buffer().flatten()
    hw_next_token = int(np.argmax(hef_logits))

    if hidden_out:
        hef_hidden_prefill = bindings.output(hidden_out).get_buffer().flatten()
        sim_prefill = ri.cosine(hef_hidden_prefill, ref["hf_hidden_prefill"])
        print(f"[prefill] cosine(last-position hidden vs HF): {sim_prefill:.6f}")
    else:
        sim_prefill = ri.cosine(hef_logits, ref["hf_logits_last"])
        print(f"[prefill] cosine(logits vs HF): {sim_prefill:.6f}")
    print(f"[prefill] argmax={hw_next_token} vs HF greedy={int(ref['next_token_id'])}")
    if extra_out and "conv12_prefill_ref_full" in ref:
        tap = bindings.output(extra_out).get_buffer().reshape(args.prefill, args.hidden)
        for i in range(args.prefill):
            c = ri.cosine(tap[i], ref["conv12_prefill_ref_full"][i])
            print(f"    position {i}: cosine = {c:.6f}")

    # ============================== TBT ==============================
    # Same ordering gotcha: cache usage is already prefill+1 when the tbt
    # mask is built (this token was just written to cache).
    print(f"==> tbt step 1 (position {args.prefill})")
    embed_next = ri.encode_embeddings_uint16(
        ref["embed_row_next"][np.newaxis, np.newaxis].astype(np.float32),
        args.emb_qp_scale, args.emb_qp_zp,
    )
    mask_tbt = ri.encode_mask_uint8(
        ri.build_mask(1, args.prefill + 1, args.seq, args.n_heads)
    )
    cos_kt, sint_kt = ri.build_rope([args.prefill], args.n_kv_heads, theta)
    cos_qt, sin_qt = ri.build_rope([args.prefill], args.n_heads, theta)

    in_buffers_t = {
        in_name(model_tbt, "input_layer1"): embed_next,
        in_name(model_tbt, "input_layer2"): mask_tbt,
        in_name(model_tbt, "input_layer3"): cos_kt,
        in_name(model_tbt, "input_layer4"): cos_qt,
        in_name(model_tbt, "input_layer5"): sint_kt,
        in_name(model_tbt, "input_layer6"): sin_qt,
    }
    # Output vstream names are group-scoped ("ts25mpipe__tbt/conv49", not
    # "ts25mpipe__prefill/conv49") -- resolve them against the tbt model.
    logits_out_t = name_with_suffix(model_tbt.output_names, args.logits_output)
    hidden_out_t = find_output(model_tbt.output_names, args.hidden_output) if hidden_out else None
    out_buffers_t = {
        logits_out_t: np.zeros((1, 1, args.vocab), dtype=np.float32),
    }
    if hidden_out_t:
        out_buffers_t[hidden_out_t] = np.zeros((1, 1, args.hidden), dtype=np.float32)
    bindings_t = configured_tbt.create_bindings(
        input_buffers=in_buffers_t, output_buffers=out_buffers_t
    )
    configured_tbt._configured_infer_model.update_cache_offset(1)
    configured_tbt.run([bindings_t], 10000)

    hef_logits_t = bindings_t.output(logits_out_t).get_buffer().flatten()
    if hidden_out_t:
        hef_hidden_tbt = bindings_t.output(hidden_out_t).get_buffer().flatten()
        sim_tbt = ri.cosine(hef_hidden_tbt, ref["hf_hidden_tbt"])
        print(f"[tbt] cosine(last-position hidden vs HF): {sim_tbt:.6f}")
    else:
        sim_tbt = ri.cosine(hef_logits_t, ref["hf_logits_tbt"])
        print(f"[tbt] cosine(logits vs HF): {sim_tbt:.6f}")
    print(f"[tbt] argmax={int(np.argmax(hef_logits_t))} vs HF greedy={int(ref['next_token_id_tbt'])}")
    if extra_out and "conv12_tbt_ref" in ref:
        extra_out_t = name_with_suffix(model_tbt.output_names, args.extra_output)
        tap_t = bindings_t.output(extra_out_t).get_buffer().flatten()
        print(f"[tbt] early-layer tap cosine = {ri.cosine(tap_t, ref['conv12_tbt_ref']):.6f}")

    print("\ninterpretation:")
    print("  prefill exact but tbt degraded -> cache-read side issue")
    print("  (see docs/findings/open-tbt-cache-read.md)")


if __name__ == "__main__":
    main()
