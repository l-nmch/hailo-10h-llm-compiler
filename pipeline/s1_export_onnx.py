#!/usr/bin/env python3
"""Step 1 — Hugging Face checkpoint → ONNX.

Loads the model from the Hub, re-expresses it with "matmul tricks" (RoPE,
GQA repeat_kv and head tiling as constant-matrix multiplies), appends the
`lm_head` matmul restricted to the LAST sequence position, validates the
reimplementation and the ONNX export against the original float32 model,
and persists the reference tensors later steps reuse.

Why a reimplementation at all? The DFC parser rejects native torch shuffle
ops (`repeat_interleave`, `Expand`) with `UnsupportedShuffleLayerError`.
Expressing them as constant-matrix multiplies keeps every op in the
matmul/add/mul subset. This mirrors what Hailo's own official LLM graphs do
internally (`matmul(groups=..., input_tiles=...)`) — it is not a hack.

Two graph-level fixes are inseparable here:

- **lm_head**: the genai runtime argmaxes the raw HEF output directly; a
  graph that stops at the hidden state is never projected onto the vocab.
- **last-position slice** (`x[:, -1:, :]`): the runtime predicts exactly one
  token, from the last context position. Slicing also shrinks lm_head's
  effective batch to one position, which is what makes the monolithic
  256x32000 matmul placeable on-chip ("Agent infeasible" otherwise).

Usage:
    python s1_export_onnx.py                       # validated TinyStories default
    python s1_export_onnx.py --model <hf-id>        # any eligible checkpoint
        [--seq N] [--prefill-size N] [--calibset-size N] [--net-scope NAME]
"""
import argparse
import math
from types import SimpleNamespace

import config  # must precede numpy imports — sets NPY_PROMOTION_STATE et al.

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Matmul-trick helpers
# ---------------------------------------------------------------------------

def make_rotate_half_matrix(hd: int, n_heads: int) -> torch.Tensor:
    """Block-diagonal matrix implementing RoPE's rotate-half on `n_heads`
    concatenated heads of width `hd`: x @ M == rotate_half(x)."""
    perm = torch.zeros(hd, hd)
    for i in range(hd // 2):
        perm[i, i + hd // 2] = 1.0
        perm[i + hd // 2, i] = -1.0
    return torch.block_diag(*([perm] * n_heads))


def make_tile_matrix(hd: int, n_heads: int) -> torch.Tensor:
    """Matrix replicating one head-width vector across `n_heads` heads:
    x @ M == tile(x, n_heads)."""
    m = torch.zeros(hd, hd * n_heads)
    for h in range(n_heads):
        for i in range(hd):
            m[i, h * hd + i] = 1.0
    return m


def make_repeat_kv_matrix(hd: int, n_kv_heads: int, n_rep: int) -> torch.Tensor:
    """Matrix expanding KV heads to query heads (GQA): x @ M == repeat_kv(x)."""
    kv_width = hd * n_kv_heads
    q_width = hd * n_kv_heads * n_rep
    m = torch.zeros(kv_width, q_width)
    for kv in range(n_kv_heads):
        for r in range(n_rep):
            out_head = kv * n_rep + r
            for d in range(hd):
                m[kv * hd + d, out_head * hd + d] = 1.0
    return m


def make_head_seg_matrix(hd: int, n_heads: int) -> torch.Tensor:
    """Segment-sum matrix over `n_heads` concatenated heads of width `hd`:
    `x @ M` gives per-head sums (shape `[..., n_heads]`); `(x @ M) @ M.T`
    broadcasts each head's sum back across its own `hd`-wide slice. Lets
    per-head RMSNorm (QK-Norm) be computed with matmuls only — no
    reshape/view, which the DFC parser's shape-format inference chokes on
    (`ValueError: width is not in list`)."""
    m = torch.zeros(hd * n_heads, n_heads)
    for h in range(n_heads):
        m[h * hd : (h + 1) * hd, h] = 1.0
    return m


def build_matmul_trick_matrices():
    """Build the RoPE/tiling/GQA/QK-Norm constant matrices for the CURRENT
    config (config.HD/NHEAD/NKVHEAD/NREP). Must run after config.load() —
    these depend on the checkpoint's architecture, not just its size."""
    return SimpleNamespace(
        rotate_half_q=make_rotate_half_matrix(config.HD, config.NHEAD),
        rotate_half_k=make_rotate_half_matrix(config.HD, config.NKVHEAD),
        tile_q=make_tile_matrix(config.HD, config.NHEAD),
        tile_k=make_tile_matrix(config.HD, config.NKVHEAD),
        repeat_kv=make_repeat_kv_matrix(config.HD, config.NKVHEAD, config.NREP),
        head_seg_q=make_head_seg_matrix(config.HD, config.NHEAD),
        head_seg_k=make_head_seg_matrix(config.HD, config.NKVHEAD),
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = config.RMS_EPS) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


def rms_norm_headwise(
    x: torch.Tensor, weight_tiled: torch.Tensor, seg_matrix: torch.Tensor,
    hd: int, eps: float = config.RMS_EPS,
) -> torch.Tensor:
    """Per-head RMSNorm (QK-Norm) on a flat `[..., n_heads*hd]` tensor,
    without reshaping to `[..., n_heads, hd]` — see `make_head_seg_matrix`."""
    sumsq = x.pow(2) @ seg_matrix  # [..., n_heads]
    variance = (sumsq @ seg_matrix.T) / hd  # [..., n_heads*hd], broadcast per head
    x = x * torch.rsqrt(variance + eps)
    return x * weight_tiled


class GQALayer(torch.nn.Module):
    """One transformer layer using only export-friendly ops."""

    def __init__(self, layer, matrices: SimpleNamespace):
        super().__init__()
        self.m = matrices
        # Weights stored transposed (in_features, out_features) so forward
        # passes are plain `x @ W` matmuls.
        self.Wq = torch.nn.Parameter(layer.self_attn.q_proj.weight.detach().T.clone())
        self.Wk = torch.nn.Parameter(layer.self_attn.k_proj.weight.detach().T.clone())
        self.Wv = torch.nn.Parameter(layer.self_attn.v_proj.weight.detach().T.clone())
        self.Wo = torch.nn.Parameter(layer.self_attn.o_proj.weight.detach().T.clone())
        self.ln1_w = torch.nn.Parameter(layer.input_layernorm.weight.detach().clone())
        self.ln2_w = torch.nn.Parameter(layer.post_attention_layernorm.weight.detach().clone())
        self.Wgate = torch.nn.Parameter(layer.mlp.gate_proj.weight.detach().T.clone())
        self.Wup = torch.nn.Parameter(layer.mlp.up_proj.weight.detach().T.clone())
        self.Wdown = torch.nn.Parameter(layer.mlp.down_proj.weight.detach().T.clone())
        # QK-Norm (Qwen3-style): optional per-head RMSNorm on Q/K, applied
        # right after projection, before RoPE. Absent on LLaMA2-style models.
        q_norm = getattr(layer.self_attn, "q_norm", None)
        k_norm = getattr(layer.self_attn, "k_norm", None)
        self.qk_norm = q_norm is not None and k_norm is not None
        if self.qk_norm:
            # Pre-tiled across heads once here (constant fold), so forward()
            # stays matmul/elementwise-only — see rms_norm_headwise().
            self.q_norm_w = torch.nn.Parameter(q_norm.weight.detach().clone() @ matrices.tile_q)
            self.k_norm_w = torch.nn.Parameter(k_norm.weight.detach().clone() @ matrices.tile_k)

    def forward(self, x, attention_mask_tiled, k_cos_t, q_cos_t, k_sin_t, q_sin_t):
        residual = x
        h = rms_norm(x, self.ln1_w)
        b, s, _ = h.shape

        q = h @ self.Wq
        k = h @ self.Wk
        v = h @ self.Wv
        if self.qk_norm:
            q = rms_norm_headwise(q, self.q_norm_w, self.m.head_seg_q, config.HD)
            k = rms_norm_headwise(k, self.k_norm_w, self.m.head_seg_k, config.HD)
        q = q * q_cos_t + (q @ self.m.rotate_half_q) * q_sin_t
        k = k * k_cos_t + (k @ self.m.rotate_half_k) * k_sin_t
        k = k @ self.m.repeat_kv  # GQA: expand KV heads to Q heads
        v = v @ self.m.repeat_kv

        q = q.view(b, s, config.NHEAD, config.HD).transpose(1, 2)
        k = k.view(b, s, config.NHEAD, config.HD).transpose(1, 2)
        v = v.view(b, s, config.NHEAD, config.HD).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(config.HD)
        scores = scores + attention_mask_tiled[:, :, :, :s]
        probs = torch.softmax(scores, dim=-1)

        attn_out = torch.matmul(probs, v)
        attn_out = attn_out.transpose(1, 2).reshape(b, s, config.Q_WIDTH)
        attn_out = attn_out @ self.Wo
        x = residual + attn_out

        residual = x
        h = rms_norm(x, self.ln2_w)
        gate = torch.nn.functional.silu(h @ self.Wgate)
        up = h @ self.Wup
        x = residual + ((gate * up) @ self.Wdown)
        return x


class ExportableModelWithHead(torch.nn.Module):
    """Full model ending in lm_head over the last position only."""

    def __init__(self, hf_model, matrices: SimpleNamespace):
        super().__init__()
        self.m = matrices
        self.layers = torch.nn.ModuleList(
            [GQALayer(hf_model.model.layers[i], matrices) for i in range(config.NLAYERS)]
        )
        self.norm_w = torch.nn.Parameter(hf_model.model.norm.weight.detach().clone())
        # Baked in as its own independent tensor either way — tied or not,
        # `lm_head.weight` already holds the right values, and DFC's graph
        # has no notion of two ops sharing one weight tensor regardless.
        #
        # Always exported as ONE matmul, even for large-VOCAB checkpoints
        # (e.g. Qwen3's ~152K tokens) where DFC's placer would reject it as
        # a single op — splitting at export time (tried, reverted) hit a
        # second, independent DFC parser bug. The working fix is a
        # `defuse(<layer>, N)` model-script directive added in
        # s6_compile_hef.py at compile time instead, which needs the
        # unsharded single-matmul graph shape parsed here. See
        # docs/findings/large-vocab-lm-head-sharding.md for the full story.
        self.Wlm = torch.nn.Parameter(hf_model.lm_head.weight.detach().T.clone())  # (HIDDEN, VOCAB)

    def forward(self, token_embeds, attention_mask_tiled, pe_k_cos, pe_q_cos, pe_k_sin, pe_q_sin):
        x = token_embeds
        k_cos_t = pe_k_cos @ self.m.tile_k
        q_cos_t = pe_q_cos @ self.m.tile_q
        k_sin_t = pe_k_sin @ self.m.tile_k
        q_sin_t = pe_q_sin @ self.m.tile_q
        for layer in self.layers:
            x = layer(x, attention_mask_tiled, k_cos_t, q_cos_t, k_sin_t, q_sin_t)
        x = x.reshape(x.shape[0], x.shape[1], config.HIDDEN)
        x = rms_norm(x, self.norm_w)
        x_last = x[:, -1:, :]  # last position only — see module docstring
        return x_last @ self.Wlm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    parser.add_argument("--model", default=None,
                        help="HF checkpoint id (default: the validated "
                             "TinyStories checkpoint). Must be untied-embedding, "
                             "RMSNorm+RoPE+SwiGLU+GQA-or-MHA — see "
                             "Porting-Another-Model.md's eligibility screen.")
    parser.add_argument("--seq", type=int, default=None,
                        help="sequence/KV-cache length (default: 24)")
    parser.add_argument("--prefill-size", type=int, default=None,
                        help="prefill scope length (default: 16)")
    parser.add_argument("--calibset-size", type=int, default=None,
                        help="quantization calibration samples (default: 32)")
    parser.add_argument("--net-scope", default=None,
                        help="HEF network-group base name (default: derived "
                             "from --model)")
    parser.add_argument("--cosine-min", type=float, default=None,
                        help="fidelity gate for steps 1-3 (default: 0.999, "
                             "the bar this pipeline was validated against on "
                             "a 4-layer model). Lower this ONLY after "
                             "confirming a drop is benign drift, not a real "
                             "bug — see config.py's COSINE_MIN docstring.")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    # Step 1 always resolves a full config (even with no flags at all) so
    # run_config.json exists for steps 2-6 to pick up — bare `config.load()`
    # only reads an *existing* run_config.json, it never re-derives.
    config.load(
        args.model or config.MODEL_ID, seq=args.seq, prefill_size=args.prefill_size,
        calibset_size=args.calibset_size, net_scope=args.net_scope,
        cosine_min=args.cosine_min,
    )
    if config.COSINE_MIN < 0.999:
        print(f"!! COSINE_MIN overridden to {config.COSINE_MIN} (validated default: 0.999) "
              "!! — every fidelity gate in steps 1-3 uses this relaxed bar for this run.")
    P = config.paths()
    P.workdir.mkdir(parents=True, exist_ok=True)
    matrices = build_matmul_trick_matrices()

    print(f"==> loading {config.MODEL_ID}")
    # attn_implementation="eager" makes HF's own forward pass use plain
    # matmul+softmax+matmul attention, matching our reimplementation's math
    # exactly instead of a fused SDPA kernel that may accumulate differently
    # — kept for a deterministic, apples-to-apples reference even though it
    # did NOT explain the large-model cosine gap tested in
    # docs/findings/sdk-native-cosine-drift.md (ruled out there, on CPU SDPA
    # falls back to the same math as eager anyway).
    hf_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.float32, attn_implementation="eager",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    wte = hf_model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
    assert wte.shape == (config.VOCAB, config.HIDDEN)

    # A prompt of EXACTLY SEQ real tokens (no padding): since the graph slices
    # at the last position before lm_head, comparing against a padded position
    # would be meaningless.
    prompt = "Once upon a time there was a little girl"
    long_prompt = (prompt + " ") * 6
    ids = tokenizer(long_prompt, return_tensors="pt")["input_ids"][:, : config.SEQ]
    assert ids.shape[1] == config.SEQ, "prompt too short to fill SEQ tokens"

    with torch.no_grad():
        hf_logits_full = hf_model(ids).logits.numpy()  # (1, SEQ, VOCAB)
    token_ids = ids.numpy().astype(np.int64)
    token_embeds = wte[token_ids].astype(np.float32)

    # Persist everything later steps need.
    np.save(P.wte, wte)
    P.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(P.tokenizer_dir)

    print("==> building exportable reimplementation (lm_head + last-position slice)")
    wrapped = ExportableModelWithHead(hf_model, matrices).eval()

    theta = config.head_dim_frequencies()
    positions = np.arange(config.SEQ)
    angles = np.outer(positions, theta.astype(np.float64))
    cos_full = torch.tensor(np.cos(angles), dtype=torch.float32).unsqueeze(0)
    sin_full = torch.tensor(np.sin(angles), dtype=torch.float32).unsqueeze(0)
    mask_tiled = torch.tensor(config.causal_mask_tiled(1, config.SEQ), dtype=torch.float32)

    with torch.no_grad():
        logits_wrapped = wrapped(
            torch.tensor(token_embeds), mask_tiled, cos_full, cos_full, sin_full, sin_full
        ).numpy()  # (1, 1, VOCAB)
    sim = config.cosine(hf_logits_full[:, -1:, :], logits_wrapped)
    print(f"cosine(HF last position, PyTorch reimplementation): {sim:.6f}")
    assert sim > config.COSINE_MIN, "reimplementation diverged from HF"

    print("==> exporting ONNX + validating with onnxruntime")
    torch.onnx.export(
        wrapped,
        (torch.randn(1, config.SEQ, config.HIDDEN), mask_tiled, cos_full, cos_full, sin_full, sin_full),
        str(P.onnx),
        input_names=["inputs_embeds", "attention_mask", "pe_k_cos", "pe_q_cos", "pe_k_sin", "pe_q_sin"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,  # legacy TorchScript exporter: predictable tracing, validated here
    )
    sess = ort.InferenceSession(str(P.onnx), providers=["CPUExecutionProvider"])
    onnx_inputs = {
        "inputs_embeds": token_embeds.astype(np.float32),
        "attention_mask": mask_tiled.numpy().astype(np.float32),
        "pe_k_cos": cos_full.numpy().astype(np.float32),
        "pe_q_cos": cos_full.numpy().astype(np.float32),
        "pe_k_sin": sin_full.numpy().astype(np.float32),
        "pe_q_sin": sin_full.numpy().astype(np.float32),
    }
    onnx_logits = sess.run(["logits"], onnx_inputs)[0]
    sim_onnx = config.cosine(hf_logits_full[:, -1:, :], onnx_logits)
    print(f"cosine(HF last position, ONNX/onnxruntime): {sim_onnx:.6f}")
    assert sim_onnx > config.COSINE_MIN, "ONNX export diverged from HF"

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    np.savez(
        P.hf_refs,
        token_ids=token_ids,
        token_embeds=token_embeds,
        hf_logits_last=hf_logits_full[:, -1:, :],
        wte=wte,
        pad_id=np.array(pad_id),
    )
    print(f"artifacts written to {P.workdir}")
    print("[OK] step 1 complete")


if __name__ == "__main__":
    main()
