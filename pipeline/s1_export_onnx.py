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
    python s1_export_onnx.py            # uses $DFC_WORKDIR (default ./workdir)
"""
import argparse
import math

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


ROTATE_HALF_Q = make_rotate_half_matrix(config.HD, config.NHEAD)
ROTATE_HALF_K = make_rotate_half_matrix(config.HD, config.NKVHEAD)
TILE_Q = make_tile_matrix(config.HD, config.NHEAD)
TILE_K = make_tile_matrix(config.HD, config.NKVHEAD)
REPEAT_KV_MATRIX = make_repeat_kv_matrix(config.HD, config.NKVHEAD, config.NREP)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = config.RMS_EPS) -> torch.Tensor:
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


class GQALayer(torch.nn.Module):
    """One transformer layer using only export-friendly ops."""

    def __init__(self, layer):
        super().__init__()
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

    def forward(self, x, attention_mask_tiled, k_cos_t, q_cos_t, k_sin_t, q_sin_t):
        residual = x
        h = rms_norm(x, self.ln1_w)
        b, s, _ = h.shape

        q = h @ self.Wq
        k = h @ self.Wk
        v = h @ self.Wv
        q = q * q_cos_t + (q @ ROTATE_HALF_Q) * q_sin_t
        k = k * k_cos_t + (k @ ROTATE_HALF_K) * k_sin_t
        k = k @ REPEAT_KV_MATRIX  # GQA: expand KV heads to Q heads
        v = v @ REPEAT_KV_MATRIX

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

    def __init__(self, hf_model):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [GQALayer(hf_model.model.layers[i]) for i in range(config.NLAYERS)]
        )
        self.norm_w = torch.nn.Parameter(hf_model.model.norm.weight.detach().clone())
        # Separate weight matrix — this model does NOT tie embeddings
        # (`tie_word_embeddings=False`, verified on the checkpoint).
        self.Wlm = torch.nn.Parameter(hf_model.lm_head.weight.detach().T.clone())  # (HIDDEN, VOCAB)

    def forward(self, token_embeds, attention_mask_tiled, pe_k_cos, pe_q_cos, pe_k_sin, pe_q_sin):
        x = token_embeds
        k_cos_t = pe_k_cos @ TILE_K
        q_cos_t = pe_q_cos @ TILE_Q
        k_sin_t = pe_k_sin @ TILE_K
        q_sin_t = pe_q_sin @ TILE_Q
        for layer in self.layers:
            x = layer(x, attention_mask_tiled, k_cos_t, q_cos_t, k_sin_t, q_sin_t)
        x = x.reshape(x.shape[0], x.shape[1], config.HIDDEN)
        x = rms_norm(x, self.norm_w)
        x_last = x[:, -1:, :]  # last position only — see module docstring
        return x_last @ self.Wlm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", default=None, help="override $DFC_WORKDIR")
    args = parser.parse_args()
    if args.workdir:
        config.set_workdir(args.workdir)
    P = config.paths()
    P.workdir.mkdir(parents=True, exist_ok=True)

    print(f"==> loading {config.MODEL_ID}")
    hf_model = AutoModelForCausalLM.from_pretrained(config.MODEL_ID, torch_dtype=torch.float32).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    wte = hf_model.model.embed_tokens.weight.detach().numpy().astype(np.float32)
    assert wte.shape == (config.VOCAB, config.HIDDEN)
    assert not (hf_model.lm_head.weight is hf_model.model.embed_tokens.weight), (
        "unexpected tied word embeddings — this pipeline requires tie_word_embeddings=False"
    )

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
    wrapped = ExportableModelWithHead(hf_model).eval()

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
    assert sim > 0.999, "reimplementation diverged from HF"

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
    assert sim_onnx > 0.999, "ONNX export diverged from HF"

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
