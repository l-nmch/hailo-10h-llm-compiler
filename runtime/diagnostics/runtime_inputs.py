"""Shared helpers for building raw HEF inputs by hand (diagnostics).

The genai runtime normally constructs these inputs host-side. When debugging
below that layer (manual prefill/tbt runs through the low-level InferModel
API), you must reproduce its exact conventions:

- **attention mask**: additive float mask (0 allowed / -100 blocked),
  quantized on the wire to uint8 (allowed -> 255, blocked -> 0);
- **embeddings**: fp32 rows quantized on the wire to uint16 codes using the
  input layer's quantization parameters (scale/zp — read them from your
  HAR/HEF; defaults here match the reference model of this repo);
- **RoPE cos/sin**: computed from a theta table
  ``1/(theta ** (arange(0, HD, 2)/HD))`` concatenated twice, tiled per head
  group. Positions are integers; the mask/RoPE layout below mirrors
  pre_process.cpp's block structure (see docs/findings/).
"""
import numpy as np


def rope_frequencies(head_dim: int, rope_theta: float) -> np.ndarray:
    """theta_size = head_dim/2 exponentials, concatenated twice."""
    base = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    return np.concatenate([base, base]).astype(np.float32)


def build_mask(
    layer_input_tokens_size: int,
    mask_cache_usage: int,
    cache_size: int,
    n_heads: int,
    allowed: float = 0.0,
    blocked: float = -100.0,
) -> np.ndarray:
    """Replicate the runtime's mask structure for one network group.

    The mask has two blocks over `cache_size` columns:

    - block 1 (`layer_input_tokens_size - mask_cache_usage` rows): prompt
      positions already fully in cache — everything allowed;
    - block 2 (`mask_cache_usage` rows): unused columns blocked, then the
      freshly-written cache columns allowed, then a causal lower-triangular
      self-attention block.

    Returns shape [1, rows, n_heads * cache_size] (head-tiled).
    """
    block1_rows = max(layer_input_tokens_size - mask_cache_usage, 0)
    block2_rows = min(mask_cache_usage, layer_input_tokens_size)
    rows = block1_rows + block2_rows

    m = np.full((rows, cache_size), blocked, dtype=np.float32)
    if block1_rows > 0:
        m[:block1_rows, :] = allowed
    if block2_rows > 0:
        r0 = block1_rows
        unused_cols = cache_size - mask_cache_usage
        m[r0:, :unused_cols] = blocked

        mid_cols = mask_cache_usage - block2_rows
        if mid_cols > 0:
            m[r0:, unused_cols:unused_cols + mid_cols] = allowed

        right_col = unused_cols + mid_cols
        self_block = np.full((block2_rows, block2_rows), blocked, dtype=np.float32)
        tri = np.tril_indices(block2_rows, k=0)
        self_block[tri] = allowed
        m[r0:, right_col:right_col + block2_rows] = self_block

    return np.tile(m[np.newaxis, :, :], (1, 1, n_heads)).astype(np.float32)


def build_rope(positions, groups: int, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cos/sin tables for `positions`, tiled across `groups` heads.

    `theta` is the doubled table (cos-half + sin-half, theta_size = head_dim
    entries per position); tiling along columns already yields the final
    `groups * theta_size` width consumed by the HEF inputs (e.g.
    K = 16 x 8 = 128, Q = 16 x 16 = 256).

    Returns (cos, sin), each [1, len(positions), groups * theta_size].
    """
    positions = np.asarray(list(positions))
    angles = np.outer(positions.astype(np.float64), theta.astype(np.float64))
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    theta_size = theta.shape[0]
    cos_tiled = np.tile(cos, (1, groups)).reshape(1, len(positions), groups * theta_size)
    sin_tiled = np.tile(sin, (1, groups)).reshape(1, len(positions), groups * theta_size)
    return cos_tiled.astype(np.float32), sin_tiled.astype(np.float32)


def encode_embeddings_uint16(float_rows: np.ndarray, scale: float, zp: float) -> np.ndarray:
    """Quantize fp32 embedding rows to the uint16 codes the device expects."""
    codes = np.round(float_rows.astype(np.float64) / scale + zp)
    return np.clip(codes, 0, 65535).astype(np.uint16)


def encode_mask_uint8(float_mask: np.ndarray, allowed: float = 0.0) -> np.ndarray:
    """Encode an additive float mask to the raw uint8 values on the wire."""
    return np.where(float_mask == allowed, 255, 0).astype(np.uint8)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
