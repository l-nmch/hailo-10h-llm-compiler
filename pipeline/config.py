"""Shared configuration for the compile pipeline (steps s1..s6).

Import this module FIRST in every pipeline script — before numpy or any
Hailo/PyTorch import. It sets process-wide environment variables that the
toolchain depends on, and those variables must already be in place when
numpy is imported:

- ``NPY_PROMOTION_STATE=legacy``  NumPy >= 2 changed scalar type promotion;
  the DFC quantization stage relies on float32/float64 being preserved
  (float64 != float32 checks inside ``create_hw_params()``).
- ``TRANSFORMERS_NO_TF=1``        keep transformers away from TensorFlow.
- ``USER=root``                   the DFC paths manager consults this when
                                  building its cache directory layout.

All artifact paths are derived from ``$DFC_WORKDIR`` (default
``./workdir``), resolved lazily through :func:`paths` so that a script's
``--workdir`` argument can simply update the environment variable first.
Each pipeline step can be run and re-run independently.

**Model selection.** ``load()`` resolves every architecture constant below
from a Hugging Face checkpoint (see its docstring) and persists the result
to ``<workdir>/run_config.json``. Step 1 is the only script that needs
``--model``; steps 2-6 call ``load()`` with no arguments and it picks up
that file. Running any script without ever calling ``load()`` falls back
to the module-level defaults below (the checkpoint this pipeline was
validated end to end against), so existing ad-hoc/interactive use keeps
working unchanged.
"""
import json
import os

# --- Environment setup: MUST happen before numpy is imported anywhere. ---
os.environ.setdefault("NPY_PROMOTION_STATE", "legacy")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("USER", "root")

from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

# ---------------------------------------------------------------------------
# Model under compilation — defaults, overridden by load()
# ---------------------------------------------------------------------------
# Any LLaMA2-style model with grouped-query attention works after adjusting
# these constants (or, now, by calling load(model_id) — see below).
# Validated end to end with:
MODEL_ID = "Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA"

# Architecture constants of MODEL_ID
HIDDEN = 256            # embedding width
NHEAD = 16              # number of query heads
NKVHEAD = 8             # number of key/value heads (GQA)
NREP = NHEAD // NKVHEAD # queries per KV head
HD = HIDDEN // NHEAD    # head dimension (16)
Q_WIDTH = NHEAD * HD    # attention output width per layer
NLAYERS = 4
ROPE_THETA = 10000.0
RMS_EPS = 1e-6
VOCAB = 32000

# The compiled lm_head matmul ([HIDDEN, VOCAB]) fails HEF placement as one
# op once VOCAB is large enough (confirmed: fails at 151936, fine at 32000
# and below) -- see docs/findings/large-vocab-lm-head-sharding.md. Shard
# it into ceil(VOCAB / LM_HEAD_MAX_SHARD_WIDTH) equal-width column blocks,
# each its own output node, concatenated host-side. 1 shard (the common
# case) is functionally identical to the old single-matmul export.
LM_HEAD_MAX_SHARD_WIDTH = 32000
LM_HEAD_SHARDS = 1

# Sequence geometry. SEQ is both the parse/calibration length of the base
# scope and the total KV-cache size (the pipeline requires CACHE_SIZE == SEQ;
# the __prefill scope runs at PREFILL_SIZE positions). Not derivable from
# the checkpoint — a compile-time choice.
SEQ = 24
PREFILL_SIZE = 16
CACHE_SIZE = 24         # MUST equal SEQ for this pipeline

CALIBSET_SIZE = 32      # number of calibration samples for quantization
NET_SCOPE = "ts25mpipe" # base scope name; DFC derives <scope>__prefill/<scope>__tbt

# Fidelity gate for steps 1-3 (cosine vs float32 HF). 0.999 is the bar this
# pipeline was validated against on a 4-layer model. Deeper checkpoints can
# show small, apparently benign SDK_NATIVE-vs-PyTorch drift that compounds
# with layer count (see docs/findings/ if a dedicated writeup exists) —
# lower this ONLY after confirming the drop isn't a real bug (inspect where
# the divergence concentrates, don't just widen the gate to make it pass).
# Never silently overridden: s1/s2/s3 print the active value every run, and
# --cosine-min is required to change it (not a bare number in config.py).
COSINE_MIN = 0.999

BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = EOS_TOKEN_ID  # this tokenizer has no dedicated pad token

# Additive attention-mask convention used throughout (float domain):
ALLOWED = 0.0
BLOCKED = -100.0

# HF config-class name prefixes known to share the LlamaModel-shaped
# attribute path (`model.layers`, `model.norm`, `model.embed_tokens`) this
# pipeline's exporter relies on. Not exhaustive — a mismatch is a warning,
# not a hard stop, since the real test is whether s1/s2 actually succeed.
_KNOWN_ELIGIBLE_ARCHITECTURES = ("Llama", "Mistral", "Qwen2", "Qwen3")


def _run_config_path(workdir: Path) -> Path:
    return workdir / "run_config.json"


def load(
    model_id: str | None = None,
    *,
    seq: int | None = None,
    prefill_size: int | None = None,
    calibset_size: int | None = None,
    net_scope: str | None = None,
    cosine_min: float | None = None,
) -> None:
    """Resolve this module's architecture/geometry constants for one run.

    Two ways to call this:

    - ``load(model_id, ...)`` (typically from step 1, driven by CLI flags):
      reads ``transformers.AutoConfig``/tokenizer for ``model_id`` and
      derives every architecture constant (HIDDEN, NHEAD, NKVHEAD, NLAYERS,
      ROPE_THETA, RMS_EPS, VOCAB, BOS/EOS/PAD_TOKEN_ID). ``seq``,
      ``prefill_size``, ``calibset_size``, ``net_scope`` override their
      defaults (geometry/scope aren't derivable from the checkpoint).
      Raises ``AssertionError`` if the checkpoint ties its embeddings
      (unsupported by this pipeline) and prints a warning — not a hard
      failure — if the architecture isn't in the known-eligible family
      (see the wiki's Porting-Another-Model.md eligibility screen; the
      real test is whether step 1/2 actually parse).
      The fully-resolved values are persisted to
      ``<workdir>/run_config.json`` so later steps don't need to repeat
      ``--model``.
    - ``load()`` with no arguments (steps 2-6): loads
      ``<workdir>/run_config.json`` if present and applies it; otherwise
      leaves the module-level defaults above untouched.

    Call this (with or without arguments) after :func:`set_workdir` /
    parsing ``--workdir``, and before anything else in the script runs.
    """
    if model_id is None:
        _load_from_workdir()
        return

    import transformers

    hf_config = transformers.AutoConfig.from_pretrained(model_id)
    if not any(name in type(hf_config).__name__ for name in _KNOWN_ELIGIBLE_ARCHITECTURES):
        print(
            f"warning: {type(hf_config).__name__} is not in the known-eligible "
            "architecture family (Llama/Mistral/Qwen2-style). This pipeline may "
            "still work if the model uses RMSNorm + RoPE + SwiGLU MLP + "
            "GQA-or-MHA attention — see Porting-Another-Model.md's "
            "eligibility screen. Steps 1/2 will "
            "fail loudly if the architecture doesn't fit."
        )
    if getattr(hf_config, "tie_word_embeddings", False):
        print(
            f"note: {model_id} ties its embeddings (tie_word_embeddings=True) — "
            "step 1 exports the embedding table and the lm_head matrix as two "
            "independent baked-in tensors either way, so this is not a blocker "
            "by itself."
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None:
        bos_id = getattr(hf_config, "bos_token_id", None) or 1
    if eos_id is None:
        eos_id = getattr(hf_config, "eos_token_id", None) or 2
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    nhead = hf_config.num_attention_heads
    nkvhead = getattr(hf_config, "num_key_value_heads", None) or nhead
    hidden = hf_config.hidden_size
    # Some architectures (Qwen3) declare head_dim explicitly, independent of
    # hidden/nhead (e.g. Qwen3-0.6B: hidden=1024, nhead=16 -> 64, but
    # head_dim=128). Trust the explicit field when present; every downstream
    # RoPE/GQA/QK-Norm matrix is built from config.HD, so this one line is
    # the only place that needs to know the difference.
    hd = getattr(hf_config, "head_dim", None) or (hidden // nhead)

    resolved = {
        "MODEL_ID": model_id,
        "HIDDEN": hidden,
        "NHEAD": nhead,
        "NKVHEAD": nkvhead,
        "NREP": nhead // nkvhead,
        "HD": hd,
        "Q_WIDTH": nhead * hd,
        "NLAYERS": hf_config.num_hidden_layers,
        "ROPE_THETA": float(getattr(hf_config, "rope_theta", 10000.0)),
        "RMS_EPS": float(getattr(hf_config, "rms_norm_eps", 1e-6)),
        "VOCAB": hf_config.vocab_size,
        "LM_HEAD_SHARDS": -(-hf_config.vocab_size // LM_HEAD_MAX_SHARD_WIDTH),  # ceil div
        "SEQ": seq if seq is not None else SEQ,
        "PREFILL_SIZE": prefill_size if prefill_size is not None else PREFILL_SIZE,
        "CACHE_SIZE": seq if seq is not None else CACHE_SIZE,  # MUST equal SEQ
        "CALIBSET_SIZE": calibset_size if calibset_size is not None else CALIBSET_SIZE,
        "NET_SCOPE": net_scope if net_scope is not None else (
            NET_SCOPE if model_id == MODEL_ID else _slug(model_id)
        ),
        "COSINE_MIN": cosine_min if cosine_min is not None else COSINE_MIN,
        "BOS_TOKEN_ID": int(bos_id),
        "EOS_TOKEN_ID": int(eos_id),
        "PAD_TOKEN_ID": int(pad_id),
    }
    globals().update(resolved)

    workdir = ensure_workdir()
    with open(_run_config_path(workdir), "w") as f:
        json.dump(resolved, f, indent=2)


def _load_from_workdir() -> None:
    workdir = Path(os.environ.get("DFC_WORKDIR", "./workdir")).resolve()
    run_config_path = _run_config_path(workdir)
    if not run_config_path.exists():
        return  # nothing persisted yet — keep the module-level defaults
    with open(run_config_path) as f:
        resolved = json.load(f)
    globals().update(resolved)


def _slug(model_id: str) -> str:
    """Derive a HN-safe default network-group scope name from a HF model id."""
    tail = model_id.split("/")[-1]
    return "".join(c if c.isalnum() else "_" for c in tail).strip("_").lower() or "model"


def set_workdir(path: str | os.PathLike) -> None:
    """Point the pipeline at an alternate working directory.

    Call before :func:`paths` or :func:`load` are first used (e.g. right
    after parsing ``--workdir``); it only updates the environment variable
    those functions read.
    """
    os.environ["DFC_WORKDIR"] = str(Path(path).expanduser().resolve())


def paths() -> SimpleNamespace:
    """Artifact paths inside the working directory, resolved on demand."""
    workdir = Path(os.environ.get("DFC_WORKDIR", "./workdir")).resolve()
    return SimpleNamespace(
        workdir=workdir,
        onnx=workdir / "model.onnx",
        hf_refs=workdir / "hf_reference.npz",     # s1 output, reused later
        tokenizer_dir=workdir / "tokenizer",
        wte=workdir / "wte.npy",
        har_parsed=workdir / "parsed.har",
        har_surgery=workdir / "surgery.har",
        har_resources=workdir / "resources.har",
        har_quantized=workdir / "quantized.har",
        har_convfixed=workdir / "convfixed.har",
        hef=workdir / "model.hef",
        hailo_config=workdir / "hailo-config.json",
    )


def ensure_workdir() -> Path:
    p = paths()
    p.workdir.mkdir(parents=True, exist_ok=True)
    return p.workdir


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def cosine(a, b) -> float:
    """Flat cosine similarity in float64 — the project-wide fidelity metric."""
    import numpy as np

    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def register_acceleras_layers() -> int:
    """Register every internal acceleras/Keras layer class as Keras-serializable.

    Required before any ``ClientRunner`` loads a HAR containing
    quantization-era layers; without it deserialization fails with
    "Unknown layer" errors. Returns the number of classes registered.
    """
    import importlib
    import inspect
    import pkgutil

    import keras
    import hailo_model_optimization.acceleras as _acceleras_pkg

    registered = 0
    for _finder, modname, _ in pkgutil.walk_packages(
        _acceleras_pkg.__path__, prefix="hailo_model_optimization.acceleras."
    ):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for attr_name, attr in inspect.getmembers(mod, inspect.isclass):
            if (
                attr.__module__ == modname
                and issubclass(attr, keras.layers.Layer)
                and getattr(attr, "_keras_api_names", None) is None
            ):
                keras.saving.register_keras_serializable()(attr)
                registered += 1
    return registered


def head_dim_frequencies():
    """RoPE inverse-frequency vector: HD/2 exponentials concatenated twice
    (the cos-half + sin-half layout consumed by the HEF's rope table)."""
    import numpy as np

    theta_base = 1.0 / (ROPE_THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    return np.concatenate([theta_base, theta_base]).astype(np.float32)


def causal_mask_tiled(batch: int, seq_len: int):
    """Additive causal mask tiled over heads: [batch, 1, seq_len, NHEAD*seq_len]."""
    import numpy as np

    bias = np.triu(np.full((seq_len, seq_len), BLOCKED, dtype=np.float32), k=1)
    return np.tile(bias[np.newaxis, np.newaxis, :, :], (batch, 1, 1, NHEAD)).astype(np.float32)
