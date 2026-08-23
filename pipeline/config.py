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
"""
import os

# --- Environment setup: MUST happen before numpy is imported anywhere. ---
os.environ.setdefault("NPY_PROMOTION_STATE", "legacy")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("USER", "root")

from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

# ---------------------------------------------------------------------------
# Model under compilation
# ---------------------------------------------------------------------------
# Any LLaMA2-style model with grouped-query attention works after adjusting
# these constants. Validated end to end with:
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

# Sequence geometry. SEQ is both the parse/calibration length of the base
# scope and the total KV-cache size (the pipeline requires CACHE_SIZE == SEQ;
# the __prefill scope runs at PREFILL_SIZE positions).
SEQ = 24
PREFILL_SIZE = 16
CACHE_SIZE = 24         # MUST equal SEQ for this pipeline

CALIBSET_SIZE = 32      # number of calibration samples for quantization
NET_SCOPE = "ts25mpipe" # base scope name; DFC derives <scope>__prefill/<scope>__tbt

BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
PAD_TOKEN_ID = EOS_TOKEN_ID  # this tokenizer has no dedicated pad token

# Additive attention-mask convention used throughout (float domain):
ALLOWED = 0.0
BLOCKED = -100.0


def set_workdir(path: str | os.PathLike) -> None:
    """Point the pipeline at an alternate working directory.

    Call before :func:`paths` is first used (e.g. right after parsing
    ``--workdir``); it only updates the environment variable that
    :func:`paths` reads.
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
