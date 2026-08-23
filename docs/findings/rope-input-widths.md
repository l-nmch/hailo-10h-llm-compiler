# Finding 3 — RoPE inputs are asymmetrically tiled (K ≠ Q widths)

**Status: fixed (pipeline step 3, surgery 1/2).**

## Symptom

Positional information was subtly wrong on hardware while everything
validated in simulation: outputs looked "almost right" — the worst kind of
wrong.

## Investigation

What does the runtime actually write into `input_layer3-6` at run time?
Structural inspection of an official LLM HEF's native graph
(the `.native.hn` inside official HEFs, readable with
[hef_audit.py](../../runtime/diagnostics/hef_audit.py)) shows the four RoPE
inputs declared at **asymmetric** final widths:

- K cos/sin: `theta_size × num_key_value_heads` (= 8×16 = 128 here)
- Q cos/sin: `theta_size × num_attention_heads` (= 16×16 = 256 here)

i.e. each input already carries one full cos/sin row *per head*, not a
single shared row.

Our parsed graph instead declared all four inputs uniform-width
(`theta_size` = 16) and let the parser insert small duplication convolutions
(`conv1-4`) downstream to widen them. Two problems:

1. The duplication conv tiled with period-16 semantics — exactly the host's
   `tile_along_last_axis`, meaning the tiling happened **twice**: once
   host-side by the runtime writing wide buffers, once on-chip by the conv.
2. The declared input shapes did not match what the runtime writes, so any
   consumer assuming the declared shape was reading the wrong layout.

## Root cause

Declared input width vs actual runtime buffer width mismatch, compounded by
a redundant on-chip duplication pass.

## Fix

Delete the four duplication convs; re-declare `input_layer3-6` directly at
their final tiled widths; reconnect consumers to the input layers. The
external theta resource is attached with per-group tile counts so the
device computes cos/sin for all heads itself:

```python
{"theta": theta_table,
 "tile": np.array([1, 1, n_kv_heads or n_heads], dtype=np.int32),
 "factor": np.array([1.0], dtype=np.float32)}
```

## Verification

Post-surgery SDK_NATIVE cosine returned to exactly 1.000000, and on
hardware the position-dependence of outputs matched the float32 reference.
The full mask/RoPE input construction used to validate this on-device lives
in [../../runtime/diagnostics/runtime_inputs.py](../../runtime/diagnostics/runtime_inputs.py).
