# Finding 2 — output shape must be `[1, 1, vocab]` (last position only)

**Status: fixed (pipeline step 1).**

## Symptom

Even after adding `lm_head`, two problems remained: the compiler struggled
to place the huge matmul ("Agent infeasible" placement failures), and the
output format/order did not match what the runtime expects.

## Investigation

Structural comparison with official Hailo LLM HEFs (via
[hef_audit.py](../../runtime/diagnostics/hef_audit.py)) showed every official
model exposes a single output vstream shaped `[1, 1, vocab]` with NHWC-style
format order. Reading the server flow explains why: genai never predicts
more than one token at a time, always from the last context position —
there is no consumer for full-sequence logits.

## Root cause

Our graph computed and emitted logits for the entire sequence
(`[1, seq, vocab]`, FCR order). That wastes most of the compute, forces a
different format than the runtime reads, and — decisively — multiplies the
lm_head matmul's effective size by the sequence length, which pushed it past
placement feasibility on-chip.

## Fix

Slice to the last position immediately before the projection:

```python
x_last = x[:, -1:, :]   # [1, 1, hidden]
logits = x_last @ Wlm   # [1, 1, vocab]
```

## Consequences

- Output matches the contract: one position, NHWC order.
- The monolithic `hidden × vocab` matmul becomes placeable at
  `compiler_optimization_level=0` (with 16 positions it was not).
- Validation subtlety: comparisons against HF references must use the last
  position only; comparing against padded positions is meaningless.

This finding and [missing-lm-head.md](missing-lm-head.md) are inseparable:
without lm_head the runtime samples garbage; without the slice the graph
often does not compile at all.
