# Finding 6 — tokenizer/BOS mismatch shifts every position

**Status: understood and worked around; worth re-checking for any new model.**

## Symptom

Generation quality differed depending on *which component tokenized the
prompt* (host-side Python tokenizer vs the HEF's embedded `tokenizer.json`
consumed by the server), even with identical text.

## Investigation

Diffing tokenizations showed two distinct sources of divergence:

1. **BOS handling.** Some tokenization paths prepend a beginning-of-sequence
   token automatically, others do not. A one-token shift changes every RoPE
   position, every mask row, and which cache slot each token occupies —
   a one-off-by-one that degrades everything downstream.
2. **Vocabulary identity.** The embedded `tokenizer.json` must be exactly
   the model's own — not a same-family lookalike. Piece-for-piece identity
   was confirmed by decoding embedded-vocab ids against both.

## Root cause

Position-sensitive models have no tolerance for prompt-prefix ambiguity: an
off-by-one at position 0 is not cosmetic, it re-indexes the whole sequence.

## Workaround / rules

- Pin the convention explicitly in every tool that builds inputs:
  prompts start with BOS (id 1 here), generation stops on EOS (id 2);
- When comparing host-driven runs to server-driven runs, dump the server's
  effective token ids first (its logs include them) and assert equality;
- The diagnostics helpers
  ([runtime_inputs.py](../../runtime/diagnostics/runtime_inputs.py)) take
  explicit id lists — no implicit re-tokenization anywhere below genai.

## Note

This finding explains several historical "same prompt, different behavior"
episodes before it was identified. If you adapt this pipeline to another
model, verify its tokenizer's add-BOS behavior once and write it down.
