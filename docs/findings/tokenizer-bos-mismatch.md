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

## Server side: what the genai server actually does with tokenizer.json

Once KV-cache generation was fixed ([tbt-cache-read.md](tbt-cache-read.md)),
hailo-ollama still diverged from host-driven runs in two ways, both traced
to how the server consumes the embedded `tokenizer.json`:

1. **The post-processor is ignored.** LLaMA-style tokenizers add BOS through
   a `TemplateProcessing` post-processor; the server never applies it, so
   prompts reached the model without BOS. Evidence: the same prompt through
   hailo-ollama gave `. Dad, Little John was`; prefixing the prompt with the
   literal `<s>` gave `. She was always looking for something` — the
   host-driven continuation's first four tokens.
2. **Detokenization is token by token.** The LLaMA decoder ends with a
   `Strip` of one leading space, which is right for a whole sequence but
   removes the space in front of *every* word when each token is decoded
   alone: output came back as `Shewasalwayslookingfor…`.

**Fix** ([s3_surgery_and_resources.py](../../pipeline/s3_surgery_and_resources.py),
`server_tokenizer()`): the BOS text of the post-processor's template (if
any) is prepended to the `chat_template` in `hailo-config.json`, and
`Strip` steps are removed from the decoder of the embedded
`tokenizer.json`. Tokenizers without a leading special token (e.g.
Qwen-style byte-level BPE) are left unchanged. Not yet checked: multi-turn
conversations — if the server re-applies the template to each new message,
BOS would be re-inserted at every turn.

## Note

This finding explains several historical "same prompt, different behavior"
episodes before it was identified. If you adapt this pipeline to another
model, verify its tokenizer's add-BOS behavior once and write it down.
