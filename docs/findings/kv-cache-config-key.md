# Finding 5 — `prefill_input_tokens_count`, not `_size`

**Status: fixed (pipeline step 3).**

## Symptom

A HEF that was structurally correct still behaved as if the prefill length
were wrong: prompts longer than some threshold mis-generated, in ways
sensitive to prompt length rather than content.

## Investigation

Reading the public HailoRT LLM server source (`llm_server.cpp`) shows which
keys of the embedded `hailo-config.json` are consulted when configuring
pre-processing. The prefill length key is read as:

```json
{"pre_process_params": {"prefill_input_tokens_count": 16}}
```

An earlier draft of our config used `prefill_input_tokens_size`. The server
never reads that name; on a missing key it silently falls back to its
hardcoded default (96).

## Root cause

Key-name mismatch with silent fallback. Nothing errors — the server just
uses a default tuned for someone else's model, producing length-dependent
misbehavior that looks like a model bug.

## Fix

Emit `prefill_input_tokens_count` (and `kv_cache_size`,
`num_attention_heads`, `num_key_value_heads`) under `pre_process_params`.
See [../../pipeline/s3_surgery_and_resources.py](../../pipeline/s3_surgery_and_resources.py)'s
`build_hailo_config()` for the full document shape.

## Tooling

For HEFs already compiled:
[runtime/diagnostics/hotpatch_hailo_config.py](../../runtime/diagnostics/hotpatch_hailo_config.py)
rewrites the embedded config in place (appends new bytes, repoints the
resource descriptor, refreshes the xxhash signature) without recompiling.
And [hef_audit.py](../../runtime/diagnostics/hef_audit.py) decodes any
HEF's embedded config so you can check which keys yours carries.

## Lesson

When a runtime contract includes a JSON sidecar, diff your keys against what
the *server source* reads, not against what feels natural. Silent defaults
are indistinguishable from working configuration.
