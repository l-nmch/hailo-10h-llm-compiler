---
name: device-debug
description: Low-level Hailo-10H device debugging playbook — use when a client-side error (timeout, generic failure code, hang) doesn't explain itself and you need to see what the device is actually doing.
---

# Device-side debugging playbook

The most common trap in this project: a generic client-side error
(`HAILO_TIMEOUT`, a hang, a vague failure code) hides a specific,
immediately-legible error that only exists in the device's own logs.
Chasing client-visible symptoms (file size, PCIe throughput, retry logic)
before checking device-side logs has cost real time more than once — check
this first, not last.

## Why the client alone can lie to you

The Hailo-10H runs its own firmware with an embedded mini-Linux (visible
in `dmesg` as `u-boot-spl.bin` / `fitImage` / `image-fs` loaded by the PCIe
driver at boot). The "server" side of any client/server exchange
(`hailort.log` talks about "sending N chunks to server") is that firmware,
not a host process. It can fail immediately and specifically, then have
its *own* generic timeout/error path fire seconds later — which is the
only thing the client ever reports. If you only read the client log, you
are debugging the echo, not the cause.

## The technique

```bash
hailortcli logs runtime          # LLM/inference server logs — start here
hailortcli logs system_control    # control-plane / configuration logs
hailortcli logs nnc                # neural network core logs
```

Run this immediately after a failure, before forming hypotheses from
client-side symptoms alone. Look for the *first* CHECK/error line, not the
last — the device's own error-reporting path can itself time out and add
a second, misleading failure a few seconds later.

## General elimination order for opaque failures

1. **Device-side logs first** (above) — often gives the exact failing
   check and file:line in one shot.
2. **Structural comparison** against a known-good official artifact
   (`hef_audit.py`, `parse-hef`) — rules in/out the artifact itself before
   blaming the runtime.
3. **Client-side logs** (`HAILORT_CONSOLE_LOGGER_LEVEL=debug`,
   `hailort.log`) — timing and chunking, useful once you know what to
   look for.
4. **Version alignment** — driver, HailoRT, and server packages must come
   from the same software-suite drop; mixed versions fail in confusing
   ways. Check all three before deep-diving logic.

## Things that look like this class of bug but aren't

- **Interrupt/session failures on long-lived processes** — environmental,
  not artifact-specific; a retry in a *fresh process* succeeds. See
  [Runtime-Pitfalls](../../../../../wiki/Runtime-Pitfalls) — mitigated by
  subprocess isolation, already implemented in
  [`genai_generate.py`](../../../runtime/genai_generate.py).
- **Tokenizer/BOS mismatches** — produce quality/coherence symptoms that
  look like model or cache bugs; see
  [tokenizer-bos-mismatch.md](../../../docs/findings/tokenizer-bos-mismatch.md).

## What never helps

Patching a runtime binary's data section to change a `constexpr` timeout
does nothing — build-time constants get inlined as immediates at every
call site, so the patched `.rodata` symbol is never read at runtime. Don't
spend time on binary patching; read the public headers/source instead
([hailort public source](https://github.com/hailo-ai/hailort)). And never
patch/flash the device's own firmware — see the ground rules in
[CLAUDE.md](../../../CLAUDE.md).
