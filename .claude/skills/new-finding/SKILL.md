---
name: new-finding
description: Write a new docs/findings/ page for this repo — use when a bug's root cause is understood (fixed or still open) and needs to be recorded so the next person doesn't re-derive it.
---

# Writing a finding page

A finding page is the unit of knowledge transfer in this repo. It exists
so the next investigation (yours in six months, or someone else's) starts
from the conclusion instead of from zero.

## Shape (fixed, don't deviate)

1. **Symptom** — what was observed, in terms a reader can reproduce
   (error message, numeric mismatch, behavior).
2. **Investigation** — what was tried, including what did *not* work and
   what it ruled out. Negative results are as valuable as the fix.
3. **Root cause** — the actual mechanism, backed by a source pointer
   (public HailoRT/DFC source, header, or structural comparison) wherever
   possible.
4. **Fix** — the concrete change, with a file:line pointer into
   `pipeline/` or `runtime/` if one exists.
5. **Verification** — the reproducible evidence: cosine similarity,
   argmax match, structural diff of `.hn`/HEF layout. A finding without a
   number attached is a hypothesis, not a finding.

If the bug is still open, write the same shape minus Fix, and add a
"Current best hypothesis" + "Reproduce it" section (see
[open-tbt-cache-read.md](../../../docs/findings/open-tbt-cache-read.md)
for the template to copy).

## Rules while writing

- **No dates or timelines** — the repo is evergreen.
- **No proprietary excerpts** — describe behavior and quote short
  identifiers (function/key names); don't paste long blocks of
  proprietary source. Link the public repo instead.
- **No official HEFs/`.alls`/`.hn` graphs, no firmware, no DFC wheels** —
  neither pasted inline nor referenced as attached files.
- Claims need evidence of a kind listed in
  [docs/findings/index.md](../../../docs/findings/index.md)'s "How to read
  these" section — structural comparison, public source reading, or
  on-hardware numerics.

## After writing

1. Add a row to the summary table in
   [docs/findings/index.md](../../../docs/findings/index.md).
2. Update [docs/status.md](../../../docs/status.md) if the finding changes
   what works or what doesn't at the stage-by-stage level.
3. If the finding closes an item in
   [Investigation-Chronicle](../../../../wiki/Investigation-Chronicle) or
   opens a new one, note it there too — the wiki narrates, the repo is
   canonical (docs/ wins on conflict).
4. Run the [`preflight`](../preflight/SKILL.md) skill before committing.
