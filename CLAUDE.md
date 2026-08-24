# Working in this repository

This file is for any agent (human or AI) picking up the investigation.
Read [docs/status.md](docs/status.md) first for the current state, then
this page for how to work without breaking the project's invariants.

## Non-negotiable ground rules

- **The chip runs AS-IS.** Read/decompile/probe freely; never patch or
  flash a modified firmware or driver image, even reversibly. The entire
  point of this project is compiling a self-made model for the *stock*
  device.
- **Nothing installed on the host.** All tooling — DFC, PyTorch, HailoRT
  clients — runs inside the Docker images in [docker/](docker/). Don't
  `pip install` or `apt-get` on bare metal.
- **One command, one container.** Each independent DFC/PyTorch invocation
  gets its own `docker run --name <x>`; don't `docker exec` into a shared
  long-lived container — state becomes unreadable and non-reproducible.
  Read results back via `docker logs`.
- **Proprietary material policy** — see [CONTRIBUTING.md](CONTRIBUTING.md).
  No DFC wheels, official HEFs/`.alls`/`.hn`, firmware images, or long
  verbatim excerpts of proprietary source, ever, in any commit.
- **No dates or timelines** anywhere in committed content — keeps the repo
  evergreen and diff-stable ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Working patterns that avoid known traps

- **One persistent session for serving**, subprocess-per-attempt for
  unattended/CI generation — see
  [Runtime-Pitfalls](../../wiki/Runtime-Pitfalls) for why (interrupt/ioctl
  failure mode). Never open a fresh VDevice/LLM per attempt in a serving
  loop; use `clear_context()`.
- **GPU quantization: try plain GPU before `use_saitama=True`.** The
  Saitama PyTorch engine exists to unlock AMD/ROCm GPUs (the stock
  optimizer is CUDA/TensorFlow-bound) — see
  [Quantization-Deep-Dive](../../wiki/Quantization-Deep-Dive). Don't reach
  for it on NVIDIA where the default path already works.
- **Numerical comparisons are one batched `infer()` call**, teacher-forced,
  `batch_size == calibset_size`. Never loop token-by-token for a cosine
  check — it's slow and leaks memory for no accuracy gain.
- **Memory-constrained hosts** (small SBCs, or a 32 GB workstation running
  DFC's optimizer): never load a multi-GB file whole into RAM, and never
  run an uncapped `optimize()` — see the memory-discipline note in
  [Runtime-Pitfalls](../../wiki/Runtime-Pitfalls).
- **Degraded/unreachable device state**: recover with a full power cycle
  of the host only. Do not loop software rescans, power-state pokes, or
  driver unbind/rebind over a remote link — this has wedged hosts harder
  every time it was tried.

## Documentation discipline

- Log discoveries **immediately**, in the relevant `docs/findings/` page
  or [docs/status.md](docs/status.md) — not at the end of a session.
  Undocumented discoveries have cost entire sessions to a redo.
- New findings follow the fixed shape: *symptom → investigation → root
  cause → fix → verification*, with reproducible evidence (cosine,
  argmax, structural diff). See the
  [`new-finding`](.claude/skills/new-finding/SKILL.md) skill.
- Before any commit or publish step, run the
  [`preflight`](.claude/skills/preflight/SKILL.md) skill to catch
  proprietary material, dates, and leaked internal paths/hostnames.
- Low-level device debugging (opaque client-side errors, timeouts,
  session failures) — check the
  [`device-debug`](.claude/skills/device-debug/SKILL.md) skill before
  re-deriving a technique from scratch.
