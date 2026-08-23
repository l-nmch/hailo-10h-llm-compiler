# Contributing to hailo-10h-llm-compiler

Thanks for your interest in improving this project! The goal is simple:
make self-compiled LLMs on Hailo-10H reproducible by anyone. Every
contribution that moves toward that goal — code, documentation, or a
negative result — is welcome.

## Ways to contribute

- **Fix or extend the pipeline** ([pipeline/](pipeline/)) — e.g. support
  additional architectures in the step-1 exporter.
- **Attack the open issue** ([docs/findings/open-tbt-cache-read.md](docs/findings/open-tbt-cache-read.md))
  — cache-read truncation during token-by-token generation. Even a
  well-documented failed experiment is valuable; add it to the finding page.
- **Improve diagnostics** ([runtime/diagnostics/](runtime/diagnostics/)) —
  better probes make every future bug cheaper.
- **Documentation** — anything you had to figure out that isn't written
  down yet belongs in [docs/](docs/).

## Development setup

1. Build a toolchain image ([docker/README.md](docker/README.md)) — you
   need your own Dataflow Compiler wheel.
2. Run the pipeline end-to-end once before modifying anything, so you have
   a known-good baseline: all steps green, `workdir/model.hef` produced.
3. Device-side tools run on any host with HailoRT installed
   ([docs/device-setup.md](docs/device-setup.md)).

## Ground rules

### Proprietary material policy (strict)

This repository must remain redistributable. **Never commit:**

- Hailo Dataflow Compiler wheels or any other proprietary Hailo binaries;
- official HEF files, official `.alls` recipes, or `.hn` graphs extracted
  from official HEFs;
- device firmware images;
- long verbatim excerpts of proprietary source code.

Describing behavior, quoting short identifiers (function/key names), and
linking public sources is fine — the findings pages do exactly that.

### No dates or timelines

Do not add build dates, changelog dates, or schedule statements anywhere.
This keeps the repository evergreen and diff-stable.

### Code style

- Python: stdlib + numpy/hailo imports only where already used; type hints
  on public functions; docstrings explaining **why**, not what.
- Scripts are standalone CLIs with `argparse`; shared constants live in
  [pipeline/config.py](pipeline/config.py).
- All artifacts go to `$DFC_WORKDIR` — scripts never write next to the
  repository.
- Match the existing comment density and tone.

### Documentation style

- Findings follow the shape *symptom → investigation → root cause → fix →
  verification*.
- Claims need evidence of one of the kinds listed in
  [docs/findings/index.md](docs/findings/index.md).
- Update [docs/status.md](docs/status.md) when you change what works and
  what doesn't.

## Submitting changes

1. Fork / branch.
2. Make the change, including docs.
3. Verify: pipeline still completes; new claims are backed by a test or a
   documented measurement.
4. Open a pull request describing *what* changed and *how you know it
   works*.

For behavior changes on hardware, include the diagnostic output
(cosines, argmax comparisons) demonstrating the effect.

## Reporting issues

Include: DFC version, HailoRT version, device part number, the exact
command, and the relevant output. For generation-quality reports, include
the prompt ids (not just text) so tokenization differences can be ruled
out.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
