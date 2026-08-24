# hailo-10h-llm-compiler

Compile **your own** small LLM into a Hailo-10H HEF and run it with
[hailo-ollama](docs/references.md#hailo-software) — end to end, from a
Hugging Face checkpoint to a self-contained, servable binary.

This repository documents and packages an experimental, reverse-engineered
methodology for using the Hailo Dataflow Compiler (DFC) **LLM compilation flow** — the same
flow Hailo uses internally to produce its official generative-AI HEFs — on a
model of your choice. It was built and validated on
[Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA](https://huggingface.co/Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA),
a 25M-parameter LLaMA2-style model with grouped-query attention, small enough
to iterate quickly yet structurally identical to modern production LLMs
(RMSNorm, SwiGLU, RoPE, GQA, KV-cache).

> ⚠️ **Highly experimental — do not expect it to work.** This project was
> built by reverse-engineering an undocumented compiler flow against one
> pinned SDK version (DFC 5.3.0) and one 25M-parameter model. Nothing here is
> supported, stable, or guaranteed: steps can fail mid-way, APIs move between
> toolchain versions, and multi-token generation still produces degraded text
> ([one open issue](docs/findings/open-tbt-cache-read.md)). Read
> [docs/status.md](docs/status.md) for exactly what works today — and expect
> to debug everything else yourself. Everything needed to reach the current
> state is in this repository; getting further is on you.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Running the compiled model](#running-the-compiled-model)
- [Documentation](#documentation)
- [Current limitations](#current-limitations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Why this exists

The Hailo-10H ships with official pre-compiled LLM HEFs, but the toolchain
that produces them — the LLM branch of the Hailo Dataflow Compiler — is
barely documented. The public documentation describes the flow at a high
level; nothing explains how to bring *your own* Hugging Face model through
it, and the runtime (`genai.LLM`, hailo-ollama) rejects any HEF that does not
match its exact expectations.

This project reverse-documented that path by compiling a small open model
from scratch. Along the way it identified and fixed **five distinct
incompatibilities** between a vanilla Hugging Face export and what the
Hailo-10H LLM stack requires (missing `lm_head`, output-shape constraints,
RoPE input widths, attention-mask broadcasting, a `hailo-config.json` key
mismatch) — each is documented in
[`docs/findings/`](docs/findings/index.md) so you do not have to rediscover
them.

**What you get at the end:** a self-contained HEF (embeddings, tokenizer and
generation config embedded in the binary) that hailo-ollama can register and
serve exactly like an official model.

**What this is not:** a redistribution of Hailo's proprietary software. The
Dataflow Compiler wheel and the device firmware are **not** included and
**not** redistributable; [docker/README.md](docker/README.md) explains how to
obtain them from Hailo's developer portal yourself.

## How it works

```
 Hugging Face checkpoint (safetensors)
        │
        ▼
 s1_export_onnx.py      PyTorch → ONNX (opset 17), with the five
        │               Hailo-specific graph fixes applied
        ▼
 s2_parse_har.py        ONNX → HAR (DFC parser, hailo10h target)
        │
        ▼
 s3_surgery_resources.py  HAR graph surgery: RoPE/mask input wiring,
        │                 then attach external resources (embeddings,
        │                 tokenizer, rope table, hailo-config.json)
        ▼
 s4_optimize_kvcache.py   Quantization with KV-cache duplication
        │                 (INT4 weights / INT8 activations, a16 embeddings)
        ▼
 s5_fix_convolutions.py   Post-quantization shape-consistency pass
        │
        ▼
 s6_compile_hef.py      HAR → HEF (two network groups: __prefill + __tbt)
        │
        ▼
 register_hailo_ollama.py   Publish into hailo-ollama's model store
        │
        ▼
 hailo-ollama serve / hailo-ollama run <family>:<tag>
```

The model graph itself is duplicated by the compiler into two network
groups: `__prefill` (processes the whole prompt at once) and `__tbt`
(token-by-token, reading and writing the KV-cache). This duplication is
driven by `set_kv_cache_global_params()` — the entry point of the DFC LLM
flow.

## Repository layout

```
hailo-10h-llm-compiler/
├── docker/          Dockerfiles: NVIDIA CUDA base, AMD ROCm base, Jupyter
├── pipeline/        The six compile steps + shared config (s1 → s6)
├── notebooks/       The same compile chain and the diagnostics tooling,
│                    as self-contained executable walkthroughs
├── runtime/         Device-side tools: hailo-ollama registration,
│                    generation, and low-level diagnostics
├── docs/
│   ├── terminology.md        Vocabulary: HAR, HEF, scope, KV-cache, ...
│   ├── device-setup.md       Host + device bring-up (drivers, HailoRT, ...)
│   ├── status.md             What works / what does not, in detail
│   ├── references.md         Every external link used by this project
│   └── findings/             The reverse-engineering findings, one page each
├── .claude/skills/  Working playbooks for continuing the investigation
│                    (writing a finding, preflight checks, device-level
│                    debugging) — see CLAUDE.md
├── CLAUDE.md        Ground rules + working patterns for picking this up
├── LICENSE          MIT
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

## Getting started

### 0. Prerequisites

| Component | Notes |
|---|---|
| Hailo-10H device | PCIe M.2 card or SoM, with working drivers (see [docs/device-setup.md](docs/device-setup.md)) |
| Hailo Dataflow Compiler 5.3.0 | Proprietary, free account required — see [docker/README.md](docker/README.md) |
| NVIDIA GPU (CUDA) **or** AMD GPU (ROCm) — **optional** | Speeds up the quantization step only (CPU-only works but is much slower). NVIDIA: Turing (sm_75) or newer. AMD: gfx906 is what we tested — nothing older verified |
| Docker | All tooling runs in containers; nothing is installed on the host |
| ~30 GB disk | DFC image ≈ 15 GB, working directory for artifacts |

> **Memory note:** the DFC optimization step is memory-hungry. On a 32 GB
> host, cap the container (e.g. `--memory=24g`) — see
> [docs/device-setup.md](docs/device-setup.md#memory-limits).

### 1. Build the toolchain image

```bash
# Place the DFC wheel in ./docker/ first (see docker/README.md)
docker build -f docker/Dockerfile.nvidia -t dfc-nvidia:5.3.0 docker/
# or, for AMD GPUs:
docker build -f docker/Dockerfile.amd -t dfc-amd:5.3.0 docker/
```

### 2. Run the pipeline

```bash
git clone https://github.com/l-nmch/hailo-10h-llm-compiler.git
cd hailo-10h-llm-compiler

docker run --rm -it --gpus all \
    -v "$PWD":/repo -v "$PWD/workdir":/workdir \
    dfc-nvidia:5.3.0 bash
# inside the container:
export DFC_WORKDIR=/workdir
python /repo/pipeline/s1_export_onnx.py
python /repo/pipeline/s2_parse_har.py
python /repo/pipeline/s3_surgery_and_resources.py
python /repo/pipeline/s4_optimize_kvcache.py
python /repo/pipeline/s5_fix_convolutions.py
python /repo/pipeline/s6_compile_hef.py
```

Each step reads and writes only `DFC_WORKDIR` (default `./workdir`) and can
be re-run independently. To compile a different model, edit
[`pipeline/config.py`](pipeline/config.py) — every architectural constant
(hidden size, heads, RoPE theta, sequence lengths…) lives there.

> Prefer a single interactive run? [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb)
> executes the same chain end to end, with every step's logic unfolded in
> cells and validation gates between steps — no scripts required.

### 3. Deploy to the device and serve

On the host that has the Hailo-10H device (with HailoRT and hailo-ollama
installed — [docs/device-setup.md](docs/device-setup.md)):

```bash
python runtime/register_hailo_ollama.py --hef workdir/model.hef \
    --family tinystories25m --tag my-first-model
hailo-ollama run tinystories25m:my-first-model
```

## Running the compiled model

Three complementary ways, in decreasing order of "how much magic":

1. **hailo-ollama** — the end goal. `register_hailo_ollama.py` publishes the
   HEF into hailo-ollama's content-addressed model store with a manifest;
   the model then behaves like any official one.
2. **`genai_generate.py`** — direct `genai.LLM` Python API generation, with
   subprocess-isolated retry logic that absorbs a known host-side
   interrupt bug (see
   [docs/findings/sdk-behavior-notes.md](docs/findings/sdk-behavior-notes.md)).
3. **`runtime/diagnostics/`** — low-level `InferModel` probes (manual
   prefill + token-by-token with hand-built masks and RoPE tables) used to
   debug the runtime contract layer by layer. Start with
   [`runtime/diagnostics/README.md`](runtime/diagnostics/README.md).

## Documentation

| Document | Content |
|---|---|
| [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb) | The full HF → ONNX → HAR → surgery → quantization → HEF chain as one executable notebook |
| [notebooks/diagnostics.ipynb](notebooks/diagnostics.ipynb) | Wire-contract reconstruction (mask/RoPE/encodings), offline HEF audit, device probes |
| [docs/terminology.md](docs/terminology.md) | HAR vs HEF, network scopes, `__prefill`/`__tbt`, KV-cache mechanics, quantization flavors |
| [docs/device-setup.md](docs/device-setup.md) | PCIe driver, HailoRT, hailo-ollama installation; package sources; memory limits |
| [docs/status.md](docs/status.md) | Honest, detailed state of every pipeline stage and runtime path |
| [docs/references.md](docs/references.md) | All external resources: model zoo entry, hailort source, blog posts, papers |
| [docs/findings/index.md](docs/findings/index.md) | Index of all reverse-engineering findings (the five compile fixes + SDK behavior + the open issue) |

## Current limitations

- **KV-cache generation quality** — the open issue: cache reads during
  token-by-token inference return truncated tensors (~30% of columns
  structurally zeroed), degrading multi-token coherence. Prefill is exact.
  Details and everything tried so far:
  [docs/findings/open-tbt-cache-read.md](docs/findings/open-tbt-cache-read.md).
- **Small models only, so far** — validated on a 25M model; nothing
  structural prevents larger ones, but calibration memory and compile time
  grow quickly.
- **Single model family** — the pipeline hardcodes a LLaMA2-style
  architecture in `pipeline/config.py` + `s1_export_onnx.py`. Porting to
  other architectures (GPT-2-style, Phi-style) means adapting the export
  script; the compile steps are architecture-agnostic.
- **DFC version pinning** — everything here is validated against DFC 5.3.0.
  The LLM flow is young and its APIs move; expect breakage on other
  versions.

## Contributing

Contributions are welcome — especially on the
[open issue](docs/findings/open-tbt-cache-read.md). See
[CONTRIBUTING.md](CONTRIBUTING.md) for the workflow and the proprietary-
material policy (short version: never commit DFC wheels, official HEFs, or
anything derived from them).

## License

[MIT](LICENSE) — for everything in this repository. Note that Hailo's own
components have different licenses (HailoRT core is MIT, the kernel driver
is GPL-2.0, the Dataflow Compiler is proprietary); see
[docs/references.md](docs/references.md) for the breakdown.

## Acknowledgements

- [Mxode](https://huggingface.co/Mxode) for the excellent family of tiny
  LLaMA2 models this was validated on
  ([TinyStories-LLaMA2-25M-256h-4l-GQA](https://huggingface.co/Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA)).
- Hailo for the Hailo-10H hardware, the
  [public HailoRT source](https://github.com/hailo-ai/hailort) (MIT), and
  the ["Bringing Generative AI to the Edge"](https://hailo.ai/blog/bringing-generative-ai-to-the-edge-hailo-10h-llm-compiler/)
  blog series that documents the LLM compiler flow.
- The [TinyStories](https://arxiv.org/abs/2305.07759) authors (Microsoft
  Research) — small models that generate real English are the ideal testbed
  for edge LLM compilation.
