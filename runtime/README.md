# Runtime tools

Everything here runs on the **device host** — the machine with the Hailo-10H
installed, HailoRT user-space libraries and (optionally) hailo-ollama. See
[../docs/device-setup.md](../docs/device-setup.md) for installing that stack.

| Tool | Purpose |
|---|---|
| [register_hailo_ollama.py](register_hailo_ollama.py) | Publish a compiled HEF into hailo-ollama's model store |
| [genai_generate.py](genai_generate.py) + [genai_worker.py](genai_worker.py) | Generate through the `genai.LLM` Python API with retry/timeout protection |
| [diagnostics/](diagnostics/README.md) | Low-level probes when something misbehaves |

## Typical flow

```bash
# 1. register the HEF compiled by pipeline/s6
python register_hailo_ollama.py --hef workdir/model.hef \
    --family tinystories25m --tag my-model

# 2a. serve it like an official model (manifests are scanned at startup)
hailo-ollama
# then call its Ollama-compatible REST API, e.g. to pull/select the model:
curl --silent http://localhost:8000/api/pull \
    -H 'Content-Type: application/json' \
    -d '{"model": "tinystories25m:my-model", "stream": true}'

# 2b. or exercise the genai API directly
python genai_generate.py --hef workdir/model.hef --prompt "Once upon a time"
```

## Which entry point should I use?

- **hailo-ollama** is the end goal: full server semantics (sampling params,
  chat template, stop tokens all come from the HEF's embedded
  `hailo-config.json`).
- **genai_generate.py** talks to the same underlying stack minus the HTTP
  layer; its subprocess-per-attempt design also survives a known host-side
  HailoRT failure mode that otherwise wedges long-lived processes.
- **diagnostics/** bypasses genai entirely and drives raw network groups —
  use it to attribute a problem to the model, the runtime contract, or the
  cache mechanism.
