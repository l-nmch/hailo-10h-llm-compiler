# Terminology

The vocabulary of this project, and how the pieces fit together. Hailo's own
documentation introduces these terms; this page records what they mean in
practice for LLM compilation.

## Artifacts

| Term | What it is |
|---|---|
| **ONNX** | Interchange graph format produced by step 1 (PyTorch export). |
| **HAR** | *Hailo Archive* — a tarball holding the graph representation (`.hn` JSON files) plus metadata/quantization state. It is the DFC's working format: parse produces one, each optimization stage refines it. |
| **`.hn`** | The graph itself inside a HAR: layers as JSON (`conv`, `matmul`, `ew_add`, …) with `input`/`input_shapes`/`params`. Graph surgery = editing this JSON directly (step 3). A HAR may contain several `.hn` variants (main, `.fp.hn`, `.native.hn`). |
| **HEF** | *Hailo Executable Format* — the compiled binary the device runs. Contains instruction blobs (CCWs), network group descriptors and — for LLM HEFs — embedded external resources (embedding table, tokenizer, RoPE table, `hailo-config.json`). |
| **CCW** | Control-command words: the compiled instruction stream of a network context inside the HEF. |

## Scopes and network groups

| Term | Meaning |
|---|---|
| **Scope** | A named subgraph. The pipeline parses everything under one base scope (`ts25mpipe` here). |
| **Network group** | A schedulable unit in the final HEF. The compile script explicitly declares `<scope>__prefill` and `<scope>__tbt` groups. |
| **`__prefill`** | The duplicated graph instance that processes an entire prompt block at once (here 16 positions). |
| **`__tbt`** | *Token by token*: the duplicated graph instance processing one token per run, reading/writing the KV-cache. |
| **KV-cache** | On-device storage of past key/value tensors. `set_kv_cache_global_params(prefill_size, cache_size)` tells DFC to duplicate the graph into prefill/tbt scopes wired to that cache. |

The duplication is why quantization-time decisions (fusing, precision of
shared inputs) affect two graph instances at once.

## Model inputs (the six-input contract)

Every LLM HEF served by genai exposes exactly these inputs (suffixes
declared in `hailo-config.json → input_layers_names_suffixes`):

| Input | Content | Shape (NHWC convention) |
|---|---|---|
| `input_layer1` | embedding codes for the token(s), uint16 on the wire | `[1, seq, hidden]` |
| `input_layer2` | additive attention mask, head-tiled host-side; uint8 raw (255 allowed / 0 blocked) | `[1, rows, n_heads*cache_size]` |
| `input_layer3` / `4` | RoPE cos for K heads / Q heads, float32 | `[1, seq, theta_size*n_kv_heads]` / `[…*n_heads]` |
| `input_layer5` / `6` | RoPE sin for K / Q | same widths as 3 / 4 |

The asymmetry between K and Q widths is real — the runtime writes tiled
buffers, not uniform ones (see [findings/rope-input-widths.md](findings/rope-input-widths.md)).

## Quantization terms

| Term | Meaning |
|---|---|
| **a8_w4 / a16_w16** | Activation/weight bit widths. Convs run INT8 activations over INT4 weights; embeddings stay a16_w16 because the runtime reads them as uint16 codes. |
| **flavor** | DFC's compression/accuracy policy bundle: `compression_level=4` (aggressive INT4), `optimization_level=0` (no implicit adaround/bias_correction/finetune — any higher level re-enables them silently). |
| **ew_add_fusing** | Pre-quantization optimization fusing residual adds into convs. Official LLM recipes disable it; leaving it enabled misaligns the duplicated scopes (see [findings/quantization-recipe.md](findings/quantization-recipe.md)). |
| **saitama** | DFC's PyTorch-based optimization engine (`use_saitama=True`), used instead of the default TensorFlow engine — this is what makes AMD GPUs usable and avoids CUDA-only code paths. |
| **calibset / calibration** | Sample batch fed through the graph to collect activation ranges. RoPE inputs receive raw integer positions; DFC derives cos/sin internally via its `conversion_type` mechanism. |

## Runtime stack

| Term | Meaning |
|---|---|
| **HailoRT** | User-space driver library + CLI (`hailortcli`) talking to the device over PCIe. |
| **genai.LLM** | High-level Python/C++ API wrapping an LLM HEF: handles tokenization (embedded tokenizer.json), mask/RoPE construction, cache bookkeeping, sampling (embedded `hailo-config.json` params). |
| **hailo-ollama** | HTTP server (ollama-compatible API) serving registered LLM HEFs; models live in a content-addressed blob store + manifests. |
| **InferModel** | Low-level HailoRT API beneath genai: you provide every input buffer yourself. Used by all diagnostics tools. |

## Fidelity metric

All numerical comparisons in this project quote a flat **cosine similarity**
(float64) between flattened activation vectors against the float32 Hugging
Face reference. Cosine ≈ 1.0 means numerically faithful; it says nothing
about absolute scale, which is fine for detecting structural divergence.
