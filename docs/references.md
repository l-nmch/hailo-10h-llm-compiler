# References

Every external resource this project relied on or learned from.

## Model

- [Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA](https://huggingface.co/Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA)
  — the reference model this pipeline was validated on (LLaMA2-style, GQA,
  RMSNorm/SwiGLU/RoPE). [Mxode](https://huggingface.co/Mxode) publishes a
  whole family of tiny LLaMA2 models; any of them should work after
  adjusting `pipeline/config.py`.
- [TinyStories: Towards Small Stories That Can Generate Stories](https://arxiv.org/abs/2305.07759)
  — the paper behind the TinyStories dataset these models are trained on.

## Hailo software and documentation

<a id="hailo-software"></a>
- [Hailo Developer Zone](https://hailo.ai/developer-zone/) — account,
  documentation, and the **Hailo Software Suite** downloads (Dataflow
  Compiler, HailoRT, drivers, hailo-ollama).
- [Hailo Dataflow Compiler documentation](https://hailo.ai/developer-zone/documentation/dataflow-compiler/)
  — the official DFC docs; the LLM flow (`set_kv_cache_global_params`,
  network groups) is only lightly covered there.
- ["Bringing Generative AI to the Edge"](https://hailo.ai/blog/bringing-generative-ai-to-the-edge-hailo-10h-llm-compiler/)
  — Hailo's blog series introducing the Hailo-10H LLM compiler; the best
  public overview of the prefill/tbt architecture this repo compiles for.
- [hailo-ai/hailort](https://github.com/hailo-ai/hailort) — public HailoRT
  source (MIT core). Invaluable when debugging the runtime contract:
  `libhailort/src/hef/` documents the HEF binary format, the LLM server
  sources document which `hailo-config.json` keys are actually read.
- [hailo-ai/meta-hailo](https://github.com/hailo-ai/meta-hailo) — Yocto
  layers; a second distribution channel for driver + HailoRT.
- [hailo-ai/hailort-drivers](https://github.com/hailo-ai/hailort-drivers) —
  public source of the `hailo_pci` kernel driver (GPL-2.0); the device
  firmware is built together with it.
- [hailo-ai/hailo_model_zoo_genai](https://github.com/hailo-ai/hailo_model_zoo_genai)
  — MIT-licensed public source of **hailo-ollama** (the Ollama-compatible
  REST server used for first-contact serving tests in this repo), plus the
  official HEF download manifests.
- [hailo-ai/hailo-apps](https://github.com/hailo-ai/hailo-apps) —
  MIT-licensed Hailo applications; its LICENSE covers genai runtime
  components alongside the hailo_model_zoo_genai one.
- [hailo-ai/hailo_model_zoo](https://github.com/hailo-ai/hailo_model_zoo)
  — MIT-licensed model zoo (mostly vision; useful as DFC usage reference).
- [hailo-ai/tappas](https://github.com/hailo-ai/tappas) — Hailo application
  post-processing library (LGPL-2.1); context for the ecosystem.

License breakdown of the above: HailoRT core is MIT, its GStreamer plugin
is LGPL-2.1, the kernel driver is GPL-2.0, meta-hailo is MIT, and the
hailo-ollama / genai-runtime sources are published under MIT licenses
([hailo_model_zoo_genai](https://github.com/hailo-ai/hailo_model_zoo_genai/blob/main/LICENSE),
[hailo-apps](https://github.com/hailo-ai/hailo-apps/blob/main/LICENSE));
the Dataflow Compiler wheel and the device firmware remain proprietary —
neither of those is redistributed by this repository.

## Tooling used by the Docker images

- [mixa3607/pytorch-gfx906](https://hub.docker.com/r/mixa3607/pytorch-gfx906)
  — community ROCm PyTorch base images enabling AMD-GPU quantization
  (`Dockerfile.amd` builds on it).
- [NVIDIA CUDA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  — what makes `--gpus all` work for `Dockerfile.nvidia`.

## Background reading

- [ONNX opset 17](https://onnxruntime.ai/docs/reference/compatibility.html)
  — export target used by step 1 (legacy TorchScript exporter,
  `do_constant_folding=False`).
- [Hugging Face transformers](https://github.com/huggingface/transformers)
  — checkpoint/tokenizer loading in steps 1 and 4.
