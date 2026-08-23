# Compile pipeline

Six self-contained steps taking a Hugging Face checkpoint to a servable HEF.
Each step is a plain CLI script that reads/writes only the working directory
(`$DFC_WORKDIR`, default `./workdir`) and validates its own output where
possible.

| Step | Script | Input → Output | Validation performed |
|---|---|---|---|
| 1 | [s1_export_onnx.py](s1_export_onnx.py) | HF checkpoint → `model.onnx` + reference tensors | cosine vs float32 HF (PyTorch reimpl + ONNX) |
| 2 | [s2_parse_har.py](s2_parse_har.py) | `model.onnx` → `parsed.har` | cosine in SDK_NATIVE context |
| 3 | [s3_surgery_and_resources.py](s3_surgery_and_resources.py) | `parsed.har` → `resources.har` (+ hailo-config.json) | cosine in SDK_NATIVE context post-surgery |
| 4 | [s4_optimize_kvcache.py](s4_optimize_kvcache.py) | `resources.har` → `quantized.har` | none available — see note below |
| 5 | [s5_fix_convolutions.py](s5_fix_convolutions.py) | `quantized.har` → `convfixed.har` | structural (repairs conv input order if needed) |
| 6 | [s6_compile_hef.py](s6_compile_hef.py) | `convfixed.har` → `model.hef` | on-device testing (see ../runtime/) |

```bash
export DFC_WORKDIR=/workdir          # or pass --workdir to each script
python s1_export_onnx.py && \
python s2_parse_har.py  && \
python s3_surgery_and_resources.py && \
python s4_optimize_kvcache.py && \
python s5_fix_convolutions.py && \
python s6_compile_hef.py
```

## Requirements

- Run inside one of the toolchain images from [../docker/](../docker/)
  (`NPY_PROMOTION_STATE=legacy` is baked in there; [config.py](config.py)
  also sets it, but only if not already overridden).
- Steps 1–3 are CPU-only. Step 4 uses the GPU through DFC's PyTorch
  optimization engine (`use_saitama=True`); step 6 compiles on CPU.
- First run of step 1 downloads the model from the Hugging Face Hub.

## Notes

- **No cosine after step 4.** The SDK's quantized emulator is structurally
  broken on KV-cache graphs ([../docs/findings/sdk-behavior-notes.md](../docs/findings/sdk-behavior-notes.md));
  hardware is the only judge for quantized behavior.
- **Adapting another model**: edit the constants at the top of
  [config.py](config.py) (architecture + sequence geometry). The export in
  step 1 assumes a LLaMA2-style block (RMSNorm/SwiGLU/RoPE/GQA); other
  architectures need their own exporter, steps 2–6 are unchanged.
- **The five runtime-contract fixes** applied across steps 1 and 3 are each
  documented with evidence in [../docs/findings/](../docs/findings/index.md).
